
#!/usr/bin/env python3
"""
inference.pyâ LLM agent loop for DistTrainEnv.

Usage:
    HF_TOKEN=gsk_... python inference.py
    HF_TOKEN=gsk_... python inference.py --task easy
    python inference.py --dry-run
"""

import os
import sys
import json
import argparse
from typing import Optional, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import DistTrainEnv
from environment.models import Action
from graders.run_graders import run_grader, run_all_graders

# read from environment variables injected by HF Spaces
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_NAME     = "dist-train-env"

MAX_STEPS_PER_TASK = 15
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an AI operations agent managing a distributed ML training cluster.
The cluster uses ring all-reduce to synchronize gradients across 8 worker nodes (node_0 to node_7).

Each step you receive a JSON observation and must return a JSON action.
Your goal: detect faults, recover the cluster, and keep training healthy.

AVAILABLE ACTIONS:
- restart_node: bring a crashed node back online (requires target_node)
- remove_from_ring: remove a slow/faulty node from the all-reduce ring (requires target_node)
- reduce_batch: reduce batch size on a node showing memory pressure (requires target_node)
- checkpoint: save training checkpoint (use sparingly â halves throughput this step)
- inspect: get detailed diagnostics on a node (requires target_node)
- noop: do nothing this step

RESPONSE FORMAT (JSON only, no explanation):
{"action_type": "<action>", "target_node": "<node_id or null>"}

KEY SIGNALS TO WATCH:
- node status: healthy / slow / oom / crashed
- node memory: climbing above 0.75 = early OOM warning
- gradient_staleness: above 0.3 = training degrading
- loss vs expected_loss: diverging = serious problem
- alerts: read these carefully â they name problematic nodes

STRATEGY:
- Crashed node -> restart_node immediately
- Slow node with high latency -> remove_from_ring
- Node with climbing memory -> reduce_batch EARLY (before it crashes)
- If you see a symptom (slow node) AND another node has high memory, fix the memory node first
- Do NOT restart healthy nodes"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def observation_to_prompt(obs_dict: dict, step: int, task_id: str) -> str:
    """format observation as a compact prompt for the LLM."""
    nodes_summary = []
    for node in obs_dict.get("nodes", []):
        if node["status"] != "healthy" or node["memory"] > 0.65:
            nodes_summary.append(node)
        else:
            nodes_summary.append({
                "id": node["id"],
                "status": "healthy",
                "memory": round(node["memory"], 2),
                "throughput": round(node["throughput"], 2),
            })
    job = obs_dict.get("job", {})
    compact_obs = {
        "step": step,
        "task": task_id,
        "nodes": nodes_summary,
        "job": {
            "loss": round(job.get("loss", 0), 4),
            "expected_loss": round(job.get("expected_loss", 0), 4),
            "cluster_throughput": round(job.get("cluster_throughput", 0), 3),
            "gradient_staleness": round(job.get("gradient_staleness", 0), 3),
            "loss_diverging": job.get("loss_diverging", False),
        },
        "ring_order": obs_dict.get("ring_order", []),
        "alerts": obs_dict.get("alerts", []),
    }
    return f"Observation (step {step}):\n{json.dumps(compact_obs, indent=2)}"


def parse_action(response_text: str) -> Action:
    """parse LLM response into an Action. falls back to noop on error."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        data = json.loads(text.strip())
        return Action(
            action_type=data.get("action_type", "noop"),
            target_node=data.get("target_node"),
            parameters=data.get("parameters"),
        )
    except Exception:
        return Action(action_type="noop")


def action_to_str(action: Action) -> str:
    """compact action string for [STEP] log."""
    if action.target_node:
        return f"{action.action_type}('{action.target_node}')"
    return action.action_type


def _rule_based_action(obs_dict: dict) -> Action:
    """simple rule-based agent for dry runs."""
    nodes = obs_dict.get("nodes", [])
    for node in nodes:
        if node["status"] == "crashed":
            return Action(action_type="restart_node", target_node=node["id"])
    for node in nodes:
        if node["memory"] > 0.75 and node["status"] in ["slow", "oom"]:
            return Action(action_type="reduce_batch", target_node=node["id"])
    for node in nodes:
        if node["status"] == "slow" and node.get("in_ring", True):
            return Action(action_type="remove_from_ring", target_node=node["id"])
    return Action(action_type="noop")


def run_task(client: Optional[OpenAI], task_id: str, dry_run: bool = False) -> dict:
    """run one full episode and emit [START] [STEP] [END] logs."""
    env = DistTrainEnv(task_id=task_id)
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    episode_history = []
    rewards: List[float] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    for step_num in range(1, MAX_STEPS_PER_TASK + 1):
        user_content = observation_to_prompt(obs_dict, step_num, task_id)
        messages.append({"role": "user", "content": user_content})

        error_str = None
        if dry_run:
            action = _rule_based_action(obs_dict)
        else:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.0,
                )
                response_text = response.choices[0].message.content
                messages.append({"role": "assistant", "content": response_text})
                action = parse_action(response_text)
            except Exception as e:
                error_str = str(e)[:80]
                action = Action(action_type="noop")

        result = env.step(action)
        new_obs_dict = result.observation.model_dump()
        reward = result.reward.value
        done = result.done
        rewards.append(reward)

        log_step(step=step_num, action=action_to_str(action), reward=reward, done=done, error=error_str)

        episode_history.append({
            "step": step_num,
            "observation": obs_dict,
            "action": action.model_dump(),
            "reward": result.reward.model_dump(),
            "done": done,
        })
        obs_dict = new_obs_dict

        if done:
            break

    score = run_grader(task_id, episode_history)
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=len(episode_history), score=score, rewards=rewards)
    return {"history": episode_history, "score": score}


def main():
    parser = argparse.ArgumentParser(description="DistTrainEnv inference agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    client = None

    if not dry_run:
        if not API_KEY:
            print("ERROR: HF_TOKEN not set. use --dry-run for testing.")
            sys.exit(1)
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    all_results = {}

    for task_id in tasks_to_run:
        result = run_task(client, task_id, dry_run=dry_run)
        all_results[task_id] = result["history"]

    summary = run_all_graders(all_results)
    print(f"\n{'='*55}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*55}", flush=True)
    for task_id, score in summary["scores"].items():
        print(f"  {task_id:8s}: {score:.4f}", flush=True)
    print(f"  {'WEIGHTED':8s}: {summary['weighted_total']:.4f}", flush=True)


if __name__ == "__main__":
    main()
