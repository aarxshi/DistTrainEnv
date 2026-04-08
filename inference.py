# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from typing import Optional, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import DistTrainEnv
from environment.models import Action

# ----------------------------------------------------------
# Required env vars per submission spec: HF_TOKEN, API_BASE_URL, MODEL_NAME
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")  # default per spec requirement
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_NAME = "dist-train-env"

# ----------------------------------------------------------
MAX_STEPS_PER_TASK = 15
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

# ----------------------------------------------------------
SYSTEM_PROMPT = """You are an AI ops agent managing a distributed ML cluster.

Return ONLY JSON:
{"action_type": "<action>", "target_node": "<node_id or null>"}

Valid actions:
restart_node, remove_from_ring, reduce_batch, checkpoint, inspect, noop

STRICT RULES - act immediately, never noop when a fault exists:
- If any node status == "crashed" → restart_node on that node
- If any node status == "oom" → reduce_batch on that node
- If any node status == "slow" or "straggler" → remove_from_ring on that node
- If any node memory > 0.75 → reduce_batch on that node
- Only use noop if ALL nodes are healthy and memory < 0.75
- Always pick the most critical node (crashed > oom > slow > high memory)
"""

# ----------------------------------------------------------
# LOGGING — must match spec exactly
# ----------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # FIX: score uses 2 decimal places to match the spec example (score=1.00)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ----------------------------------------------------------

def parse_action(response_text: str) -> Action:
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        data = json.loads(text)
        return Action(
            action_type=data.get("action_type", "noop"),
            target_node=data.get("target_node"),
        )
    except Exception:
        return Action(action_type="noop")

def action_to_str(action: Action) -> str:
    if action.target_node:
        return f"{action.action_type}('{action.target_node}')"
    return action.action_type

# ----------------------------------------------------------

def _get_urgent_action(obs_dict: dict) -> Optional[Action]:
    nodes = obs_dict.get("nodes", [])

    # Priority 1: crashed
    for node in nodes:
        if node["status"] == "crashed":
            return Action(action_type="restart_node", target_node=node["id"])

    # Priority 2: oom status OR high memory (>0.4) and in ring
    for node in nodes:
        if node["status"] == "oom":
            return Action(action_type="reduce_batch", target_node=node["id"])
    for node in nodes:
        if node.get("memory", 0) > 0.4 and node.get("in_ring", False):
            return Action(action_type="reduce_batch", target_node=node["id"])

    # Priority 3: slow/straggler AND still in ring
    for node in nodes:
        if node["status"] in ("slow", "straggler") and node.get("in_ring", True):
            return Action(action_type="remove_from_ring", target_node=node["id"])

    # Priority 4: high memory (>0.35) as early warning
    for node in nodes:
        if node.get("memory", 0) > 0.35 and node.get("in_ring", False):
            return Action(action_type="reduce_batch", target_node=node["id"])

    return None

def _rule_based_action(obs_dict: dict) -> Action:
    urgent = _get_urgent_action(obs_dict)
    if urgent:
        return urgent
    return Action(action_type="noop")

# ----------------------------------------------------------

def compute_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    avg = sum(rewards) / len(rewards)
    return min(1.0, max(0.0, (avg - 0.4) / 0.8))

# ----------------------------------------------------------

def run_task(client: Optional[OpenAI], task_id: str, dry_run: bool) -> dict:
    env = DistTrainEnv(task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_id, ENV_NAME, MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id)
        obs_dict = obs.model_dump()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            error_str = None

            # -------- ACTION --------
            if dry_run:
                action = _rule_based_action(obs_dict)
            else:
                # Always try rule-based first for obvious faults
                urgent = _get_urgent_action(obs_dict)
                if urgent:
                    action = urgent
                else:
                    try:
                        messages.append({
                            "role": "user",
                            "content": json.dumps(obs_dict)
                        })

                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            max_tokens=100,
                            temperature=0.0,
                        )

                        text = response.choices[0].message.content or ""
                        messages.append({"role": "assistant", "content": text})
                        action = parse_action(text)

                    except Exception as e:
                        error_str = str(e)[:80]
                        action = Action(action_type="noop")

            # -------- ENV STEP --------
            result = env.step(action)

            obs_dict = result.observation.model_dump()
            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_to_str(action), reward, done, error_str)

            if done:
                break

        score = compute_score(rewards)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {str(e)[:100]}", flush=True)

    finally:
        try:
            env.close()
        except Exception:
            pass
        # FIX: log_end is ONLY called here (inside run_task), never duplicated in main()
        log_end(success, steps_taken, score, rewards)

    return {"score": score}

# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    client = None

    if not dry_run:
        # Use top-level vars read from HF_TOKEN / API_BASE_URL / MODEL_NAME as required by spec
        if API_KEY and API_BASE_URL:
            # Eval environment — uses injected API_BASE_URL and HF_TOKEN
            client = OpenAI(
                api_key=API_KEY,
                base_url=API_BASE_URL,
                timeout=60.0,
            )
        elif API_KEY:
            # Local testing fallback (no API_BASE_URL set)
            client = OpenAI(
                api_key=API_KEY,
                base_url="https://api.groq.com/openai/v1",
                timeout=60.0,
            )
        else:
            print("[DEBUG] No HF_TOKEN found. Falling back to dry-run.", flush=True)
            dry_run = True

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    scores = {}

    for task_id in tasks_to_run:
        try:
            result = run_task(client, task_id, dry_run)
            scores[task_id] = result["score"]
        except Exception as e:
            # FIX: do NOT call log_end here — run_task's finally block already did it.
            # Just record a zero score and move on.
            print(f"[DEBUG] Outer error in task {task_id}: {e}", flush=True)
            scores[task_id] = 0.0

    print("\n=======================================================", flush=True)
    print("FINAL SCORES", flush=True)
    print("=======================================================", flush=True)

    for t, s in scores.items():
        print(f"{t:8s}: {s:.4f}", flush=True)

    weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    total = sum(scores.get(t, 0.0) * weights[t] for t in tasks_to_run)
    print(f"{'WEIGHTED':8s}: {total:.4f}", flush=True)

# ----------------------------------------------------------

if __name__ == "__main__":
    main()