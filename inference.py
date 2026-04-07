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
# Env vars (MANDATORY)
# ----------------------------------------------------------
HF_TOKEN     = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_NAME     = "dist-train-env"

# ----------------------------------------------------------
MAX_STEPS_PER_TASK = 15
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

# ----------------------------------------------------------
SYSTEM_PROMPT = """You are an AI operations agent managing a distributed ML cluster.

Return ONLY JSON:
{"action_type": "<action>", "target_node": "<node_id or null>"}

Valid actions:
restart_node, remove_from_ring, reduce_batch, checkpoint, inspect, noop
"""

# ----------------------------------------------------------
# Logging (STRICT FORMAT)
# ----------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ----------------------------------------------------------

def observation_to_prompt(obs_dict: dict) -> str:
    return json.dumps(obs_dict)

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

# ----------------------------------------------------------

def action_to_str(action: Action) -> str:
    if action.target_node:
        return f"{action.action_type}('{action.target_node}')"
    return action.action_type

# ----------------------------------------------------------

def _rule_based_action(obs_dict: dict) -> Action:
    for node in obs_dict.get("nodes", []):
        if node["status"] == "crashed":
            return Action("restart_node", node["id"])

    for node in obs_dict.get("nodes", []):
        if node["memory"] > 0.75:
            return Action("reduce_batch", node["id"])

    for node in obs_dict.get("nodes", []):
        if node["status"] == "slow":
            return Action("remove_from_ring", node["id"])

    return Action("noop")

# ----------------------------------------------------------

def compute_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    avg = sum(rewards) / len(rewards)
    return min(1.0, max(0.0, (avg - 0.4) / 0.8))

# ----------------------------------------------------------

def run_task(client: Optional[OpenAI], task_id: str, dry_run: bool):
    env = DistTrainEnv(task_id=task_id)

    rewards = []
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
            if dry_run or client is None:
                action = _rule_based_action(obs_dict)
            else:
                try:
                    messages.append({
                        "role": "user",
                        "content": observation_to_prompt(obs_dict)
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
                    action = Action("noop")

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
        # HARD SAFETY NET — prevents crash
        print(f"ERROR: {str(e)[:100]}", flush=True)

    finally:
        try:
            env.close()
        except Exception:
            pass

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

    # -------- SAFE CLIENT INIT --------
    if not dry_run:
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN missing → using dry-run", flush=True)
            dry_run = True
        else:
            try:
                client = OpenAI(
                    api_key=HF_TOKEN,
                    base_url=API_BASE_URL,
                    timeout=60.0,
                )
            except Exception as e:
                print(f"WARNING: OpenAI init failed: {e}", flush=True)
                dry_run = True
                client = None

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    scores = {}

    for task_id in tasks_to_run:
        try:
            result = run_task(client, task_id, dry_run)
            scores[task_id] = result["score"]
        except Exception as e:
            print(f"ERROR in task {task_id}: {e}", flush=True)
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