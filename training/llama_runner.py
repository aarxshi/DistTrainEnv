import json, os, re, sys, time
from dataclasses import dataclass
from typing import List, Optional
import requests

try:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    REPO_ROOT = os.path.abspath(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment.env import DistTrainEnv
from environment.models import Action

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


@dataclass
class RunnerConfig:
    task: str = "easy"
    num_episodes: int = 10
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_tokens: int = 300
    temperature: float = 0.2
    wandb_project: str = "DistTrainEnv"
    wandb_run_name: str = ""
    save_dir: str = "checkpoints"
    hf_token: str = ""
    api_retries: int = 3
    retry_delay: float = 2.0
    step_delay: float = 0.25


SYSTEM_PROMPTS = {
"easy": (
    "You are an expert distributed systems engineer managing an 8-node ring all-reduce training cluster.\n\n"
    "ENVIRONMENT FACTS:\n"
    "- Ring throughput = speed of the SLOWEST node.\n"
    "- Node statuses: healthy -> slow -> oom -> crashed\n"
    "- crashed nodes are OUT of the ring and contribute nothing.\n\n"
    "YOUR TASK: EASY - Node Crash Recovery\n"
    "- node_3 will CRASH at step 1.\n"
    "- You MUST use restart_node on node_3 to bring it back.\n"
    "- Do NOT use remove_from_ring on a crashed node.\n\n"
    "DECISION TREE:\n"
    "1. If ANY node status == crashed -> restart_node immediately.\n"
    "2. If ANY node status == oom    -> reduce_batch.\n"
    "3. If ANY node status == slow   -> remove_from_ring.\n"
    "4. If cluster healthy           -> noop.\n\n"
    "Respond ONLY with JSON: "
    '{"action_type": "restart_node", "target_node": "node_3", "reasoning": "brief reason"}'
),
"medium": (
    "You are an expert distributed systems engineer managing an 8-node ring all-reduce training cluster.\n\n"
    "ENVIRONMENT FACTS:\n"
    "- Ring throughput = speed of the SLOWEST node.\n"
    "- A slow node throttles throughput but does NOT trigger a crash alert.\n"
    "- There is ONE false alarm node: high memory but status=healthy. Do NOT restart it.\n\n"
    "YOUR TASK: MEDIUM - Straggler Detection\n"
    "- node_5 runs at 30% speed from step 1 (status=slow).\n"
    "- Correct fix: remove_from_ring on node_5.\n"
    "- Restarting a healthy node = -0.5 penalty.\n\n"
    "DECISION TREE:\n"
    "1. Find node with status==slow and lowest throughput -> remove_from_ring.\n"
    "2. If status==oom     -> reduce_batch.\n"
    "3. If status==crashed -> restart_node.\n"
    "4. If status==healthy (even high memory) -> noop.\n\n"
    "Respond ONLY with JSON: "
    '{"action_type": "remove_from_ring", "target_node": "node_5", "reasoning": "brief reason"}'
),
"hard": (
    "You are an expert distributed systems engineer managing an 8-node ring all-reduce training cluster.\n\n"
    "YOUR TASK: HARD - Cascading OOM Fault Recovery (3 phases, 120 steps)\n\n"
    "FAULT CHAIN:\n"
    "- node_2 memory climbs ~6% per step from step 1 (SILENT - no crash alert initially).\n"
    "- At step 5: node_7 becomes slow (downstream SYMPTOM of node_2).\n"
    "- Gradient staleness builds, loss diverges.\n\n"
    "ROOT CAUSE = node_2. SYMPTOM = node_7.\n"
    "Fixing node_7 only = 0.30 score. Fixing node_2 early = 1.0 score.\n\n"
    "ACTION PLAN:\n"
    "1. Every step: find node with HIGHEST memory.\n"
    "2. If any node memory > 0.65 -> reduce_batch on that node.\n"
    "3. If any node memory > 0.88 -> restart_node (crash imminent).\n"
    "4. Use checkpoint ONCE in steps 2-10.\n"
    "5. Only after fixing node_2: if node_7 slow -> remove_from_ring.\n"
    "6. Phase 3 (steps 91-120): noop when healthy.\n\n"
    "NEVER target node_7 as your first action.\n"
    "NEVER restart healthy-status nodes.\n\n"
    "Respond ONLY with JSON: "
    '{"action_type": "reduce_batch", "target_node": "node_2", "reasoning": "brief reason"}'
),
}


def build_user_prompt(obs, history, task):
    nodes = obs.get("nodes", [])
    job = obs.get("job", {})
    step = obs.get("step", 0)
    phase = obs.get("current_phase", 1)
    sorted_nodes = sorted(nodes, key=lambda n: n.get("memory", 0), reverse=True)
    node_lines = []
    for n in sorted_nodes:
        flags = ""
        if n.get("status") == "crashed":   flags += " CRASHED"
        elif n.get("status") == "oom":     flags += " OOM"
        elif n.get("status") == "slow":    flags += " SLOW"
        if n.get("memory", 0) > 0.65:     flags += " HIGH-MEM"
        if not n.get("in_ring"):           flags += " [OUT-OF-RING]"
        node_lines.append(
            "  " + n["id"] + ": status=" + n.get("status","") + flags +
            " | mem=" + "{:.0%}".format(n.get("memory",0)) +
            " | thr=" + "{:.0%}".format(n.get("throughput",0))
        )
    hist_lines = [
        "  step " + str(h["step"]) + ": " + h["action_type"] +
        " " + str(h["target_node"]) + " r=" + "{:+.3f}".format(h["reward"])
        for h in history[-5:]
    ] or ["  (none)"]
    hints = []
    if task == "easy":
        crashed = [n["id"] for n in nodes if n.get("status") == "crashed"]
        if crashed:
            hints.append("REQUIRED: " + str(crashed[0]) + " is CRASHED -> restart_node")
    elif task == "medium":
        slow = [(n["id"], n.get("throughput",1)) for n in nodes if n.get("status") == "slow"]
        if slow:
            w = min(slow, key=lambda x: x[1])
            hints.append("STRAGGLER: " + w[0] + " at " + "{:.0%}".format(w[1]) + " -> remove_from_ring")
        fa = [n["id"] for n in nodes if n.get("status") == "healthy" and n.get("memory",0) > 0.72]
        if fa:
            hints.append("FALSE ALARM: " + str(fa) + " healthy despite high memory - do NOT restart")
    elif task == "hard":
        hm = sorted([(n["id"], n.get("memory",0)) for n in nodes if n.get("memory",0) > 0.55],
                    key=lambda x: x[1], reverse=True)
        if hm:
            top_id, top_mem = hm[0]
            if top_mem > 0.88:
                hints.append("CRITICAL: " + top_id + " mem=" + "{:.0%}".format(top_mem) + " -> restart_node NOW")
            elif top_mem > 0.70:
                hints.append("ROOT CAUSE: " + top_id + " mem=" + "{:.0%}".format(top_mem) + " -> reduce_batch NOW")
            else:
                hints.append("EARLY WARNING: " + top_id + " mem=" + "{:.0%}".format(top_mem) + " -> reduce_batch")
        slow = [n["id"] for n in nodes if n.get("status") == "slow"]
        if slow and hm:
            hints.append("Slow nodes " + str(slow) + " are SYMPTOMS - fix OOM first")
        if obs.get("job",{}).get("gradient_staleness",0) > 0.5:
            hints.append("HIGH STALENESS - fix slow/oom nodes urgently")
    if obs.get("unnecessary_restarts", 0) > 0:
        hints.append("WARNING: unnecessary restarts detected - do NOT restart healthy nodes")
    thr = job.get("cluster_throughput", 0)
    staleness = job.get("gradient_staleness", 0)
    return (
        "CLUSTER STATUS Step " + str(step) + " Phase " + str(phase) + "\n\n"
        "NODES (highest memory risk first):\n" +
        "\n".join(node_lines) + "\n\n"
        "JOB: throughput=" + "{:.3f}".format(thr) +
        " loss=" + "{:.4f}".format(job.get("loss",0)) +
        " (expected=" + "{:.4f}".format(job.get("expected_loss",0)) + ")" +
        " staleness=" + "{:.3f}".format(staleness) +
        " diverging=" + str(job.get("loss_diverging", False)) + "\n"
        "ALERTS: " + (", ".join(obs.get("alerts", [])) or "none") + "\n"
        "RING: " + str(obs.get("ring_order", [])) + "\n\n"
        "RECENT ACTIONS:\n" + "\n".join(hist_lines) + "\n\n"
        "ANALYSIS:\n" + ("\n".join(hints) if hints else "Cluster stable.") + "\n\n"
        "Respond with ONLY the JSON action object."
    )


def call_llama(system_prompt, user_prompt, cfg):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": "Bearer " + cfg.hf_token, "Content-Type": "application/json"}
    payload = {
        "model": cfg.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": False,
    }
    for attempt in range(cfg.api_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = cfg.retry_delay * (2 ** attempt)
                print("    [API] Rate limited - waiting " + str(wait) + "s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print("    [API] HTTP " + str(resp.status_code) + ": " + resp.text[:200])
                if attempt < cfg.api_retries - 1:
                    time.sleep(cfg.retry_delay)
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # strip code fences without using backtick literals
            fence = chr(96) * 3
            if content.startswith(fence):
                content = content[content.index("\n")+1:] if "\n" in content else content[3:]
            if content.endswith(fence):
                content = content[:-3]
            content = content.strip()
            m = re.search(r"\{[^{}]+\}", content, re.DOTALL)
            if m:
                content = m.group(0)
            return json.loads(content)
        except requests.exceptions.Timeout:
            print("    [API] Timeout attempt " + str(attempt+1))
            if attempt < cfg.api_retries - 1:
                time.sleep(cfg.retry_delay)
        except json.JSONDecodeError as e:
            print("    [API] JSON error: " + str(e))
            if attempt < cfg.api_retries - 1:
                time.sleep(cfg.retry_delay)
        except Exception as e:
            print("    [API] Error: " + str(e))
            if attempt < cfg.api_retries - 1:
                time.sleep(cfg.retry_delay)
    return None


VALID_ACTIONS = {"restart_node", "remove_from_ring", "reduce_batch", "checkpoint", "inspect", "noop"}

def parse_action(raw, obs):
    if raw is not None:
        at = raw.get("action_type", "").strip()
        if at in VALID_ACTIONS:
            t = raw.get("target_node")
            if isinstance(t, str) and t.lower() in ("null", "none", ""):
                t = None
            return Action(action_type=at, target_node=t)
    nodes = {n["id"]: n for n in obs.get("nodes", [])}
    crashed = [nid for nid, n in nodes.items() if n.get("status") == "crashed"]
    if crashed:
        return Action(action_type="restart_node", target_node=crashed[0])
    ooms = [nid for nid, n in nodes.items() if n.get("status") == "oom"]
    if ooms:
        return Action(action_type="reduce_batch", target_node=ooms[0])
    slow = [nid for nid, n in nodes.items() if n.get("status") == "slow" and n.get("in_ring")]
    if slow:
        return Action(action_type="remove_from_ring", target_node=slow[0])
    return Action(action_type="noop")


def run_episode(env, cfg, episode_num, system_prompt):
    obs = env.reset()
    obs_dict = obs.dict()
    total_reward = 0.0
    step_rewards, throughputs, loss_healths = [], [], []
    causal_fixes = api_failures = step_count = 0
    action_history = []
    print("\n  Episode " + str(episode_num) + " | task=" + cfg.task.upper())
    while True:
        user_prompt = build_user_prompt(obs_dict, action_history, cfg.task)
        raw = call_llama(system_prompt, user_prompt, cfg)
        action = parse_action(raw, obs_dict)
        if raw is None:
            api_failures += 1
        reasoning = (raw or {}).get("reasoning", "")
        result = env.step(action)
        obs_dict = result.observation.dict()
        reward, done, info = result.reward, result.done, result.info
        total_reward += reward.value
        step_rewards.append(reward.value)
        throughputs.append(reward.throughput_score)
        loss_healths.append(reward.loss_health_score)
        if reward.causal_fix_bonus > 0.4:
            causal_fixes += 1
        step_count += 1
        action_history.append({
            "step": info["step"], "action_type": action.action_type,
            "target_node": str(action.target_node), "reward": reward.value,
            "throughput": reward.throughput_score,
        })
        icon = "OK" if reward.value > 0.3 else "!!"
        print("  [" + icon + "] step=" + str(info["step"]) +
              " | " + action.action_type + " " + str(action.target_node) +
              " | r=" + "{:+.3f}".format(reward.value) +
              " | thr=" + "{:.2f}".format(reward.throughput_score) +
              " | causal=" + "{:.2f}".format(reward.causal_fix_bonus) +
              " | pen=" + "{:.2f}".format(reward.penalty))
        if reasoning:
            print("       " + reasoning[:110])
        if _WANDB:
            wandb.log({
                cfg.task + "/step_reward": reward.value,
                cfg.task + "/throughput": reward.throughput_score,
                cfg.task + "/loss_health": reward.loss_health_score,
                cfg.task + "/causal_fix": reward.causal_fix_bonus,
                cfg.task + "/penalty": reward.penalty,
                cfg.task + "/staleness": obs_dict["job"].get("gradient_staleness", 0),
                cfg.task + "/cluster_throughput": obs_dict["job"].get("cluster_throughput", 0),
                "episode": episode_num, "step": info["step"],
            })
        if done:
            break
        time.sleep(cfg.step_delay)
    mean_reward = total_reward / max(1, step_count)
    mean_thr = sum(throughputs) / max(1, len(throughputs))
    mean_lh = sum(loss_healths) / max(1, len(loss_healths))
    root_fixed = info.get("root_cause_fixed", False)
    print("  -> steps=" + str(step_count) +
          " mean_reward=" + "{:+.3f}".format(mean_reward) +
          " root_cause_fixed=" + str(root_fixed))
    stats = {
        "episode": episode_num, "task": cfg.task, "steps": step_count,
        "total_reward": round(total_reward, 4), "mean_reward": round(mean_reward, 4),
        "max_reward": round(max(step_rewards), 4), "min_reward": round(min(step_rewards), 4),
        "mean_throughput": round(mean_thr, 4), "mean_loss_health": round(mean_lh, 4),
        "causal_fix_steps": causal_fixes,
        "false_alarm_restarts": obs_dict.get("unnecessary_restarts", 0),
        "api_failures": api_failures, "root_cause_fixed": root_fixed,
    }
    if _WANDB:
        wandb.log({
            cfg.task + "/ep_total_reward": total_reward,
            cfg.task + "/ep_mean_reward": mean_reward,
            cfg.task + "/ep_mean_throughput": mean_thr,
            cfg.task + "/ep_mean_loss_health": mean_lh,
            cfg.task + "/ep_root_cause_fixed": int(root_fixed),
            cfg.task + "/ep_causal_fix_steps": causal_fixes,
            cfg.task + "/ep_false_alarms": obs_dict.get("unnecessary_restarts", 0),
            "episode": episode_num,
        })
    return stats


def run(cfg):
    if not cfg.hf_token:
        cfg.hf_token = os.environ.get("HF_TOKEN", "")
    if not cfg.hf_token:
        raise ValueError("HF_TOKEN not set.")
    if cfg.task not in SYSTEM_PROMPTS:
        raise ValueError("Unknown task: " + cfg.task)
    system_prompt = SYSTEM_PROMPTS[cfg.task]
    run_name = cfg.wandb_run_name or cfg.task + "_llama70b_" + str(int(time.time()))
    if _WANDB:
        wandb.init(project=cfg.wandb_project, name=run_name,
                   config={"task": cfg.task, "model": cfg.model_id,
                           "num_episodes": cfg.num_episodes, "temperature": cfg.temperature},
                   reinit=True)
    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.save_dir, "metrics_" + cfg.task + ".json")
    env = DistTrainEnv(task_id=cfg.task)
    all_episodes = []
    print("\n" + "="*60)
    print("  Task: " + cfg.task.upper() + " | Episodes: " + str(cfg.num_episodes))
    print("="*60)
    for ep in range(1, cfg.num_episodes + 1):
        stats = run_episode(env, cfg, ep, system_prompt)
        all_episodes.append(stats)
        with open(metrics_path, "w") as f:
            json.dump(all_episodes, f, indent=2)
        print("  Saved -> " + metrics_path)
    overall = sum(e["mean_reward"] for e in all_episodes) / len(all_episodes)
    root_fixed = sum(1 for e in all_episodes if e["root_cause_fixed"])
    print("\n" + "="*60)
    print("  " + cfg.task.upper() + " COMPLETE")
    print("  Mean reward: " + "{:+.3f}".format(overall))
    print("  Root fixed:  " + str(root_fixed) + "/" + str(cfg.num_episodes))
    print("="*60)
    if _WANDB:
        wandb.finish()
    return all_episodes


print("llama_runner loaded OK")
