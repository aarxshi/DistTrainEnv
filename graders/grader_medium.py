
"""
graders/grader_medium.py — Grader for Medium task: Straggler Detection.

Key difference from easy: the straggler is slow, not crashed.
Agent must identify node_5 as the bottleneck from throughput
and latency signals without an explicit crash alert.
"""

from typing import List, Dict, Any


def grade(episode_history: List[Dict[str, Any]]) -> float:
    """
    Grade a completed medium task episode.
    Returns float in [0.0, 1.0].
    """
    if not episode_history:
        return 0.0

    TARGET_THROUGHPUT = 0.80
    MAX_STEPS = 20
    STRAGGLER_NODE = "node_5"

    # --------------------------------------------------------
    # 1. Throughput recovery score (35%)
    # same as easy — average over last third of episode
    # --------------------------------------------------------
    last_third_start = max(0, len(episode_history) * 2 // 3)
    last_steps = episode_history[last_third_start:]
    throughput_values = [
        s["observation"]["job"]["cluster_throughput"] for s in last_steps
    ]
    avg_final_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0.0
    throughput_score = min(1.0, avg_final_throughput / TARGET_THROUGHPUT)

    # --------------------------------------------------------
    # 2. Detection speed score (30%)
    # how quickly did the agent take action on node_5?
    # aNY targeted action on node_5 counts as detection.
    # --------------------------------------------------------
    detection_step = None
    for i, step_data in enumerate(episode_history):
        action = step_data.get("action", {})
        if (
            action.get("action_type") in [
                "remove_from_ring", "restart_node",
                "reduce_batch", "inspect"
            ]
            and action.get("target_node") == STRAGGLER_NODE
        ):
            detection_step = i + 1
            break

    if detection_step is not None:
        # bonus for very early detection (first 5 steps)
        if detection_step <= 5:
            detection_score = 1.0
        else:
            detection_score = max(0.1, 1.0 - (detection_step - 5) / (MAX_STEPS - 5))
    else:
        detection_score = 0.0  # never found the straggler

    # --------------------------------------------------------
    # 3. Loss health score (25%)
    # --------------------------------------------------------
    final_obs = episode_history[-1]["observation"]
    final_loss = final_obs["job"]["loss"]
    final_expected = final_obs["job"]["expected_loss"]

    if final_expected > 0:
        loss_deviation = abs(final_loss - final_expected) / final_expected
        loss_health_score = max(0.0, 1.0 - min(1.0, loss_deviation * 3.0))
    else:
        loss_health_score = 0.0

    # --------------------------------------------------------
    # 4. Ring integrity score (10%)
    # node_5 should NOT be in ring at end (it was a straggler)
    # full score: removed from ring
    # partial score: restarted and healthy (acceptable)
    # zero: still in ring and slow (problem not fixed)
    # --------------------------------------------------------
    final_nodes = {n["id"]: n for n in final_obs.get("nodes", [])}
    straggler_state = final_nodes.get(STRAGGLER_NODE)

    if straggler_state is None:
        ring_score = 0.5
    elif not straggler_state["in_ring"]:
        ring_score = 1.0   # removed — correct resolution for straggler
    elif straggler_state["status"] == "healthy" and straggler_state["throughput"] > 0.8:
        ring_score = 0.8   # recovered in ring — acceptable
    elif straggler_state["status"] == "slow":
        ring_score = 0.0   # still slow — not fixed
    else:
        ring_score = 0.3

    # --------------------------------------------------------
    # weighted final score
    # --------------------------------------------------------
    score = (
        0.35 * throughput_score
        + 0.30 * detection_score
        + 0.25 * loss_health_score
        + 0.10 * ring_score
    )

    return round(min(1.0, max(0.0, score)), 4)


def grade_with_breakdown(episode_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same as grade() but returns full breakdown for debugging."""
    if not episode_history:
        return {"score": 0.0, "error": "empty episode"}

    TARGET_THROUGHPUT = 0.80
    MAX_STEPS = 20
    STRAGGLER_NODE = "node_5"

    last_third_start = max(0, len(episode_history) * 2 // 3)
    last_steps = episode_history[last_third_start:]
    throughput_values = [s["observation"]["job"]["cluster_throughput"] for s in last_steps]
    avg_final_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0.0
    throughput_score = min(1.0, avg_final_throughput / TARGET_THROUGHPUT)

    detection_step = None
    for i, step_data in enumerate(episode_history):
        action = step_data.get("action", {})
        if action.get("action_type") in ["remove_from_ring", "restart_node", "reduce_batch", "inspect"]                 and action.get("target_node") == STRAGGLER_NODE:
            detection_step = i + 1
            break
    if detection_step is not None:
        detection_score = 1.0 if detection_step <= 5 else max(0.1, 1.0 - (detection_step - 5) / (MAX_STEPS - 5))
    else:
        detection_score = 0.0

    final_obs = episode_history[-1]["observation"]
    final_loss = final_obs["job"]["loss"]
    final_expected = final_obs["job"]["expected_loss"]
    loss_deviation = abs(final_loss - final_expected) / final_expected if final_expected > 0 else 1.0
    loss_health_score = max(0.0, 1.0 - min(1.0, loss_deviation * 3.0))

    final_nodes = {n["id"]: n for n in final_obs.get("nodes", [])}
    straggler_state = final_nodes.get(STRAGGLER_NODE)
    if straggler_state is None: ring_score = 0.5
    elif not straggler_state["in_ring"]: ring_score = 1.0
    elif straggler_state["status"] == "healthy" and straggler_state["throughput"] > 0.8: ring_score = 0.8
    elif straggler_state["status"] == "slow": ring_score = 0.0
    else: ring_score = 0.3

    final_score = round(min(1.0, max(0.0,
        0.35 * throughput_score + 0.30 * detection_score
        + 0.25 * loss_health_score + 0.10 * ring_score
    )), 4)

    return {
        "score": final_score,
        "throughput_score": round(throughput_score, 4),
        "detection_score": round(detection_score, 4),
        "loss_health_score": round(loss_health_score, 4),
        "ring_score": round(ring_score, 4),
        "detection_step": detection_step,
        "avg_final_throughput": round(avg_final_throughput, 4),
        "loss_deviation_pct": round(loss_deviation * 100, 2),
        "total_steps": len(episode_history),
    }
