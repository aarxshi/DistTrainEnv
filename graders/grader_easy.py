
"""
graders/grader_easy.py — Grader for Easy task: Node Crash Recovery.

Scores a completed episode based on:
  - Throughput recovery
  - Loss trajectory adherence
  - Steps to recovery (efficiency)
  - Ring integrity restoration

Always returns a float in [0.0, 1.0]. Deterministic.
"""

from typing import List, Dict, Any


def grade(episode_history: List[Dict[str, Any]]) -> float:
    """
    Grade a completed easy task episode.

    Args:
        episode_history: list of step dicts, each containing:
            {
                "step":          int,
                "observation":   Observation dict,
                "action":        Action dict,
                "reward":        Reward dict,
                "done":          bool,
            }

    Returns:
        float in [0.0, 1.0]
    """
    if not episode_history:
        return 0.0

    TARGET_THROUGHPUT = 0.85
    MAX_STEPS = 15
    FAULT_NODE = "node_3"

    # --------------------------------------------------------
    # 1. Throughput recovery score (40%)
    # average throughput in the LAST THIRD of the episode.
    # rewards sustained recovery, not just one lucky step.
    # --------------------------------------------------------
    last_third_start = max(0, len(episode_history) * 2 // 3)
    last_steps = episode_history[last_third_start:]

    throughput_values = [
        s["observation"]["job"]["cluster_throughput"]
        for s in last_steps
    ]
    avg_final_throughput = (
        sum(throughput_values) / len(throughput_values)
        if throughput_values else 0.0
    )
    throughput_score = min(1.0, avg_final_throughput / TARGET_THROUGHPUT)

    # --------------------------------------------------------
    # 2. Loss health score (30%)
    # check final step: how close is actual loss to expected?
    # score = 1 - clamp(abs_deviation / expected_loss, 0, 1)
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
    # 3. Steps efficiency score (20%)
    # find the step where the crashed node was acted on.
    # earlier action = higher score.
    # --------------------------------------------------------
    recovery_step = None
    for i, step_data in enumerate(episode_history):
        action = step_data.get("action", {})
        if (
            action.get("action_type") in ["restart_node", "remove_from_ring"]
            and action.get("target_node") == FAULT_NODE
        ):
            recovery_step = i + 1  # 1-indexed
            break

    if recovery_step is not None:
        # linear decay: step 1 = 1.0, step MAX_STEPS = 0.1
        steps_score = max(0.1, 1.0 - (recovery_step - 1) / MAX_STEPS)
    else:
        steps_score = 0.0  # never acted on the fault node

    # --------------------------------------------------------
    # 4. Ring integrity score (10%)
    # check final state: is node_3 back in ring OR removed?
    # both are acceptable resolutions.
    # --------------------------------------------------------
    final_nodes = {n["id"]: n for n in final_obs.get("nodes", [])}
    fault_node_state = final_nodes.get(FAULT_NODE)

    if fault_node_state is None:
        ring_score = 0.0
    elif fault_node_state["status"] in ["healthy"] and fault_node_state["in_ring"]:
        ring_score = 1.0   # fully recovered
    elif not fault_node_state["in_ring"]:
        ring_score = 0.7   # removed from ring — acceptable resolution
    else:
        ring_score = 0.0   # still crashed and in ring — bad

    # --------------------------------------------------------
    # weighted final score
    # --------------------------------------------------------
    score = (
        0.40 * throughput_score
        + 0.30 * loss_health_score
        + 0.20 * steps_score
        + 0.10 * ring_score
    )

    return round(min(1.0, max(0.0, score)), 4)


def grade_with_breakdown(episode_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Same as grade() but returns full breakdown for debugging.
    """
    if not episode_history:
        return {"score": 0.0, "error": "empty episode"}

    TARGET_THROUGHPUT = 0.85
    MAX_STEPS = 15
    FAULT_NODE = "node_3"

    last_third_start = max(0, len(episode_history) * 2 // 3)
    last_steps = episode_history[last_third_start:]
    throughput_values = [
        s["observation"]["job"]["cluster_throughput"] for s in last_steps
    ]
    avg_final_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0.0
    throughput_score = min(1.0, avg_final_throughput / TARGET_THROUGHPUT)

    final_obs = episode_history[-1]["observation"]
    final_loss = final_obs["job"]["loss"]
    final_expected = final_obs["job"]["expected_loss"]
    loss_deviation = abs(final_loss - final_expected) / final_expected if final_expected > 0 else 1.0
    loss_health_score = max(0.0, 1.0 - min(1.0, loss_deviation * 3.0))

    recovery_step = None
    for i, step_data in enumerate(episode_history):
        action = step_data.get("action", {})
        if action.get("action_type") in ["restart_node", "remove_from_ring"]                 and action.get("target_node") == FAULT_NODE:
            recovery_step = i + 1
            break
    steps_score = max(0.1, 1.0 - (recovery_step - 1) / MAX_STEPS) if recovery_step else 0.0

    final_nodes = {n["id"]: n for n in final_obs.get("nodes", [])}
    fault_node_state = final_nodes.get(FAULT_NODE)
    if fault_node_state is None:
        ring_score = 0.0
    elif fault_node_state["status"] == "healthy" and fault_node_state["in_ring"]:
        ring_score = 1.0
    elif not fault_node_state["in_ring"]:
        ring_score = 0.7
    else:
        ring_score = 0.0

    final_score = round(min(1.0, max(0.0,
        0.40 * throughput_score
        + 0.30 * loss_health_score
        + 0.20 * steps_score
        + 0.10 * ring_score
    )), 4)

    return {
        "score": final_score,
        "throughput_score": round(throughput_score, 4),
        "loss_health_score": round(loss_health_score, 4),
        "steps_score": round(steps_score, 4),
        "ring_score": round(ring_score, 4),
        "recovery_step": recovery_step,
        "avg_final_throughput": round(avg_final_throughput, 4),
        "loss_deviation_pct": round(loss_deviation * 100, 2),
        "total_steps": len(episode_history),
    }
