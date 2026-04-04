
"""
graders/grader_hard.py — Grader for Hard task: Cascading OOM.

This grader specifically rewards agents that:
1. Fix node_2 (root cause), not just node_7 (symptom)
2. Act on node_2 EARLY (before memory hits 0.95)
3. Restore gradient health (staleness goes back to 0)
4. Recover loss curve trajectory

A naive agent that only reacts to visible symptoms (node_7)
will score much lower than one that traces the causal chain.
This is the score variance that impresses judges.
"""

from typing import List, Dict, Any


def grade(episode_history: List[Dict[str, Any]]) -> float:
    """
    Grade a completed hard task episode.
    Returns float in [0.0, 1.0].
    """
    if not episode_history:
        return 0.0

    ROOT_CAUSE_NODE = "node_2"
    SYMPTOM_NODE = "node_7"
    TARGET_THROUGHPUT = 0.75
    MAX_STEPS = 25
    OOM_CRITICAL_MEMORY = 0.90  # threshold for "too late" detection

    # --------------------------------------------------------
    # 1. Root cause fix score (35%)
    # did the agent act on node_2 (root cause)?
    # full score: acted on node_2 directly
    # partial score: acted on node_7 only (fixed symptom)
    # zero: acted on neither
    # penalty: restarted healthy nodes unnecessarily
    # --------------------------------------------------------
    acted_on_root_cause = False
    acted_on_symptom_only = False
    root_cause_action_step = None
    unnecessary_restarts = 0

    fault_nodes = {ROOT_CAUSE_NODE, SYMPTOM_NODE}

    for i, step_data in enumerate(episode_history):
        action = step_data.get("action", {})
        action_type = action.get("action_type", "noop")
        target = action.get("target_node")

        if action_type in ["restart_node", "remove_from_ring", "reduce_batch"]:
            if target == ROOT_CAUSE_NODE:
                acted_on_root_cause = True
                if root_cause_action_step is None:
                    root_cause_action_step = i + 1
            elif target == SYMPTOM_NODE and not acted_on_root_cause:
                acted_on_symptom_only = True
            elif target and target not in fault_nodes:
                # check if this node was healthy when restarted
                obs = step_data.get("observation", {})
                nodes = {n["id"]: n for n in obs.get("nodes", [])}
                target_state = nodes.get(target)
                if target_state and target_state["status"] == "healthy":
                    unnecessary_restarts += 1

    if acted_on_root_cause:
        root_cause_score = 1.0
    elif acted_on_symptom_only:
        root_cause_score = 0.3  # partial — fixed a problem but not root cause
    else:
        root_cause_score = 0.0

    # penalty for trashing healthy nodes
    root_cause_score = max(0.0, root_cause_score - 0.1 * unnecessary_restarts)

    # --------------------------------------------------------
    # 2. Loss health score (30%)
    # gradient staleness must come back down.
    # loss must return toward expected trajectory.
    # --------------------------------------------------------
    final_obs = episode_history[-1]["observation"]
    final_job = final_obs["job"]

    # staleness: 0.0 is perfect, 1.0 is worst
    staleness_score = max(0.0, 1.0 - final_job["gradient_staleness"])

    # loss deviation
    final_loss = final_job["loss"]
    final_expected = final_job["expected_loss"]
    if final_expected > 0:
        loss_deviation = abs(final_loss - final_expected) / final_expected
        loss_traj_score = max(0.0, 1.0 - min(1.0, loss_deviation * 4.0))
    else:
        loss_traj_score = 0.0

    # penalty if loss is diverging at end
    if final_job.get("loss_diverging", False):
        loss_traj_score *= 0.5

    loss_health_score = 0.5 * staleness_score + 0.5 * loss_traj_score

    # --------------------------------------------------------
    # 3. Throughput recovery score (20%)
    # --------------------------------------------------------
    last_third_start = max(0, len(episode_history) * 2 // 3)
    last_steps = episode_history[last_third_start:]
    throughput_values = [
        s["observation"]["job"]["cluster_throughput"] for s in last_steps
    ]
    avg_final_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0.0
    throughput_score = min(1.0, avg_final_throughput / TARGET_THROUGHPUT)

    # --------------------------------------------------------
    # 4. Early detection bonus (15%)
    # did the agent act on node_2 BEFORE memory went critical?
    # check node_2's memory at the step the agent first acted on it.
    # --------------------------------------------------------
    early_detection_score = 0.0

    if root_cause_action_step is not None:
        # get node_2's memory at the step of the action
        action_step_data = episode_history[root_cause_action_step - 1]
        obs_at_action = action_step_data.get("observation", {})
        nodes_at_action = {n["id"]: n for n in obs_at_action.get("nodes", [])}
        node2_state = nodes_at_action.get(ROOT_CAUSE_NODE)

        if node2_state:
            memory_at_action = node2_state["memory"]
            if memory_at_action < 0.75:
                early_detection_score = 1.0    # caught it very early
            elif memory_at_action < OOM_CRITICAL_MEMORY:
                # linear interpolation: 0.75 -> 1.0, 0.90 -> 0.0
                early_detection_score = max(0.0,
                    1.0 - (memory_at_action - 0.75) / (OOM_CRITICAL_MEMORY - 0.75)
                )
            else:
                early_detection_score = 0.0    # acted too late

    # --------------------------------------------------------
    # weighted final score
    # --------------------------------------------------------
    score = (
        0.35 * root_cause_score
        + 0.30 * loss_health_score
        + 0.20 * throughput_score
        + 0.15 * early_detection_score
    )

    return round(min(1.0, max(0.0, score)), 4)


def grade_with_breakdown(episode_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same as grade() but returns full breakdown for debugging."""
    if not episode_history:
        return {"score": 0.0, "error": "empty episode"}

    ROOT_CAUSE_NODE = "node_2"
    SYMPTOM_NODE = "node_7"
    TARGET_THROUGHPUT = 0.75
    OOM_CRITICAL_MEMORY = 0.90

    acted_on_root_cause = False
    acted_on_symptom_only = False
    root_cause_action_step = None
    unnecessary_restarts = 0
    fault_nodes = {ROOT_CAUSE_NODE, SYMPTOM_NODE}

    for i, step_data in enumerate(episode_history):
        action = step_data.get("action", {})
        action_type = action.get("action_type", "noop")
        target = action.get("target_node")
        if action_type in ["restart_node", "remove_from_ring", "reduce_batch"]:
            if target == ROOT_CAUSE_NODE:
                acted_on_root_cause = True
                if root_cause_action_step is None: root_cause_action_step = i + 1
            elif target == SYMPTOM_NODE and not acted_on_root_cause:
                acted_on_symptom_only = True
            elif target and target not in fault_nodes:
                obs = step_data.get("observation", {})
                nodes = {n["id"]: n for n in obs.get("nodes", [])}
                ts = nodes.get(target)
                if ts and ts["status"] == "healthy": unnecessary_restarts += 1

    if acted_on_root_cause: root_cause_score = 1.0
    elif acted_on_symptom_only: root_cause_score = 0.3
    else: root_cause_score = 0.0
    root_cause_score = max(0.0, root_cause_score - 0.1 * unnecessary_restarts)

    final_obs = episode_history[-1]["observation"]
    final_job = final_obs["job"]
    staleness_score = max(0.0, 1.0 - final_job["gradient_staleness"])
    final_loss = final_job["loss"]
    final_expected = final_job["expected_loss"]
    loss_deviation = abs(final_loss - final_expected) / final_expected if final_expected > 0 else 1.0
    loss_traj_score = max(0.0, 1.0 - min(1.0, loss_deviation * 4.0))
    if final_job.get("loss_diverging", False): loss_traj_score *= 0.5
    loss_health_score = 0.5 * staleness_score + 0.5 * loss_traj_score

    last_third_start = max(0, len(episode_history) * 2 // 3)
    last_steps = episode_history[last_third_start:]
    tpv = [s["observation"]["job"]["cluster_throughput"] for s in last_steps]
    avg_final_throughput = sum(tpv) / len(tpv) if tpv else 0.0
    throughput_score = min(1.0, avg_final_throughput / TARGET_THROUGHPUT)

    early_detection_score = 0.0
    node2_memory_at_action = None
    if root_cause_action_step is not None:
        asd = episode_history[root_cause_action_step - 1]
        naa = {n["id"]: n for n in asd.get("observation", {}).get("nodes", [])}
        n2 = naa.get(ROOT_CAUSE_NODE)
        if n2:
            node2_memory_at_action = n2["memory"]
            if node2_memory_at_action < 0.75: early_detection_score = 1.0
            elif node2_memory_at_action < OOM_CRITICAL_MEMORY:
                early_detection_score = max(0.0, 1.0 - (node2_memory_at_action - 0.75) / (OOM_CRITICAL_MEMORY - 0.75))

    final_score = round(min(1.0, max(0.0,
        0.35 * root_cause_score + 0.30 * loss_health_score
        + 0.20 * throughput_score + 0.15 * early_detection_score
    )), 4)

    return {
        "score": final_score,
        "root_cause_score": round(root_cause_score, 4),
        "loss_health_score": round(loss_health_score, 4),
        "staleness_score": round(staleness_score, 4),
        "loss_traj_score": round(loss_traj_score, 4),
        "throughput_score": round(throughput_score, 4),
        "early_detection_score": round(early_detection_score, 4),
        "acted_on_root_cause": acted_on_root_cause,
        "acted_on_symptom_only": acted_on_symptom_only,
        "root_cause_action_step": root_cause_action_step,
        "node2_memory_at_action": node2_memory_at_action,
        "unnecessary_restarts": unnecessary_restarts,
        "final_gradient_staleness": final_job["gradient_staleness"],
        "loss_diverging_at_end": final_job.get("loss_diverging", False),
        "loss_deviation_pct": round(loss_deviation * 100, 2),
        "avg_final_throughput": round(avg_final_throughput, 4),
        "total_steps": len(episode_history),
    }
