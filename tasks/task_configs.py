
"""
tasks/task_configs.py — Task metadata for DistTrainEnv.

Finale upgrades:
- fault_node and root_cause_node removed (now dynamic via faults.py)
- max_steps updated: medium=60, hard=120
- descriptions updated to reflect stochastic fault structure
"""

from typing import Dict, Any


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {

    "easy": {
        "task_id": "easy",
        "name": "Node Crash Recovery",
        "description": (
            "A single worker node crashes at step 1. "
            "Detect it and restart or remove it from the ring."
        ),
        "max_steps": 15,
        "success_threshold": 0.7,
        "target_throughput": 0.85,
        "fault_node": "node_3",        # easy stays deterministic
        "root_cause_node": "node_3",
        "weights": {
            "throughput_recovery": 0.40,
            "loss_health":         0.30,
            "steps_efficiency":    0.20,
            "ring_integrity":      0.10,
        },
    },

    "medium": {
        "task_id": "medium",
        "name": "Stochastic Compound Fault Recovery",
        "description": (
            "A compound fault hits the cluster each episode — "
            "a primary straggler, a secondary OOM, and an intermittent "
            "straggler on random nodes with random timing. "
            "One healthy node shows elevated memory (false alarm). "
            "Identify the root cause, not just visible symptoms. "
            "Agent cannot memorize — every episode is different."
        ),
        "max_steps": 60,               # updated from 20
        "success_threshold": 0.6,
        "target_throughput": 0.80,
        "fault_node": None,            # dynamic — set by faults.py each episode
        "root_cause_node": None,       # dynamic — set by faults.py each episode
        "weights": {
            "throughput_recovery": 0.35,
            "detection_speed":     0.30,
            "loss_health":         0.25,
            "ring_integrity":      0.10,
        },
    },

    "hard": {
        "task_id": "hard",
        "name": "3-Phase Long-Horizon Fault Recovery",
        "description": (
            "A 3-phase distributed training run across 120 steps. "
            "Phase 1 (1-30): cluster warmup — agent allocates nodes. "
            "Phase 2 (31-90): fault storm — OOM root cause, cascade straggler, "
            "intermittent faults, second OOM on random nodes. "
            "Phase 3 (91-120): stabilize and optimize. "
            "Early Phase 1 mistakes compound into Phase 2 failures. "
            "One false alarm node visible from Phase 1."
        ),
        "max_steps": 120,              # updated from 25
        "success_threshold": 0.5,
        "target_throughput": 0.75,
        "fault_node": None,            # dynamic — set by faults.py each episode
        "root_cause_node": None,       # dynamic — set by faults.py each episode
        "weights": {
            "root_cause_fixed":    0.35,
            "loss_health":         0.30,
            "throughput_recovery": 0.20,
            "early_detection":     0.15,
        },
    },

}


def get_task_config(task_id: str) -> Dict[str, Any]:
    """Get config dict for a task. Raises KeyError if unknown."""
    if task_id not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task_id: {task_id!r}. "
            f"Valid options: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_id]
