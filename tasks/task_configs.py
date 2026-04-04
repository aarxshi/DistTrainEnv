
"""
tasks/task_configs.py — Task metadata for DistTrainEnv.

Each task config is a dict the grader and inference.py consume.
Fault injection comes from environment/faults.py.
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
        # grading targets
        "target_throughput": 0.85,    # cluster throughput to recover to
        "fault_node": "node_3",       # which node crashes (matches faults.py)
        "root_cause_node": "node_3",  # same as fault for easy task
        # scoring weights
        "weights": {
            "throughput_recovery": 0.40,
            "loss_health":         0.30,
            "steps_efficiency":    0.20,
            "ring_integrity":      0.10,
        },
    },

    "medium": {
        "task_id": "medium",
        "name": "Straggler Detection and Removal",
        "description": (
            "A worker node becomes a straggler at step 1 running at 30% speed. "
            "It is slow but not crashed. Identify and remove it from the ring."
        ),
        "max_steps": 20,
        "success_threshold": 0.6,
        "target_throughput": 0.80,
        "fault_node": "node_5",
        "root_cause_node": "node_5",
        "weights": {
            "throughput_recovery": 0.35,
            "detection_speed":     0.30,  # how quickly straggler was identified
            "loss_health":         0.25,
            "ring_integrity":      0.10,
        },
    },

    "hard": {
        "task_id": "hard",
        "name": "Cascading OOM Fault Recovery",
        "description": (
            "A silent OOM fault begins on node_2 at step 1 (root cause). "
            "Memory climbs gradually. At step 5 node_7 becomes a straggler "
            "as a symptom of node_2s retry storm. Fix the root cause (node_2), "
            "not just the visible symptom (node_7). "
            "Gradient staleness builds silently and loss diverges if untreated."
        ),
        "max_steps": 25,
        "success_threshold": 0.5,
        "target_throughput": 0.75,
        "fault_node": "node_7",       # visible symptom
        "root_cause_node": "node_2",  # actual root cause
        "weights": {
            "root_cause_fixed":    0.35,  # biggest weight — did agent fix node_2?
            "loss_health":         0.30,
            "throughput_recovery": 0.20,
            "early_detection":     0.15,  # bonus for catching OOM before crash
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
