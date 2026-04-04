
"""
graders/run_graders.py — Unified grader entry point.

Usage:
    from graders.run_graders import run_grader
    score = run_grader(task_id, episode_history)
    breakdown = run_grader(task_id, episode_history, breakdown=True)
"""

from typing import List, Dict, Any, Union
from graders import grader_easy, grader_medium, grader_hard


GRADERS = {
    "easy":   grader_easy,
    "medium": grader_medium,
    "hard":   grader_hard,
}


def run_grader(
    task_id: str,
    episode_history: List[Dict[str, Any]],
    breakdown: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Run the grader for the given task.

    Args:
        task_id: "easy", "medium", or "hard"
        episode_history: list of step dicts from inference loop
        breakdown: if True, return full breakdown dict instead of float

    Returns:
        float in [0.0, 1.0] if breakdown=False
        dict with score + sub-scores if breakdown=True
    """
    if task_id not in GRADERS:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS.keys())}"
        )

    grader = GRADERS[task_id]

    if breakdown:
        return grader.grade_with_breakdown(episode_history)
    else:
        return grader.grade(episode_history)


def run_all_graders(
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Run all three graders given a dict of {task_id: episode_history}.
    Returns summary with individual scores and weighted total.

    Weights: easy=0.25, medium=0.35, hard=0.40
    (harder tasks count more)
    """
    task_weights = {"easy": 0.25, "medium": 0.35, "hard": 0.40}
    scores = {}
    breakdowns = {}

    for task_id, history in results.items():
        scores[task_id] = run_grader(task_id, history)
        breakdowns[task_id] = run_grader(task_id, history, breakdown=True)

    # weighted total over completed tasks only
    total_weight = sum(task_weights[t] for t in scores)
    weighted_total = (
        sum(scores[t] * task_weights[t] for t in scores) / total_weight
        if total_weight > 0 else 0.0
    )

    return {
        "scores": scores,
        "weighted_total": round(weighted_total, 4),
        "breakdowns": breakdowns,
    }
