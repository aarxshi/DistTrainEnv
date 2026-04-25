
"""
environment/env.py — OpenEnv-compliant DistTrainEnv environment.

Finale upgrades:
- Phase-aware reset for hard task (3-phase structure)
- False alarm node wiring
- Stochastic fault configs for medium and hard
- Updated max steps: medium=60, hard=120
- Minimum episode length per task
- Phase info passed through to Observation
"""

from typing import Optional
from environment.ring_cluster import RingCluster
from environment.reward import RewardEngine
from environment.models import (
    Observation, Action, Reward, StepResult,
    NodeObservation, JobObservation
)
from environment.faults import (
    easy_fault_config,
    medium_fault_config,
    medium_false_alarm_config,
    hard_fault_config,
    hard_false_alarm_config,
    EASY_MAX_STEPS,
    MEDIUM_MAX_STEPS,
    HARD_MAX_STEPS,
)

TASK_CONFIGS = {
    "easy":   easy_fault_config,
    "medium": medium_fault_config,
    "hard":   hard_fault_config,
}

MAX_STEPS = {
    "easy":   EASY_MAX_STEPS,
    "medium": MEDIUM_MAX_STEPS,
    "hard":   HARD_MAX_STEPS,
}

# Minimum steps before episode can end on recovery
# Ensures training signal has enough steps to be meaningful
MIN_STEPS = {
    "easy":   8,
    "medium": 15,
    "hard":   HARD_MAX_STEPS,  # hard always runs full length
}


class DistTrainEnv:
    """
    OpenEnv-compliant distributed training fault recovery environment.

    Tasks:
        easy   — deterministic single node crash (baseline anchor)
        medium — stochastic compound fault, 60 steps, false alarm
        hard   — 3-phase long-horizon, 120 steps, compound faults,
                 false alarm, intermittent stragglers
    """

    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.max_steps = MAX_STEPS[task_id]
        self._done = False

        fault_events = TASK_CONFIGS[task_id]()
        false_alarm = self._get_false_alarm(task_id, fault_events)

        self.cluster = RingCluster(
            n_nodes=8,
            fault_events=fault_events,
            false_alarm=false_alarm,
            task_id=task_id,
        )
        self.reward_engine = RewardEngine(self.cluster)

    def _get_false_alarm(self, task_id, fault_events):
        """Returns false alarm config for medium and hard tasks."""
        if task_id == "medium":
            return medium_false_alarm_config(fault_events)
        elif task_id == "hard":
            return hard_false_alarm_config(fault_events)
        return None

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset environment to initial state.
        Generates new stochastic fault config each reset.
        Returns initial Observation.
        """
        if task_id and task_id in TASK_CONFIGS:
            self.task_id = task_id
            self.max_steps = MAX_STEPS[task_id]

        self._done = False

        fault_events = TASK_CONFIGS[self.task_id]()
        false_alarm = self._get_false_alarm(self.task_id, fault_events)

        self.cluster.reset(
            fault_events=fault_events,
            false_alarm=false_alarm,
            task_id=self.task_id,
        )
        self.reward_engine.reset()

        initial_state = self.cluster.get_state()
        return self._build_observation(initial_state)

    def step(self, action: Action) -> StepResult:
        """
        Apply action and advance simulation one step.
        Returns StepResult with observation, reward, done, info.
        """
        if self._done:
            obs = self._build_observation(self.cluster.get_state())
            reward = Reward(
                value=0.0,
                throughput_score=0.0,
                loss_health_score=0.0,
            )
            return StepResult(
                observation=obs,
                reward=reward,
                done=True,
                info={"message": "Episode already done. Call reset()."}
            )

        prev_state = self.cluster.get_state()

        success, msg = self.cluster.apply_action(
            action_type=action.action_type,
            target_node=action.target_node,
            parameters=action.parameters,
        )

        curr_state = self.cluster.tick()

        reward_dict = self.reward_engine.compute(
            action_type=action.action_type,
            target_node=action.target_node,
            action_success=success,
            prev_state=prev_state,
            curr_state=curr_state,
        )

        reward = Reward(**reward_dict)
        done = self._check_done(curr_state)
        self._done = done
        obs = self._build_observation(curr_state)

        info = {
            "action_success": success,
            "action_message": msg,
            "step": curr_state["step"],
            "max_steps": self.max_steps,
            "root_cause_fixed": curr_state.get("root_cause_fixed", False),
            "is_healthy": self.cluster.is_healthy(),
            "current_phase": curr_state.get("current_phase", 1),
            "false_alarm_restarted": curr_state.get(
                "false_alarm_restarted", False),
            "unnecessary_restarts": curr_state.get(
                "unnecessary_restarts", 0),
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )

    def state(self) -> dict:
        """Return full internal state. Required by OpenEnv spec."""
        raw = self.cluster.get_state()
        return {
            "task_id": self.task_id,
            "current_step": raw["step"],
            "max_steps": self.max_steps,
            "done": self._done,
            "nodes": raw["nodes"],
            "job": raw["job"],
            "ring_order": raw["ring_order"],
            "alerts": raw["alerts"],
            "root_cause_fixed": raw.get("root_cause_fixed", False),
            "is_healthy": self.cluster.is_healthy(),
            "current_phase": raw.get("current_phase", 1),
            "false_alarm_restarted": raw.get("false_alarm_restarted", False),
            "unnecessary_restarts": raw.get("unnecessary_restarts", 0),
        }

    def _check_done(self, state: dict) -> bool:
        """
        Episode termination logic.

        Easy/Medium:
        - Must run at least MIN_STEPS before ending on recovery
        - Ends when healthy OR max steps reached OR unrecoverable

        Hard:
        - Runs all 120 steps (max steps)
        - Can end early only if unrecoverable
        - Phase 3 healthy = success but episode continues to max steps
          for full training signal
        """
        current_step = state["step"]
        min_steps = MIN_STEPS.get(self.task_id, 8)

        # Never end before minimum steps
        if current_step < min_steps:
            return False

        # Max steps always ends episode
        if current_step >= self.max_steps:
            return True

        # Unrecoverable state — end early regardless of task
        critical_nodes = sum(
            1 for n in state["nodes"]
            if n["status"] in ["oom", "crashed"]
        )
        if state["job"]["loss_diverging"] and critical_nodes >= 3:
            return True

        # Easy and medium — end when healthy
        if self.task_id != "hard":
            if self.cluster.is_healthy():
                return True
            return False

        # Hard — must complete all phases
        # Only ends early if unrecoverable (handled above)
        return False

    def _build_observation(self, state: dict) -> Observation:
        """Convert raw cluster state dict to typed Observation model."""
        nodes = [
            NodeObservation(
                id=n["id"],
                status=n["status"],
                memory=n["memory"],
                throughput=n["throughput"],
                latency=n["latency"],
                in_ring=n["in_ring"],
            )
            for n in state["nodes"]
        ]

        job = JobObservation(
            step=state["job"]["step"],
            loss=state["job"]["loss"],
            expected_loss=state["job"]["expected_loss"],
            cluster_throughput=state["job"]["cluster_throughput"],
            gradient_staleness=state["job"]["gradient_staleness"],
            loss_diverging=state["job"]["loss_diverging"],
        )

        return Observation(
            nodes=nodes,
            job=job,
            ring_order=state["ring_order"],
            alerts=state["alerts"],
            step=state["step"],
            task_id=self.task_id,
            current_phase=state.get("current_phase", 1),
            false_alarm_restarted=state.get("false_alarm_restarted", False),
            unnecessary_restarts=state.get("unnecessary_restarts", 0),
        )
