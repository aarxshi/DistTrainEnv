
"""
environment/env.py — OpenEnv-compliant DistTrainEnv environment.

Wraps RingCluster in the standard OpenEnv interface.
This is what the FastAPI app exposes as HTTP endpoints.
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
    hard_fault_config
)

# map task_id string to fault config function
TASK_CONFIGS = {
    "easy":   easy_fault_config,
    "medium": medium_fault_config,
    "hard":   hard_fault_config,
}

# max steps per episode per task
MAX_STEPS = {
    "easy":   15,
    "medium": 20,
    "hard":   25,
}


class DistTrainEnv:
    """
    OpenEnv-compliant distributed training fault recovery environment.

    The agent must detect and recover from faults in a simulated
    distributed ML training cluster running ring all-reduce.

    Tasks:
        easy   — single node crash, agent must restart it
        medium — straggler node, agent must identify and remove it
        hard   — cascading OOM -> straggler -> gradient staleness,
                 agent must find and fix the root cause
    """

    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.max_steps = MAX_STEPS[task_id]
        self._done = False

        # build cluster with the correct fault config
        fault_events = TASK_CONFIGS[task_id]()
        self.cluster = RingCluster(n_nodes=8, fault_events=fault_events)
        self.reward_engine = RewardEngine(self.cluster)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset environment to initial state.
        Optionally switch to a different task.
        Returns initial Observation.
        """
        if task_id and task_id in TASK_CONFIGS:
            self.task_id = task_id
            self.max_steps = MAX_STEPS[task_id]

        self._done = False

        fault_events = TASK_CONFIGS[self.task_id]()
        self.cluster.reset(fault_events=fault_events)
        self.reward_engine.reset()

        # tick once to get initial state with faults visible
        # (step 0 is clean, faults start at step 1)
        initial_state = self.cluster.get_state()

        return self._build_observation(initial_state)

    def step(self, action: Action) -> StepResult:
        """
        Apply action and advance simulation one step.
        Returns StepResult with observation, reward, done, info.
        """
        if self._done:
            # episode already finished — return terminal state
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

        # 1. Save state before action for reward computation
        prev_state = self.cluster.get_state()

        # 2. Apply action to cluster
        success, msg = self.cluster.apply_action(
            action_type=action.action_type,
            target_node=action.target_node,
            parameters=action.parameters,
        )

        # 3. Advance simulation one step
        curr_state = self.cluster.tick()

        # 4. Compute reward
        reward_dict = self.reward_engine.compute(
            action_type=action.action_type,
            target_node=action.target_node,
            action_success=success,
            prev_state=prev_state,
            curr_state=curr_state,
        )

        reward = Reward(**reward_dict)

        # 5. Check done conditions
        done = self._check_done(curr_state)
        self._done = done

        # 6. Build observation
        obs = self._build_observation(curr_state)

        # 7. Build info dict (extra diagnostics for agent)
        info = {
            "action_success": success,
            "action_message": msg,
            "step": curr_state["step"],
            "max_steps": self.max_steps,
            "root_cause_fixed": curr_state.get("root_cause_fixed", False),
            "is_healthy": self.cluster.is_healthy(),
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )

    def state(self) -> dict:
        """
        Return full internal state as JSON.
        Required by OpenEnv spec.
        Used by openenv validate to inspect environment state.
        """
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
        }

    def _check_done(self, state: dict) -> bool:
      # max steps reached
      if state["step"] >= self.max_steps:
          return True

      # don't allow early termination until faults have had time to develop
      MIN_STEPS = {"easy": 3, "medium": 4, "hard": 6}
      if state["step"] < MIN_STEPS.get(self.task_id, 3):
          return False

      # cluster recovered — all nodes healthy and throughput restored
      if self.cluster.is_healthy():
          return True

      # unrecoverable — loss diverged AND multiple critical nodes
      critical_nodes = sum(
          1 for n in state["nodes"]
          if n["status"] in ["oom", "crashed"]
      )
      if state["job"]["loss_diverging"] and critical_nodes >= 3:
          return True

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
        )
