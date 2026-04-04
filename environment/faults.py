
"""
environment/faults.py — Fault injection engine for DistTrainEnv.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FaultEvent:
    """A single fault to inject at a specific timestep."""

    step: int
    node_id: str
    fault_type: str       # "crash" | "straggler" | "oom"
    slowdown_factor: float = 0.3
    is_root_cause: bool = False


class FaultInjector:
    """
    Injects faults into nodes at configured timesteps.
    Called every step by RingCluster.tick().
    """

    def __init__(self, fault_events: List[FaultEvent]):
        self.fault_events = fault_events
        self._injected = set()

    def tick(self, current_step: int, nodes: dict):
        """
        Fire any faults scheduled for this step.
        Each fault fires exactly once.
        """
        for event in self.fault_events:
            event_key = f"{event.node_id}_{event.step}_{event.fault_type}"

            if current_step == event.step and event_key not in self._injected:
                node = nodes.get(event.node_id)
                if node is None:
                    continue

                if event.fault_type == "crash":
                    node._fault_type = "crash"

                elif event.fault_type == "straggler":
                    node._fault_type = "straggler"
                    node.status = "slow"
                    node.throughput = event.slowdown_factor
                    node.latency = 80.0

                elif event.fault_type == "oom":
                    # oOM starts silently — memory climbs gradually
                    # agent must notice BEFORE it becomes critical
                    node._fault_type = "oom"
                    node._oom_progress = 0.1

                self._injected.add(event_key)
                print(f"  [FaultInjector] Step {current_step}: "
                      f"{event.fault_type} injected on {event.node_id}")

    def reset(self):
        """Clear injection history for episode reset."""
        self._injected.clear()

    def get_root_cause_node(self) -> Optional[str]:
        """
        Returns node_id of the root cause fault.
        Used by hard task grader to verify agent fixed
        the RIGHT node, not just a symptom.
        """
        for event in self.fault_events:
            if event.is_root_cause:
                return event.node_id
        return None


# pre-built fault configs — imported by task files

def easy_fault_config() -> List[FaultEvent]:
    """
    Task 1 (Easy): Single node crash at step 1.
    Agent must detect and remove/restart node_3.
    """
    return [
        FaultEvent(
            step=1,
            node_id="node_3",
            fault_type="crash",
            is_root_cause=True
        )
    ]


def medium_fault_config() -> List[FaultEvent]:
    """
    Task 2 (Medium): Straggler node from step 1.
    node_5 runs at 30% speed. Not obvious — shows as slow not crashed.
    Agent must identify bottleneck and remove from ring.
    """
    return [
        FaultEvent(
            step=1,
            node_id="node_5",
            fault_type="straggler",
            slowdown_factor=0.3,
            is_root_cause=True
        )
    ]


def hard_fault_config() -> List[FaultEvent]:
    """
    Task 3 (Hard): Cascading OOM -> straggler -> gradient staleness.

    node_2: OOM at step 1 (ROOT CAUSE — silent, memory climbs slowly)
    node_7: becomes straggler at step 5 (SYMPTOM of node_2 retry storm)

    Naive agent fixes node_7 (visible symptom).
    Smart agent traces back and fixes node_2 (root cause).
    Grader rewards root cause fix with higher score.
    """
    return [
        FaultEvent(
            step=1,
            node_id="node_2",
            fault_type="oom",
            is_root_cause=True       # ROOT CAUSE
        ),
        FaultEvent(
            step=5,
            node_id="node_7",
            fault_type="straggler",
            slowdown_factor=0.4,
            is_root_cause=False      # SYMPTOM
        )
    ]
