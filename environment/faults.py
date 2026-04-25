
"""
environment/faults.py — Fault injection engine for DistTrainEnv.

Finale upgrades:
- Stochastic fault configs for medium and hard tasks
- Compound faults (2 simultaneous issues)
- False alarm node support
- Intermittent straggler support
- Phase-aware fault injection for hard task (3-phase structure)
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class FaultEvent:
    """A single fault to inject at a specific timestep."""

    step: int
    node_id: str
    fault_type: str       # "crash" | "straggler" | "oom" | "intermittent"
    slowdown_factor: float = 0.3
    is_root_cause: bool = False
    
    # NEW: for intermittent stragglers
    is_intermittent: bool = False
    recover_at_step: int = -1
    degrade_again_at_step: int = -1


@dataclass 
class FalseAlarmConfig:
    """
    A node that looks suspicious but never actually fails.
    Memory stays elevated (0.70-0.80) but status stays healthy.
    Tests whether agent inspects before acting.
    Restarting this node gives a penalty.
    """
    node_id: str
    memory_level: float = 0.75


class FaultInjector:
    """
    Injects faults into nodes at configured timesteps.
    Called every step by RingCluster.tick().
    """

    def __init__(
        self,
        fault_events: List[FaultEvent],
        false_alarm: Optional[FalseAlarmConfig] = None,
    ):
        self.fault_events = fault_events
        self.false_alarm = false_alarm
        self._injected = set()
        self._intermittent_recovered = set()
        self._intermittent_degraded_again = set()

    def tick(self, current_step: int, nodes: dict):
        """
        Fire any faults scheduled for this step.
        Handles standard faults, intermittent stragglers,
        and false alarm memory elevation.
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
                    node._fault_type = "oom"
                    node._oom_progress = 0.1

                elif event.fault_type == "intermittent":
                    node._fault_type = "straggler"
                    node.status = "slow"
                    node.throughput = event.slowdown_factor
                    node.latency = 60.0

                self._injected.add(event_key)
                print(f"  [FaultInjector] Step {current_step}: "
                      f"{event.fault_type} injected on {event.node_id}")

            # Intermittent recovery phase
            if (event.is_intermittent
                    and event.recover_at_step > 0
                    and current_step == event.recover_at_step
                    and event.node_id not in self._intermittent_recovered):

                node = nodes.get(event.node_id)
                if node and node.status == "slow":
                    node._fault_type = "none"
                    node.status = "healthy"
                    node.throughput = 0.85
                    node.latency = 8.0
                    self._intermittent_recovered.add(event.node_id)
                    print(f"  [FaultInjector] Step {current_step}: "
                          f"intermittent straggler {event.node_id} "
                          f"appears recovered")

            # Intermittent degrade again phase
            if (event.is_intermittent
                    and event.degrade_again_at_step > 0
                    and current_step == event.degrade_again_at_step
                    and event.node_id in self._intermittent_recovered
                    and event.node_id not in self._intermittent_degraded_again):

                node = nodes.get(event.node_id)
                if node:
                    node._fault_type = "straggler"
                    node.status = "slow"
                    node.throughput = event.slowdown_factor * 0.8
                    node.latency = 90.0
                    self._intermittent_degraded_again.add(event.node_id)
                    print(f"  [FaultInjector] Step {current_step}: "
                          f"intermittent straggler {event.node_id} "
                          f"degraded again")

        # False alarm: elevate memory every step
        if self.false_alarm:
            node = nodes.get(self.false_alarm.node_id)
            if node and node.status == "healthy" and not node._is_false_alarm:
                node._is_false_alarm = True
                node._false_alarm_memory = self.false_alarm.memory_level

    def reset(self):
        """Clear all injection history for episode reset."""
        self._injected.clear()
        self._intermittent_recovered.clear()
        self._intermittent_degraded_again.clear()

    def get_root_cause_node(self) -> Optional[str]:
        for event in self.fault_events:
            if event.is_root_cause:
                return event.node_id
        return None

    def get_false_alarm_node(self) -> Optional[str]:
        if self.false_alarm:
            return self.false_alarm.node_id
        return None

    def get_all_fault_nodes(self) -> List[str]:
        return [e.node_id for e in self.fault_events]


# ----------------------------------------------------------
# EASY — deterministic, baseline anchor
# ----------------------------------------------------------

def easy_fault_config() -> List[FaultEvent]:
    """Task 1 (Easy): Single node crash at step 1. Deterministic."""
    return [
        FaultEvent(
            step=1,
            node_id="node_3",
            fault_type="crash",
            is_root_cause=True
        )
    ]


# ----------------------------------------------------------
# MEDIUM — stochastic compound fault, 60 steps
# ----------------------------------------------------------

def medium_fault_config() -> List[FaultEvent]:
    """
    Task 2 (Medium): Stochastic compound fault.
    - Primary straggler (root cause) on random node
    - Secondary OOM on different random node — fires simultaneously
    - Intermittent straggler on third random node
    Agent cannot memorize — every episode is different.
    """
    node_pool = list(range(1, 8))
    chosen = random.sample(node_pool, 4)

    primary_node      = f"node_{chosen[0]}"
    secondary_node    = f"node_{chosen[1]}"
    intermittent_node = f"node_{chosen[2]}"

    primary_step   = random.randint(1, 3)
    secondary_step = primary_step              # simultaneous compound fault
    slowdown = round(random.uniform(0.15, 0.30), 2)

    intermittent_start   = random.randint(8, 12)
    intermittent_recover = intermittent_start + random.randint(4, 6)
    intermittent_degrade = intermittent_recover + random.randint(5, 8)

    return [
        FaultEvent(
            step=primary_step,
            node_id=primary_node,
            fault_type="straggler",
            slowdown_factor=slowdown,
            is_root_cause=True
        ),
        FaultEvent(
            step=secondary_step,
            node_id=secondary_node,
            fault_type="oom",
            is_root_cause=False
        ),
        FaultEvent(
            step=intermittent_start,
            node_id=intermittent_node,
            fault_type="intermittent",
            slowdown_factor=round(random.uniform(0.3, 0.5), 2),
            is_root_cause=False,
            is_intermittent=True,
            recover_at_step=intermittent_recover,
            degrade_again_at_step=intermittent_degrade,
        ),
    ]


def medium_false_alarm_config(fault_events: List[FaultEvent]) -> FalseAlarmConfig:
    """False alarm for medium task — memory 0.68-0.82, never crashes."""
    faulted_nodes = {e.node_id for e in fault_events}
    candidates = [
        f"node_{i}" for i in range(1, 8)
        if f"node_{i}" not in faulted_nodes
    ]
    chosen = random.choice(candidates)
    return FalseAlarmConfig(
        node_id=chosen,
        memory_level=round(random.uniform(0.68, 0.82), 2)
    )


# ----------------------------------------------------------
# HARD — 3-phase structure, 120 steps
# Phase 1 (steps 1-30):   warmup
# Phase 2 (steps 31-90):  fault storm
# Phase 3 (steps 91-120): stabilize + optimize
# ----------------------------------------------------------

def hard_fault_config() -> List[FaultEvent]:
    """
    Task 3 (Hard): 3-phase long-horizon run.
    Phase 1: warmup — no faults, agent allocates/monitors
    Phase 2: fault storm — OOM root cause + cascade straggler
             + intermittent + second independent OOM
    Phase 3: no new faults — stabilize and optimize
    """
    node_pool = list(range(1, 8))
    chosen = random.sample(node_pool, 5)

    oom_root_node     = f"node_{chosen[0]}"
    straggler_node    = f"node_{chosen[1]}"
    intermittent_node = f"node_{chosen[2]}"
    second_oom_node   = f"node_{chosen[3]}"

    oom_step             = random.randint(31, 35)
    straggler_step       = oom_step + random.randint(8, 12)
    intermittent_start   = random.randint(55, 65)
    intermittent_recover = intermittent_start + random.randint(5, 8)
    intermittent_degrade = intermittent_recover + random.randint(6, 10)
    second_oom_step      = random.randint(60, 70)

    return [
        FaultEvent(
            step=oom_step,
            node_id=oom_root_node,
            fault_type="oom",
            is_root_cause=True
        ),
        FaultEvent(
            step=straggler_step,
            node_id=straggler_node,
            fault_type="straggler",
            slowdown_factor=round(random.uniform(0.3, 0.5), 2),
            is_root_cause=False
        ),
        FaultEvent(
            step=intermittent_start,
            node_id=intermittent_node,
            fault_type="intermittent",
            slowdown_factor=round(random.uniform(0.25, 0.45), 2),
            is_root_cause=False,
            is_intermittent=True,
            recover_at_step=intermittent_recover,
            degrade_again_at_step=intermittent_degrade,
        ),
        FaultEvent(
            step=second_oom_step,
            node_id=second_oom_node,
            fault_type="oom",
            is_root_cause=False
        ),
    ]


def hard_false_alarm_config(fault_events: List[FaultEvent]) -> FalseAlarmConfig:
    """False alarm for hard task — memory 0.74-0.86, visible from Phase 1."""
    faulted_nodes = {e.node_id for e in fault_events}
    candidates = [
        f"node_{i}" for i in range(1, 8)
        if f"node_{i}" not in faulted_nodes
    ]
    chosen = random.choice(candidates)
    return FalseAlarmConfig(
        node_id=chosen,
        memory_level=round(random.uniform(0.74, 0.86), 2)
    )


# ----------------------------------------------------------
# Phase boundary constants
# Imported by ring_cluster.py and env.py
# ----------------------------------------------------------

HARD_PHASE_1_END   = 30
HARD_PHASE_2_START = 31
HARD_PHASE_2_END   = 90
HARD_PHASE_3_START = 91
HARD_PHASE_3_END   = 120

MEDIUM_MAX_STEPS = 60
HARD_MAX_STEPS   = 120
EASY_MAX_STEPS   = 15
