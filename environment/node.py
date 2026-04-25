
"""
environment/node.py — Single worker node in the training cluster.

Finale upgrades:
- False alarm node support (_is_false_alarm flag)
- False alarm memory elevation in tick()
- Intermittent straggler recovery handled by FaultInjector
  (node.py just needs to support the state transitions)
"""

import random
from dataclasses import dataclass, field
from typing import Literal

NodeStatus = Literal["healthy", "slow", "oom", "crashed"]


@dataclass
class Node:
    """
    Simulates one GPU worker node in a distributed training ring.
    All values updated by RingCluster.tick() each timestep.
    """

    id: str

    status: NodeStatus = "healthy"
    in_ring: bool = True

    memory: float = 0.3
    throughput: float = 1.0
    latency: float = 5.0

    # Internal fault tracking — NOT directly observable by agent
    _oom_progress: float = field(default=0.0, repr=False)
    _fault_type: str = field(default="none", repr=False)
    _steps_since_fault: int = field(default=0, repr=False)

    # Baseline values for reward computation
    _baseline_throughput: float = field(default=1.0, repr=False)
    _baseline_memory: float = field(default=0.3, repr=False)

    # NEW: False alarm support
    # Node shows elevated memory but never actually fails
    # Set by FaultInjector.tick() when false_alarm config is present
    _is_false_alarm: bool = field(default=False, repr=False)
    _false_alarm_memory: float = field(default=0.75, repr=False)

    def tick(self, noise: bool = True):
        """
        Advance node state by one timestep.
        Applies fault progression and small random noise.
        Called by RingCluster.tick() every step.
        """

        if self.status == "crashed":
            self.throughput = 0.0
            self.in_ring = False
            return

        # --- False alarm handling ---
        # Memory stays elevated but node NEVER crashes or slows
        # Agent that restarts this node takes a penalty
        # Must be checked BEFORE OOM progression so it doesn't
        # accidentally trigger real OOM logic
        if self._is_false_alarm:
            # Small oscillation around target memory to look realistic
            target = self._false_alarm_memory
            self.memory = max(0.65, min(0.88,
                target + random.uniform(-0.02, 0.02)))
            # Status stays healthy — this is the trap
            self.status = "healthy"
            self.throughput = max(0.85, min(1.0,
                self.throughput + random.uniform(-0.02, 0.02)))
            self._steps_since_fault += 1
            return  # skip all other fault logic for false alarm nodes

        # --- OOM progression ---
        # Memory climbs gradually — agent should catch this early
        if self._fault_type == "oom":
            self._oom_progress = min(1.0, self._oom_progress + 0.06)
            self.memory = min(1.0,
                self._baseline_memory + self._oom_progress * 0.7)

            if self.memory > 0.85 and self.status == "healthy":
                self.status = "slow"
                self.throughput = max(0.1, self.throughput - 0.3)

            if self.memory > 0.95:
                self.status = "oom"
                self.throughput = max(0.05, self.throughput - 0.2)

            if self.memory >= 1.0:
                self._crash()
                return

        # --- Straggler fault ---
        if self._fault_type == "straggler":
            self.status = "slow"
            self.latency = min(200.0, self.latency + 0.5)

        # --- Crash fault ---
        if self._fault_type == "crash":
            self._crash()
            return

        # --- Small random noise on healthy nodes ---
        # Makes observations realistic, agent cant threshold on exact values
        if noise and self.status == "healthy":
            self.memory = max(0.1, min(0.95,
                self.memory + random.uniform(-0.01, 0.01)))
            self.throughput = max(0.1, min(1.0,
                self.throughput + random.uniform(-0.02, 0.02)))
            self.latency = max(1.0, min(50.0,
                self.latency + random.uniform(-0.5, 0.5)))

        self._steps_since_fault += 1

    def _crash(self):
        """Transition node to crashed state."""
        self.status = "crashed"
        self.in_ring = False
        self.throughput = 0.0
        self.latency = 9999.0

    def restart(self):
        """
        Agent action: restart this node.
        Starts at 60% throughput — recovers over 2 warmup ticks.

        IMPORTANT: restarting a false alarm node clears the false alarm
        flag but gives a penalty in reward.py — the node was healthy,
        restarting it was wasteful.
        """
        self.status = "healthy"
        self.in_ring = True
        self.memory = self._baseline_memory
        self.throughput = 0.6
        self.latency = 5.0
        self._oom_progress = 0.0
        self._fault_type = "none"
        self._steps_since_fault = 0
        # Clear false alarm flag if agent restarts it
        # (penalty already applied in reward.py)
        self._is_false_alarm = False

    def remove_from_ring(self):
        """
        Agent action: remove node from ring without restarting.
        Stops the node from blocking all-reduce.
        """
        self.in_ring = False

    def reduce_batch(self):
        """
        Agent action: halve batch size on this node.
        Reduces memory pressure. Only helps before memory >= 0.95.

        NOTE: reduce_batch on a false alarm node is a mild waste
        (small penalty) but not as bad as restarting it.
        """
        if self._is_false_alarm:
            # Slightly reduces the false alarm memory display
            # but doesnt fix anything — node was never broken
            self._false_alarm_memory = max(
                0.65, self._false_alarm_memory - 0.05)
            return

        if self.memory < 0.95:
            self._oom_progress = max(0.0, self._oom_progress - 0.3)
            self.memory = max(self._baseline_memory, self.memory - 0.2)
            self.throughput = max(0.5, self.throughput - 0.1)

    def warmup_tick(self):
        """
        Gradually restore throughput after restart.
        Simulates node coming back online over 2 steps.
        """
        if self.status == "healthy" and self.throughput < 1.0:
            self.throughput = min(1.0, self.throughput + 0.2)

    def to_state(self) -> dict:
        """
        Return observable state as dict.
        NOTE: _is_false_alarm is NOT exposed here — agent must
        infer from behavior (memory elevated but never crashes).
        """
        return {
            "id": self.id,
            "status": self.status,
            "memory": round(self.memory, 3),
            "throughput": round(self.throughput, 3),
            "latency": round(self.latency, 2),
            "in_ring": self.in_ring,
        }
