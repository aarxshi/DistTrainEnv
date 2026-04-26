
"""
environment/ring_cluster.py — Main distributed training cluster simulator.

Finale upgrades:
- Phase tracking for hard task (Phase 1 / 2 / 3)
- False alarm node wiring (passed from env.py via reset())
- Phase-aware alerts (different messages per phase)
- Phase transition detection
- Checkpoint bonus tracking for Phase 3
- False alarm restart penalty tracking
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from environment.node import Node
from environment.job import TrainingJob
from environment.faults import (
    FaultInjector, FaultEvent, FalseAlarmConfig,
    HARD_PHASE_1_END, HARD_PHASE_2_START,
    HARD_PHASE_2_END, HARD_PHASE_3_START,
)


class RingCluster:
    """
    Simulates a ring all-reduce distributed training cluster.

    Ring all-reduce speed is bounded by the SLOWEST node.
    If a node crashes, the ring breaks until reformed.

    Finale: supports 3-phase hard task structure and false alarm nodes.
    """

    def __init__(
        self,
        n_nodes: int = 8,
        fault_events: List[FaultEvent] = None,
        false_alarm: Optional[FalseAlarmConfig] = None,
        task_id: str = "easy",
    ):
        self.n_nodes = n_nodes
        self.task_id = task_id
        self.current_step = 0
        self._root_cause_fixed = False
        self.alerts: List[str] = []
        self._last_action = None

        # Phase tracking — only meaningful for hard task
        self.current_phase = 1
        self._phase_1_checkpointed = False  # did agent checkpoint in phase 1?
        self._false_alarm_restarted = False  # did agent restart false alarm?
        self._false_alarm_node_id: Optional[str] = (
            false_alarm.node_id if false_alarm else None
        )

        # Cost tracking for Phase 3
        self._unnecessary_restarts = 0
        self._checkpoints_used = 0

        self.nodes: Dict[str, Node] = {
            f"node_{i}": Node(id=f"node_{i}")
            for i in range(n_nodes)
        }

        self.ring_order: List[str] = [
            f"node_{i}" for i in range(n_nodes)
        ]

        self.job = TrainingJob()

        self.fault_injector = FaultInjector(
            fault_events=fault_events or [],
            false_alarm=false_alarm,
        )

    def reset(
        self,
        fault_events: List[FaultEvent] = None,
        false_alarm: Optional[FalseAlarmConfig] = None,
        task_id: str = None,
    ):
        """
        Reset cluster to healthy state for new episode.
        Accepts optional new fault config and false alarm.
        """
        if task_id:
            self.task_id = task_id

        self.current_step = 0
        self._root_cause_fixed = False
        self.alerts = []
        self._last_action = None
        self.current_phase = 1
        self._phase_1_checkpointed = False
        self._false_alarm_restarted = False
        self._unnecessary_restarts = 0
        self._checkpoints_used = 0
        self._false_alarm_node_id = (
            false_alarm.node_id if false_alarm else None
        )

        self.nodes = {
            f"node_{i}": Node(id=f"node_{i}")
            for i in range(self.n_nodes)
        }

        self.ring_order = [f"node_{i}" for i in range(self.n_nodes)]
        self.job = TrainingJob()

        if fault_events is not None:
            self.fault_injector = FaultInjector(
                fault_events=fault_events,
                false_alarm=false_alarm,
            )
        else:
            self.fault_injector.reset()

    def _update_phase(self):
        """
        Update current phase based on step count.
        Only meaningful for hard task — other tasks stay phase 1.
        """
        if self.task_id != "hard":
            return

        if self.current_step <= HARD_PHASE_1_END:
            self.current_phase = 1
        elif self.current_step <= HARD_PHASE_2_END:
            if self.current_phase == 1:
                # Phase transition 1 -> 2
                print(f"  [Phase] Entering Phase 2 at step "
                      f"{self.current_step} — fault storm begins")
            self.current_phase = 2
        else:
            if self.current_phase == 2:
                # Phase transition 2 -> 3
                print(f"  [Phase] Entering Phase 3 at step "
                      f"{self.current_step} — stabilize and optimize")
            self.current_phase = 3

    def apply_action(
        self,
        action_type: str,
        target_node: Optional[str] = None,
        parameters: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Apply agent action BEFORE ticking.
        Returns (success, message).

        Tracks false alarm restarts and checkpoint usage for reward.
        """
        self._last_action = action_type

        if action_type == "noop":
            return True, "No action taken."

        if action_type == "checkpoint":
            self.job.cluster_throughput *= 0.5
            self._checkpoints_used += 1
            # Track if agent checkpointed during Phase 1
            # This gives a bonus in Phase 2 if it helps recovery
            if self.current_phase == 1:
                self._phase_1_checkpointed = True
            return True, "Checkpoint saved. Throughput halved this step."

        if action_type == "inspect":
            if target_node and target_node in self.nodes:
                return True, f"Inspecting {target_node}."
            return False, f"Node {target_node} not found."

        if not target_node or target_node not in self.nodes:
            return False, f"Invalid target node: {target_node}"

        node = self.nodes[target_node]

        if action_type == "restart_node":
            # Check if this is the false alarm node
            if target_node == self._false_alarm_node_id:
                self._false_alarm_restarted = True
                self._unnecessary_restarts += 1
                # Still allow the restart but flag it for penalty
                node.restart()
                if target_node not in self.ring_order:
                    self.ring_order.append(target_node)
                return True, (
                    f"{target_node} restarted. "
                    f"WARNING: this node appeared healthy."
                )

            # Check if restarting a genuinely healthy node
            if node.status == "healthy" and not node._is_false_alarm:
                self._unnecessary_restarts += 1

            node.restart()
            if target_node not in self.ring_order:
                self.ring_order.append(target_node)
            root = self.fault_injector.get_root_cause_node()
            if target_node == root:
                self._root_cause_fixed = True
            return True, f"{target_node} restarted and rejoined ring."

        elif action_type == "remove_from_ring":
            node.remove_from_ring()
            if target_node in self.ring_order:
                self.ring_order.remove(target_node)
            # Add this:
            root = self.fault_injector.get_root_cause_node()
            if target_node == root:
                self._root_cause_fixed = True
            return True, f"{target_node} removed from ring."

        elif action_type == "reduce_batch":
            if node.status in ["healthy", "slow", "oom"]:
                node.reduce_batch()
                return True, f"Batch size reduced on {target_node}."
            return False, f"{target_node} is crashed, cannot reduce batch."

        return False, f"Unknown action: {action_type}"

    def tick(self) -> dict:
        """
        Advance simulation one timestep.
        Order: update phase → inject faults → update nodes →
               compute ring health → update job → generate alerts
        """
        self.current_step += 1

        # 0. Update phase (hard task only)
        self._update_phase()

        # 1. Inject scheduled faults
        self.fault_injector.tick(self.current_step, self.nodes)

        # 2. Update each node
        for node in self.nodes.values():
            node.tick()
            node.warmup_tick()

        # 3. Compute ring-level metrics
        ring_nodes = [
            self.nodes[nid]
            for nid in self.ring_order
            if nid in self.nodes and self.nodes[nid].in_ring
        ]

        active_count = len(ring_nodes)

        # Ring throughput = slowest node (synchronous all-reduce)
        if ring_nodes:
            ring_throughput = min(n.throughput for n in ring_nodes)
        else:
            ring_throughput = 0.0

        # Gradient staleness grows when nodes are slow or missing
        slow_nodes = [n for n in ring_nodes if n.throughput < 0.5]
        new_staleness = min(1.0,
            self.job.gradient_staleness
            + len(slow_nodes) * 0.05
            + (1.0 - ring_throughput) * 0.03
        )
        if ring_throughput > 0.8 and not slow_nodes:
            new_staleness = max(0.0, new_staleness - 0.1)

        # 4. Update training job
        self.job.tick(
            active_nodes=active_count,
            total_nodes=self.n_nodes,
            gradient_staleness=new_staleness
        )

        # 5. Generate alerts
        self._generate_alerts(ring_nodes)

        return self.get_state()

    def _generate_alerts(self, ring_nodes: list):
        """
        Generate human-readable alerts for the LLM agent.
        Phase-aware: different guidance per phase for hard task.
        False alarm nodes generate INFO alerts (looks suspicious).
        """
        self.alerts = []

        # Phase header for hard task
        if self.task_id == "hard":
            phase_labels = {
                1: "PHASE 1 (Warmup)",
                2: "PHASE 2 (Active Training)",
                3: "PHASE 3 (Optimization)"
            }
            self.alerts.append(
                f"[{phase_labels[self.current_phase]}] "
                f"Step {self.current_step}"
            )

        for node in self.nodes.values():
            if node.status == "crashed":
                self.alerts.append(
                    f"CRITICAL: {node.id} has crashed and is out of ring")
            elif node.status == "oom":
                self.alerts.append(
                    f"WARNING: {node.id} memory critical "
                    f"({node.memory:.0%})")
            elif node.status == "slow":
                self.alerts.append(
                    f"WARNING: {node.id} running slow "
                    f"({node.throughput:.0%} throughput)")
            elif node.memory > 0.75:
                # This fires for both real OOM early warning AND false alarm
                # Agent cannot distinguish — must reason from behavior
                self.alerts.append(
                    f"INFO: {node.id} memory elevated "
                    f"({node.memory:.0%})")

        if self.job.loss_diverging:
            self.alerts.append(
                f"CRITICAL: Loss diverging! "
                f"Current={self.job.loss:.3f} "
                f"Expected={self.job.expected_loss:.3f}")

        if self.job.gradient_staleness > 0.5:
            self.alerts.append(
                f"WARNING: High gradient staleness "
                f"({self.job.gradient_staleness:.0%})")

        if self.job.cluster_throughput < 0.5:
            self.alerts.append(
                f"WARNING: Cluster throughput degraded to "
                f"{self.job.cluster_throughput:.0%} of baseline")

        # Phase 3 specific: cost efficiency guidance
        if self.task_id == "hard" and self.current_phase == 3:
            if self._unnecessary_restarts > 0:
                self.alerts.append(
                    f"INFO: {self._unnecessary_restarts} unnecessary "
                    f"restart(s) detected — efficiency penalty active")
            if self._checkpoints_used > 2:
                self.alerts.append(
                    f"INFO: {self._checkpoints_used} checkpoints used "
                    f"— consider reducing checkpoint frequency")

    def get_state(self) -> dict:
        """Return full observable state for OpenEnv Observation."""
        return {
            "step": self.current_step,
            "nodes": [n.to_state() for n in self.nodes.values()],
            "job": self.job.to_state(),
            "ring_order": [
                nid for nid in self.ring_order
                if self.nodes[nid].in_ring
            ],
            "alerts": self.alerts,
            "root_cause_fixed": self._root_cause_fixed,
            # NEW: phase info for hard task
            "current_phase": self.current_phase,
            "phase_1_checkpointed": self._phase_1_checkpointed,
            "false_alarm_restarted": self._false_alarm_restarted,
            "unnecessary_restarts": self._unnecessary_restarts,
        }

    def is_healthy(self) -> bool:
        """
        True if cluster has recovered to acceptable health.
        Used by tasks to determine episode done condition.
        """
        active = sum(
            1 for n in self.nodes.values()
            if n.in_ring and n.status == "healthy"
        )
        return (
            active >= self.n_nodes * 0.75
            and self.job.cluster_throughput >= 0.7
            and not self.job.loss_diverging
        )

    def is_phase_complete(self) -> bool:
        """
        For hard task: returns True when current phase is done.
        Phase 1 ends at step 30.
        Phase 2 ends at step 90.
        Phase 3 ends at step 120 (max steps).
        """
        if self.task_id != "hard":
            return False
        if self.current_phase == 1:
            return self.current_step >= HARD_PHASE_1_END
        if self.current_phase == 2:
            return self.current_step >= HARD_PHASE_2_END
        return False
