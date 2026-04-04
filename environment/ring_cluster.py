
"""
environment/ring_cluster.py — Main distributed training cluster simulator.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from environment.node import Node
from environment.job import TrainingJob
from environment.faults import FaultInjector, FaultEvent


class RingCluster:
    """
    Simulates a ring all-reduce distributed training cluster.

    Ring all-reduce speed is bounded by the SLOWEST node.
    If a node crashes, the ring breaks until reformed.
    """

    def __init__(
        self,
        n_nodes: int = 8,
        fault_events: List[FaultEvent] = None,
    ):
        self.n_nodes = n_nodes
        self.current_step = 0
        self._root_cause_fixed = False
        self.alerts: List[str] = []
        self._last_action = None

        self.nodes: Dict[str, Node] = {
            f"node_{i}": Node(id=f"node_{i}")
            for i in range(n_nodes)
        }

        self.ring_order: List[str] = [
            f"node_{i}" for i in range(n_nodes)
        ]

        self.job = TrainingJob()

        self.fault_injector = FaultInjector(
            fault_events=fault_events or []
        )

    def reset(self, fault_events: List[FaultEvent] = None):
        """
        Reset cluster to healthy state for new episode.
        """
        self.current_step = 0
        self._root_cause_fixed = False
        self.alerts = []
        self._last_action = None

        self.nodes = {
            f"node_{i}": Node(id=f"node_{i}")
            for i in range(self.n_nodes)
        }

        self.ring_order = [f"node_{i}" for i in range(self.n_nodes)]
        self.job = TrainingJob()

        if fault_events is not None:
            self.fault_injector = FaultInjector(fault_events)
        else:
            self.fault_injector.reset()

    def apply_action(
        self,
        action_type: str,
        target_node: Optional[str] = None,
        parameters: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Apply agent action BEFORE ticking.
        Returns (success, message).
        """
        self._last_action = action_type
        self.alerts = []

        if action_type == "noop":
            return True, "No action taken."

        if action_type == "checkpoint":
            self.job.cluster_throughput *= 0.5
            return True, "Checkpoint saved. Throughput halved this step."

        if action_type == "inspect":
            if target_node and target_node in self.nodes:
                return True, f"Inspecting {target_node}."
            return False, f"Node {target_node} not found."

        if not target_node or target_node not in self.nodes:
            return False, f"Invalid target node: {target_node}"

        node = self.nodes[target_node]

        if action_type == "restart_node":
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
        Order: inject faults → update nodes → compute ring health → update job
        """
        self.current_step += 1

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

        # ring throughput = slowest node (synchronous all-reduce)
        if ring_nodes:
            ring_throughput = min(n.throughput for n in ring_nodes)
        else:
            ring_throughput = 0.0

        # gradient staleness grows when nodes are slow or missing
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
        Includes early warnings so agent can act before crisis.
        """
        self.alerts = []

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
                # early warning — key signal for hard task
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
