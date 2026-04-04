
"""
environment/reward.py — Reward computation engine for DistTrainEnv.
"""

from environment.ring_cluster import RingCluster


class RewardEngine:
    """
    Computes shaped reward after each step.
    Instantiated once per episode by env.py.
    """

    def __init__(self, cluster: RingCluster):
        self.cluster = cluster
        self._steps_degraded = 0
        self._prev_throughput = 1.0
        self._prev_loss_health = 1.0

    def compute(
        self,
        action_type: str,
        target_node: str,
        action_success: bool,
        prev_state: dict,
        curr_state: dict,
    ) -> dict:
        """
        Compute full shaped reward for one step.

        Returns dict with value and all sub-components.
        """
        job = self.cluster.job

        # --------------------------------------------------
        # 1. Throughput Score (0.0 - 1.0)
        # --------------------------------------------------
        throughput_score = job.cluster_throughput

        # --------------------------------------------------
        # 2. Loss Health Score (0.0 - 1.0)
        # --------------------------------------------------
        loss_health_score = job.loss_health()

        # --------------------------------------------------
        # 3. Early Detection Bonus
        # rewards acting on a node BEFORE it becomes critical.
        # key differentiator — catches the hard task cascade early.
        # --------------------------------------------------
        early_detection_bonus = 0.0

        if action_type in ["reduce_batch", "remove_from_ring"]:
            if target_node and target_node in self.cluster.nodes:
                node = self.cluster.nodes[target_node]
                # memory elevated but not yet critical
                if 0.65 < node.memory < 0.90:
                    early_detection_bonus = 0.3
                # node slow but not yet crashed
                elif node.status == "slow" and node._oom_progress < 0.5:
                    early_detection_bonus = 0.2

        # --------------------------------------------------
        # 4. Causal Fix Bonus
        # large bonus for fixing ROOT CAUSE.
        # small bonus for fixing a symptom.
        # creates meaningful score gap on the hard task.
        # --------------------------------------------------
        causal_fix_bonus = 0.0

        if action_type in ["restart_node", "reduce_batch"]:
            root_cause = self.cluster.fault_injector.get_root_cause_node()
            if target_node == root_cause:
                causal_fix_bonus = 0.5      # fixed the actual root cause
            elif target_node and target_node in self.cluster.nodes:
                node = self.cluster.nodes[target_node]
                if node.status in ["slow", "oom", "crashed"]:
                    causal_fix_bonus = 0.1  # fixed a symptom, still helpful

        # --------------------------------------------------
        # penalties
        # --------------------------------------------------
        penalty = 0.0

        # restarting a healthy node — explicitly penalized in spec
        if action_type == "restart_node" and target_node:
            node = self.cluster.nodes.get(target_node)
            if node and node.status == "healthy":
                penalty -= 0.4

        # urgency scaling — penalty grows the longer cluster is degraded
        # encourages the agent to act quickly, not wait
        if not self.cluster.is_healthy():
            self._steps_degraded += 1
            urgency_penalty = -min(0.10, 0.02 * self._steps_degraded)
            penalty += urgency_penalty
        else:
            self._steps_degraded = max(0, self._steps_degraded - 1)

        # catastrophe — loss fully diverged
        if job.loss_diverging and job.loss > job.expected_loss + 0.4:
            penalty -= 0.3

        # invalid action
        if not action_success:
            penalty -= 0.1

        # --------------------------------------------------
        # total reward
        # --------------------------------------------------
        value = (
            0.35 * throughput_score
            + 0.35 * loss_health_score
            + early_detection_bonus
            + causal_fix_bonus
            + penalty
        )

        self._prev_throughput = throughput_score
        self._prev_loss_health = loss_health_score

        return {
            "value": round(value, 4),
            "throughput_score": round(throughput_score, 4),
            "loss_health_score": round(loss_health_score, 4),
            "early_detection_bonus": round(early_detection_bonus, 4),
            "causal_fix_bonus": round(causal_fix_bonus, 4),
            "penalty": round(penalty, 4),
        }

    def reset(self):
        """Reset tracking state for new episode."""
        self._steps_degraded = 0
        self._prev_throughput = 1.0
        self._prev_loss_health = 1.0
