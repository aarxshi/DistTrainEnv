
"""
environment/reward.py — Reward computation engine for DistTrainEnv.

Finale upgrades:
- Noop floor removed — agent must act to score well
- False alarm penalty wired from cluster state
- Phase-aware bonuses for hard task (Phase 1/2/3)
- Intermittent straggler handling
- Urgency scaling tuned for 60/120 step episodes
- Causal reasoning heavily rewarded over symptom fixing
- Maintenance bonus — rewards holding a healthy cluster
- Clean run bonus for Phase 3 efficiency
- Success signal for Phase 3 termination
"""

from environment.ring_cluster import RingCluster


class RewardEngine:
    """
    Computes shaped reward after each step.
    Instantiated once per episode by env.py.

    Reward ranges (approximate):
    - Noop on healthy cluster:          ~0.40  (acceptable)
    - Noop on sick cluster:             ~0.10  (bad — agent must act)
    - Fix symptom node:                 ~0.75  (good)
    - Fix root cause early:             ~1.35  (great)
    - Maintain healthy in Phase 2:      ~0.55  (correct behavior)
    - Reach Phase 3 healthy:            ~0.80+ (success signal)
    - False alarm restart:              ~-0.10 (punished)
    - Loss diverged + nodes critical:   ~-0.40 (catastrophe)
    """

    def __init__(self, cluster: RingCluster):
        self.cluster = cluster
        self._steps_degraded = 0
        self._prev_throughput = 1.0
        self._prev_loss_health = 1.0
        self._root_cause_fixed_step = -1
        self._false_alarm_penalty_applied = False

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
        current_step = self.cluster.current_step
        current_phase = self.cluster.current_phase
        task_id = self.cluster.task_id
        is_healthy = self.cluster.is_healthy()

        # --------------------------------------------------
        # 1. Base Scores (kept small so bonuses dominate)
        # --------------------------------------------------
        throughput_score = job.cluster_throughput
        loss_health_score = job.loss_health()

        # --------------------------------------------------
        # 2. Throughput Delta Bonus
        # Rewards active improvement, not just passive health
        # --------------------------------------------------
        throughput_delta = throughput_score - self._prev_throughput
        delta_bonus = max(0.0, throughput_delta) * 0.5

        # --------------------------------------------------
        # 3. Early Detection Bonus
        # Rewards catching faults before they become critical
        # Key signal for hard task OOM cascade detection
        # --------------------------------------------------
        early_detection_bonus = 0.0

        if action_type in ["reduce_batch", "remove_from_ring", "restart_node"]:
            if target_node and target_node in self.cluster.nodes:
                node = self.cluster.nodes[target_node]

                # Caught OOM early — memory elevated but not yet critical
                if 0.65 < node.memory < 0.88 and node._fault_type == "oom":
                    early_detection_bonus = 0.35

                # Caught straggler before loss diverged
                elif (node.status == "slow"
                      and not job.loss_diverging
                      and node._oom_progress < 0.4):
                    early_detection_bonus = 0.25

                # Acted before full crash — memory high but running
                elif 0.88 <= node.memory < 0.95:
                    early_detection_bonus = 0.15

        # --------------------------------------------------
        # 4. Causal Fix Bonus
        # Root cause fix >> symptom fix >> noop
        # PRIMARY training signal — must dominate reward
        # --------------------------------------------------
        causal_fix_bonus = 0.0
        root_cause = self.cluster.fault_injector.get_root_cause_node()

        if action_type in ["restart_node", "reduce_batch"]:
            if target_node == root_cause:
                causal_fix_bonus = 0.6
                self._root_cause_fixed_step = current_step
            elif target_node and target_node in self.cluster.nodes:
                node = self.cluster.nodes[target_node]
                if node.status in ["slow", "oom", "crashed"]:
                    causal_fix_bonus = 0.15

        if action_type == "remove_from_ring" and target_node:
            node = self.cluster.nodes.get(target_node)
            if node and node.status == "slow":
                # Correct handling of straggler
                causal_fix_bonus = 0.20
                if target_node == root_cause:
                    causal_fix_bonus = 0.45

        # --------------------------------------------------
        # 5. Maintenance Bonus
        # Rewards holding a healthy cluster across phases
        # Prevents reward collapse when agent correctly
        # maintains a recovered cluster while waiting for
        # Phase 3 — critical for hard task training signal
        # --------------------------------------------------
        maintenance_bonus = 0.0

        if is_healthy:
            if task_id == "hard":
                if current_phase == 1:
                    # Healthy warmup — agent monitoring correctly
                    maintenance_bonus = 0.10
                elif current_phase == 2:
                    # Recovered during fault storm and holding it
                    # Strong positive — this is correct behavior
                    maintenance_bonus = 0.20
                elif current_phase == 3:
                    # Made it to stabilization healthy
                    # Strongest signal — this is the success state
                    maintenance_bonus = 0.35
            else:
                # Easy/medium healthy cluster
                maintenance_bonus = 0.15

        # --------------------------------------------------
        # 6. Phase-Aware Bonuses (hard task only)
        # --------------------------------------------------
        phase_bonus = 0.0

        if task_id == "hard":
            if current_phase == 1:
                # Proactive checkpointing in warmup
                if action_type == "checkpoint":
                    phase_bonus = 0.25
                # Inspecting nodes during warmup — good practice
                elif action_type == "inspect":
                    phase_bonus = 0.05

            elif current_phase == 2:
                # Speed bonus — faster root cause fix = bigger reward
                if (self._root_cause_fixed_step > 0
                        and current_step - self._root_cause_fixed_step < 5):
                    phase_bonus = 0.20
                # Surviving fault storm with low staleness
                if job.gradient_staleness < 0.3 and is_healthy:
                    phase_bonus += 0.15

            elif current_phase == 3:
                # Efficiency bonuses — Phase 3 judges cost not just survival
                if self.cluster._unnecessary_restarts == 0:
                    phase_bonus = 0.20    # perfect run
                elif self.cluster._unnecessary_restarts == 1:
                    phase_bonus = 0.05   # one mistake

                # Low checkpoint usage — cost efficient
                if self.cluster._checkpoints_used <= 1:
                    phase_bonus += 0.10
                elif self.cluster._checkpoints_used <= 3:
                    phase_bonus += 0.05

                # Stable high throughput in Phase 3
                if throughput_score > 0.85 and not job.loss_diverging:
                    phase_bonus += 0.15

        # --------------------------------------------------
        # 7. Penalties
        # --------------------------------------------------
        penalty = 0.0

        # --- False alarm penalty ---
        # Applied once per episode — agent panicked on healthy node
        if (self.cluster._false_alarm_restarted
                and not self._false_alarm_penalty_applied):
            penalty -= 0.5
            self._false_alarm_penalty_applied = True

        # Repeated unnecessary restarts
        if self.cluster._unnecessary_restarts > 1:
            penalty -= 0.15 * (self.cluster._unnecessary_restarts - 1)

        # --- Restarting a healthy non-false-alarm node ---
        if action_type == "restart_node" and target_node:
            node = self.cluster.nodes.get(target_node)
            false_alarm_id = self.cluster._false_alarm_node_id
            if (node
                    and node.status == "healthy"
                    and target_node != false_alarm_id):
                penalty -= 0.3

        # --- Urgency penalty ---
        # Grows the longer cluster stays degraded
        # Punishes passive noop on a sick cluster
        if not is_healthy:
            self._steps_degraded += 1
            urgency_penalty = -min(0.15, 0.02 * self._steps_degraded)
            penalty += urgency_penalty
        else:
            self._steps_degraded = max(0, self._steps_degraded - 2)

        # --- Noop on degraded cluster ---
        # Explicit punishment for doing nothing when sick
        # This is what was missing — noop was too profitable before
        if action_type == "noop" and not is_healthy:
            penalty -= 0.15

        # --- Catastrophe penalty ---
        if job.loss_diverging and job.loss > job.expected_loss + 0.4:
            penalty -= 0.4

        # --- Gradient staleness penalty ---
        # Silent killer — agent must act before staleness compounds
        if job.gradient_staleness > 0.6:
            penalty -= 0.10
        if job.gradient_staleness > 0.85:
            penalty -= 0.20  # stacked — very high staleness catastrophic

        # --- Invalid action ---
        if not action_success:
            penalty -= 0.10

        # --------------------------------------------------
        # 8. Total Reward
        #
        # Noop healthy:              0.20*T + 0.20*L + 0.15 maint ~ 0.40
        # Noop sick:                 0.20*T + 0.20*L - 0.15 noop
        #                            - urgency                     ~ 0.10
        # Fix symptom:               base + 0.25 early + 0.15 caus ~ 0.75
        # Fix root cause early:      base + 0.35 early + 0.60 caus ~ 1.35
        # Phase 3 healthy + clean:   base + 0.35 maint + 0.20 ph3  ~ 0.90
        # --------------------------------------------------
        value = (
            0.20 * throughput_score
            + 0.20 * loss_health_score
            + delta_bonus
            + early_detection_bonus
            + causal_fix_bonus
            + maintenance_bonus
            + phase_bonus
            + penalty
        )

        # Clip to prevent extreme outliers destabilizing GRPO
        value = max(-1.0, min(2.0, value))

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
        self._root_cause_fixed_step = -1
        self._false_alarm_penalty_applied = False
