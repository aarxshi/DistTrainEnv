
"""
environment/job.py — Distributed training job state and loss dynamics.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class TrainingJob:
    """
    Simulates the ML training job running across the cluster.
    Loss curve: L(t) = (L0 - L_floor) * exp(-k * t) + L_floor
    """

    step: int = 0

    # loss curve parameters
    _initial_loss: float = field(default=2.5, repr=False)
    _decay_rate: float = field(default=0.015, repr=False)
    _loss_floor: float = field(default=0.8, repr=False)

    # current values
    loss: float = 2.5
    expected_loss: float = 2.5

    # throughput
    cluster_throughput: float = 1.0

    # gradient health
    gradient_staleness: float = 0.0

    # divergence tracking
    loss_diverging: bool = False
    _divergence_counter: int = field(default=0, repr=False)

    def expected_loss_at(self, step: int) -> float:
        """
        Compute healthy expected loss at a given step.
        Used for reward computation and grader scoring.
        """
        return (
            (self._initial_loss - self._loss_floor)
            * math.exp(-self._decay_rate * step)
            + self._loss_floor
        )

    def tick(self, active_nodes: int, total_nodes: int,
             gradient_staleness: float, noise: float = 0.005):
        """
        Advance training job by one step.

        Args:
            active_nodes: nodes currently in ring
            total_nodes: total nodes in cluster
            gradient_staleness: current staleness 0.0-1.0
            noise: small random perturbation for realism
        """

        self.step += 1
        self.gradient_staleness = gradient_staleness

        # expected healthy loss at this step
        self.expected_loss = self.expected_loss_at(self.step)

        # cluster throughput ratio
        if total_nodes > 0:
            self.cluster_throughput = max(0.0,
                active_nodes / total_nodes)
        else:
            self.cluster_throughput = 0.0

        # --- Loss deviation from faults ---
        # staleness: stale gradients push loss UP
        staleness_penalty = gradient_staleness * 0.08

        # throughput lag: fewer updates = slower loss decay
        throughput_lag = (1.0 - self.cluster_throughput) * 0.05

        # target loss with fault effects
        fault_target = self.expected_loss + staleness_penalty + throughput_lag

        # smooth transition — realistic momentum
        self.loss = (
            0.85 * self.loss
            + 0.15 * fault_target
            + random.uniform(-noise, noise)
        )
        self.loss = max(self._loss_floor - 0.1, self.loss)

        # --- Divergence detection ---
        deviation = self.loss - self.expected_loss
        if deviation > 0.25:
            self._divergence_counter += 1
        else:
            self._divergence_counter = max(0, self._divergence_counter - 1)

        self.loss_diverging = self._divergence_counter >= 3

    def loss_health(self) -> float:
        """
        Returns 0.0-1.0 score of how healthy the loss curve is.
        1.0 = perfectly on trajectory, 0.0 = fully diverged.
        """
        deviation = abs(self.loss - self.expected_loss)
        return max(0.0, 1.0 - (deviation / 0.5))

    def to_state(self) -> dict:
        return {
            "step": self.step,
            "loss": round(self.loss, 4),
            "expected_loss": round(self.expected_loss, 4),
            "cluster_throughput": round(self.cluster_throughput, 3),
            "gradient_staleness": round(self.gradient_staleness, 3),
            "loss_diverging": self.loss_diverging,
        }
