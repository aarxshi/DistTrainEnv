
"""
schema.py — Shared data contract for DistTrainEnv.
All modules import from here. Do not redefine these elsewhere.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class NodeState(BaseModel):
    """Represents the observable state of one worker node."""

    id: str

    status: Literal[
        "healthy",
        "slow",       # throughput degraded, still in ring
        "oom",        # memory critical, risk of crash
        "crashed"     # out of ring, not contributing
    ]

    memory: float = Field(
        ge=0.0, le=1.0,
        description="GPU memory usage, 0.0=empty 1.0=full"
    )

    throughput: float = Field(
        ge=0.0, le=1.0,
        description="Normalized steps/sec, 1.0=healthy baseline"
    )

    latency: float = Field(
        ge=0.0,
        description="Network latency in ms to next node in ring"
    )

    in_ring: bool = Field(
        default=True,
        description="Whether this node is part of the active ring"
    )


class JobState(BaseModel):
    """Represents the state of the distributed training job."""

    step: int = Field(ge=0)

    loss: float

    expected_loss: float = Field(
        description="What loss SHOULD be at this step"
    )

    cluster_throughput: float = Field(ge=0.0)

    gradient_staleness: float = Field(ge=0.0, le=1.0)

    loss_diverging: bool = Field(default=False)


class Action(BaseModel):
    """All actions the agent can take."""

    action_type: Literal[
        "restart_node",
        "remove_from_ring",
        "reduce_batch",
        "checkpoint",
        "inspect",
        "noop"
    ]

    target_node: Optional[str] = Field(
        default=None,
        description="Node ID to act on."
    )

    parameters: Optional[dict] = Field(default=None)


class Reward(BaseModel):
    """Structured reward returned every step."""

    value: float

    throughput_score: float = Field(ge=0.0, le=1.0)

    loss_health_score: float = Field(ge=0.0, le=1.0)

    early_detection_bonus: float = Field(default=0.0)

    causal_fix_bonus: float = Field(default=0.0)

    penalty: float = Field(default=0.0)
