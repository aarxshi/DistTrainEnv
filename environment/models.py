
"""
environment/models.py — OpenEnv-compliant Pydantic models.

Finale upgrades:
- Added current_phase to Observation (hard task phase tracking)
- Added false_alarm_restarted to Observation
- Added unnecessary_restarts to Observation
- All other models unchanged for backward compatibility
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class NodeObservation(BaseModel):
    """Observable state of one worker node."""
    id: str
    status: Literal["healthy", "slow", "oom", "crashed"]
    memory: float = Field(ge=0.0, le=1.0)
    throughput: float = Field(ge=0.0, le=1.0)
    latency: float = Field(ge=0.0)
    in_ring: bool


class JobObservation(BaseModel):
    """Observable state of the training job."""
    step: int = Field(ge=0)
    loss: float
    expected_loss: float
    cluster_throughput: float = Field(ge=0.0)
    gradient_staleness: float = Field(ge=0.0, le=1.0)
    loss_diverging: bool


class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    Everything the agent can see.
    """
    nodes: List[NodeObservation] = Field(
        description="State of all worker nodes"
    )
    job: JobObservation = Field(
        description="Current training job state"
    )
    ring_order: List[str] = Field(
        description="Active node IDs in the all-reduce ring"
    )
    alerts: List[str] = Field(
        default=[],
        description="Human-readable cluster health alerts"
    )
    step: int = Field(ge=0)
    task_id: str = Field(default="easy")

    # NEW: phase tracking fields
    current_phase: int = Field(
        default=1,
        description="Current phase (1/2/3). Only meaningful for hard task."
    )
    false_alarm_restarted: bool = Field(
        default=False,
        description="True if agent restarted the false alarm node."
    )
    unnecessary_restarts: int = Field(
        default=0,
        description="Count of unnecessary restarts this episode."
    )


class Action(BaseModel):
    """Action the agent wants to take this step."""
    action_type: Literal[
        "restart_node",
        "remove_from_ring",
        "reduce_batch",
        "checkpoint",
        "inspect",
        "noop"
    ] = Field(description="Type of action to perform")

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


class StepResult(BaseModel):
    """Full result returned by step() endpoint."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default={})
