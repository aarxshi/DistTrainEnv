
"""
environment/models.py — OpenEnv-compliant Pydantic models.

These models define the exact structure of:
- Observation: what the agent sees each step
- Action: what the agent can do
- Reward: what score the agent receives

openenv validate checks these are typed correctly.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# sub-models (used inside Observation)

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


# core OpenEnv Models

class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    This is everything the agent can see.
    """

    nodes: List[NodeObservation] = Field(
        description="State of all worker nodes in the cluster"
    )

    job: JobObservation = Field(
        description="Current state of the distributed training job"
    )

    ring_order: List[str] = Field(
        description="Active node IDs currently in the all-reduce ring"
    )

    alerts: List[str] = Field(
        default=[],
        description="Human-readable alerts about cluster health"
    )

    step: int = Field(
        ge=0,
        description="Current environment timestep"
    )

    task_id: str = Field(
        default="easy",
        description="Which task is currently running: easy/medium/hard"
    )


class Action(BaseModel):
    """
    Action the agent wants to take this step.
    action_type is always required.
    target_node is required for node-specific actions.
    """

    action_type: Literal[
        "restart_node",
        "remove_from_ring",
        "reduce_batch",
        "checkpoint",
        "inspect",
        "noop"
    ] = Field(
        description="Type of action to perform"
    )

    target_node: Optional[str] = Field(
        default=None,
        description="Node ID to act on. Required for node-specific actions."
    )

    parameters: Optional[dict] = Field(
        default=None,
        description="Optional extra parameters."
    )


class Reward(BaseModel):
    """
    Structured reward returned every step.
    value is the scalar reward for RL training.
    Sub-scores provide transparency for debugging.
    """

    value: float = Field(
        description="Total reward for this step."
    )

    throughput_score: float = Field(
        ge=0.0, le=1.0,
        description="Reward from cluster throughput health."
    )

    loss_health_score: float = Field(
        ge=0.0, le=1.0,
        description="Reward from loss staying on trajectory."
    )

    early_detection_bonus: float = Field(
        default=0.0,
        description="Bonus for acting before fault becomes critical."
    )

    causal_fix_bonus: float = Field(
        default=0.0,
        description="Bonus for fixing root cause vs symptom."
    )

    penalty: float = Field(
        default=0.0,
        description="Total penalties this step."
    )


class StepResult(BaseModel):
    """Full result returned by step() endpoint."""

    observation: Observation
    reward: Reward
    done: bool = Field(
        description="True if episode is complete."
    )
    info: dict = Field(
        default={},
        description="Extra diagnostic info."
    )
