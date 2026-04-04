
"""
app.py — FastAPI server exposing DistTrainEnv as OpenEnv HTTP API.

Endpoints:
    POST /reset          — reset environment, returns Observation
    POST /step           — apply action, returns StepResult
    GET  /state          — get current state, returns dict
    GET  /health         — health check for HF Space ping
    GET  /tasks          — list available tasks

The validator script pings /reset with an empty POST.
Must return HTTP 200 with valid Observation JSON.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment.env import DistTrainEnv
from environment.models import Action, Observation, StepResult

app = FastAPI(
    title="DistTrainEnv",
    description="Distributed ML Training Fault Recovery Environment",
    version="1.0.0",
)

# allow CORS for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# global environment instance
# one instance per server — resets between episodes
env = DistTrainEnv(task_id="easy")


# request models

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


class StepRequest(BaseModel):
    action_type: str = "noop"
    target_node: Optional[str] = None
    parameters: Optional[dict] = None


# endpoints

@app.get("/health")
def health():
    """
    Health check endpoint.
    HF Spaces pings this to verify the container is running.
    """
    return {"status": "ok", "environment": "DistTrainEnv"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = None):
    """
    Reset the environment to initial state.
    Optionally specify which task to run: easy / medium / hard.
    Returns initial Observation.

    This is the FIRST endpoint the validator pings.
    Must return 200 with valid Observation JSON.
    """
    task_id = "easy"
    if request and request.task_id:
        if request.task_id not in ["easy", "medium", "hard"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_id. Must be: easy, medium, hard"
            )
        task_id = request.task_id

    observation = env.reset(task_id=task_id)
    return observation


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    """
    Apply action and advance environment one step.
    Returns StepResult with observation, reward, done, info.
    """
    valid_actions = [
        "restart_node", "remove_from_ring", "reduce_batch",
        "checkpoint", "inspect", "noop"
    ]

    if request.action_type not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type. Must be one of: {valid_actions}"
        )

    action = Action(
        action_type=request.action_type,
        target_node=request.target_node,
        parameters=request.parameters,
    )

    result = env.step(action)
    return result


@app.get("/state")
def state():
    """
    Return full current environment state.
    Required by OpenEnv spec.
    """
    return env.state()


@app.get("/tasks")
def list_tasks():
    """
    List available tasks with descriptions.
    Helpful for agents to understand what they can run.
    """
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Node Crash Recovery",
                "difficulty": "easy",
                "max_steps": 15,
                "description": (
                    "A single worker node crashes. "
                    "Detect and restart it to restore ring health."
                )
            },
            {
                "id": "medium",
                "name": "Straggler Detection",
                "difficulty": "medium",
                "max_steps": 20,
                "description": (
                    "A straggler node runs at 30% speed. "
                    "Identify and remove it from the ring."
                )
            },
            {
                "id": "hard",
                "name": "Cascading OOM Recovery",
                "difficulty": "hard",
                "max_steps": 25,
                "description": (
                    "Silent OOM on node_2 cascades to straggler on node_7. "
                    "Fix the root cause, not just the symptom."
                )
            },
        ]
    }


# entry point

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
