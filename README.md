# DistTrainEnv

An OpenEnv reinforcement learning environment simulating a distributed ML training cluster under fault conditions. An AI agent observes cluster metrics and must detect, diagnose, and recover from faults, keeping the training job healthy.

- Weighted Score (LLaMA-3.3-70B via Groq): **0.9755**
- Environment: 8-node ring all-reduce cluster
- Tasks: 3 (easy / medium / hard)
- Constraint: runs on 2 vCPU / 8 GB RAM

Built for the [Meta PyTorch OpenEnv Hackathon](https://openenv.dev).

---

## 1. Background

In distributed ML training, multiple GPU nodes collaborate using **ring all-reduce** to synchronize gradients across workers. Ring throughput is bounded by the slowest active node, meaning a single fault can degrade the entire cluster silently or catastrophically.

Common failure modes:

- **Node crash** — breaks the ring, training halts or produces stale gradients
- **Straggler** — one slow node throttles all-reduce for everyone
- **Memory OOM** — silent at first, memory climbs gradually until the node slows, then cascades into retries that overload other nodes

Today, engineers monitor dashboards and intervene manually. This environment is designed to train and evaluate RL agents to do that instead, detecting faults early and taking the right action, not just the obvious one.

---

## 2. Environment Design

```
┌──────────────────────────┐     ┌─────────────────────────────┐     ┌─────────────────────────┐
│      Fault Injector      │────▶│       Ring Cluster Sim      │────▶│       LLM / RL Agent    │
│                          │     │                             │     │                         │
│ • Deterministic per task │     │ • 8 nodes, ring topology    │     │ • Reads JSON observation│
│ • crash / straggler / OOM│     │ • Throughput = min(nodes)   │     │ • Outputs JSON action   │
│ • Root cause + symptoms  │     │ • Loss curve dynamics       │     │ • [START][STEP][END] log│
└──────────────────────────┘     │ • Gradient staleness        │     └─────────────────────────┘
                                 └──────────────┬──────────────┘
                                                │
                         ┌──────────────────────▼──────────────────────┐
                         │                   Graders                   │
                         │                                             │
                         │  easy   → throughput + loss + step speed    │
                         │  medium → detection speed + throughput      │
                         │  hard   → root cause fix + early detection  │
                         └─────────────────────────────────────────────┘
```

The simulation is pure Python — no real Docker containers, no GPU, no external APIs in the environment. One episode runs in milliseconds.

---

## 3. Observation Space

Each step the agent receives a full cluster snapshot:

```json
{
  "nodes": [
    {
      "id": "node_2",
      "status": "slow",
      "memory": 0.81,
      "throughput": 0.6,
      "latency": 5.2,
      "in_ring": true
    }
  ],
  "job": {
    "step": 4,
    "loss": 2.4471,
    "expected_loss": 2.3812,
    "cluster_throughput": 0.75,
    "gradient_staleness": 0.12,
    "loss_diverging": false
  },
  "ring_order": ["node_0", "node_1", "node_2", "node_3", "node_4", "node_5", "node_6"],
  "alerts": ["INFO: node_2 memory elevated (81%)"],
  "step": 4,
  "task_id": "hard"
}
```

Key signals:

| Signal | Threshold | Meaning |
|---|---|---|
| `memory` | > 0.75 | early OOM warning — act now |
| `memory` | > 0.90 | critical — crash imminent |
| `gradient_staleness` | > 0.30 | training degrading silently |
| `loss_diverging` | true | serious — immediate action required |
| `alerts` | — | named nodes with problems |

Node statuses: `healthy` → `slow` → `oom` → `crashed`

---

## 4. Action Space

| Action | Description | Requires `target_node` |
|---|---|---|
| `restart_node` | bring a crashed node back online | yes |
| `remove_from_ring` | remove a slow node from all-reduce | yes |
| `reduce_batch` | halve batch size to ease memory pressure | yes |
| `checkpoint` | save training state (halves throughput this step) | no |
| `inspect` | get diagnostics on a specific node | yes |
| `noop` | do nothing | no |

---

## 5. Reward Function

Dense signal every step, the agent doesn't wait until episode end to learn:

```
reward = 0.35 × throughput_score        # cluster steps/sec vs baseline
       + 0.35 × loss_health_score        # actual loss vs expected trajectory
       + early_detection_bonus           # acted before fault became critical
       + causal_fix_bonus                # fixed root cause (not just symptom)
       + penalty                         # restarting healthy nodes, urgency scaling, invalid actions
```

The `early_detection_bonus` and `causal_fix_bonus` are what create meaningful score variance between naive and smart agents especially on the hard task.

---

## 6. Tasks

### Easy — Node Crash Recovery

`node_3` crashes at step 1. The alert is explicit. The agent must detect and restart or remove the node to restore ring health.

- Max steps: 15 | Success threshold: 0.70
- Grading: throughput recovery (40%), loss health (30%), steps to recovery (20%), ring integrity (10%)

### Medium — Straggler Detection

`node_5` runs at 30% speed from step 1. It shows as `slow`, not `crashed` — no crash alert fires. The agent must identify the bottleneck from throughput and latency signals and remove it from the ring.

- Max steps: 20 | Success threshold: 0.60
- Grading: throughput recovery (35%), detection speed (30%), loss health (25%), ring integrity (10%)

### Hard — Cascading OOM Fault Recovery

`node_2` begins running out of memory at step 1 (root cause — silent, memory climbs at ~6% per step). At step 5, `node_7` becomes a straggler as a downstream symptom of `node_2`'s retry storm. Gradient staleness builds and loss starts diverging.

A naive agent fixes `node_7` (the visible symptom) and scores ~0.30 on root cause. A smart agent traces the causal chain to `node_2`, acts early (before memory exceeds 0.90), and scores full marks.

```
node_2 OOM (step 1, silent) ──▶ slow + retries ──▶ node_7 straggler (step 5)
                                                           │
                                               gradient staleness builds
                                                           │
                                                   loss diverges
```

- Max steps: 25 | Success threshold: 0.50
- Grading: root cause fixed (35%), loss + staleness health (30%), throughput recovery (20%), early detection (15%)

---

## 7. Evaluation Results

Scores from running `llama-3.3-70b-versatile` via Groq API as the agent:

| Task | Score |
|---|---|
| easy | 0.9694 |
| medium | 0.9846 |
| hard | 0.9713 |
| **weighted** | **0.9755** |

Weights: easy 0.25, medium 0.35, hard 0.40

---

## 8. Folder Structure

```
DistTrainEnv/
│
├── environment/
│   ├── schema.py          # shared data contract (NodeState, JobState, Action, Reward)
│   ├── node.py            # single worker node state machine
│   ├── job.py             # training job + loss curve dynamics
│   ├── faults.py          # fault injection engine
│   ├── ring_cluster.py    # main simulation (ties it all together)
│   ├── reward.py          # shaped reward computation
│   ├── models.py          # openenv-compliant pydantic models
│   └── env.py             # openenv api (reset / step / state)
│
├── tasks/
│   └── task_configs.py    # task metadata + grading weights
│
├── graders/
│   ├── grader_easy.py
│   ├── grader_medium.py
│   ├── grader_hard.py
│   └── run_graders.py     # unified grader entry point
│
├── inference.py           # llm agent loop ([START] [STEP] [END] format)
├── app.py                 # fastapi server (openenv http api)
├── openenv.yaml           # openenv metadata spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 9. Running the Project

**Step 1 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2 — Dry run (no API key needed)**

```bash
python inference.py --dry-run
```

Uses a rule-based agent to verify the full `[START]` → `[STEP] step=N` → `[END]` log format

**Step 3 — Run with a real LLM**

```bash
GROQ_API_KEY=gsk_... python inference.py
GROQ_API_KEY=gsk_... python inference.py --task hard
```

**Step 4 — Start the API server**

```bash
python app.py
```

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | reset environment, returns initial observation |
| POST | `/step` | apply action, returns observation + reward + done |
| GET | `/state` | full internal state |
| GET | `/health` | health check |
| GET | `/tasks` | list available tasks |

**Step 5 — Docker**

```bash
docker build -t disttrainenv .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... disttrainenv
```

---

## 10. Conclusion

This environment demonstrates that meaningful distributed systems fault recovery can be framed as an RL problem with clean, deterministic grading. The key design choices dense reward shaping, a causal fault cascade on the hard task, and early detection bonuses: create a real gap between naive and intelligent agents, making the environment genuinely useful for evaluating LLM reasoning quality.

The simulation runs entirely in-process Python with no external dependencies at runtime, making it reproducible and deployable within strict compute budgets (2 vCPU / 8 GB RAM).
