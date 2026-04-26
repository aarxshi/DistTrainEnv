---
title: DistTrainEnv
emoji: 🖥️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# DistTrainEnv

> Can a language model save a dying cluster at 3 AM?

DistTrainEnv is a reinforcement learning environment that simulates a distributed ML training cluster under fault conditions. An LLM agent observes live cluster state and must detect, diagnose, and recover from faults — including cascading failures where the root cause and the visible symptom are on different nodes.

The project has two parts: a **prompted LLaMA-3.3-70B agent** that runs inference against the environment, and an **SFT fine-tuning pipeline** (Colab, free T4) that trains the agent to internalize correct diagnostic behavior rather than derive it from a prompt every time.

- Weighted Score (LLaMA-3.3-70B): **0.7764**
- Environment: 8-node ring all-reduce cluster
- Tasks: 3 (easy / medium / hard)
- Constraint: runs on 2 vCPU / 8 GB RAM

Built for the [Meta PyTorch OpenEnv Hackathon](https://openenv.dev).

---

## Background

In distributed ML training, multiple GPU nodes collaborate using **ring all-reduce** to synchronize gradients across workers. Ring throughput is bounded by the slowest active node — a single fault can degrade the entire cluster silently or catastrophically.

Common failure modes:

- **Node crash** — breaks the ring, training halts or produces stale gradients
- **Straggler** — one slow node throttles all-reduce for everyone
- **Memory OOM** — silent at first, memory climbs gradually until the node slows, then cascades into retries that overload other nodes

Today, engineers monitor dashboards and intervene manually. This environment trains and evaluates LLM agents to do that instead — detecting faults early and taking the right action, not just the obvious one.

---

## Environment Design

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
                         │  easy   → throughput + loss + step speed    │
                         │  medium → detection speed + throughput      │
                         │  hard   → root cause fix + early detection  │
                         └─────────────────────────────────────────────┘
```

---

## Tasks

| Task | Fault | What makes it hard |
|------|-------|--------------------|
| **Easy** | `node_3` crashes at step 1 | Alert fires explicitly. Just restart it. |
| **Medium** | `node_5` runs at 30% speed | No crash alert. Agent must read throughput and latency signals. |
| **Hard** | `node_2` OOM (silent) → `node_7` slow (symptom) | Fixing the symptom scores 0.30. Catching the root cause early scores 1.0. |

The hard task cascade:

```
node_2 OOM (step 1, silent)
    └──▶ retry storm
            └──▶ node_7 straggler (step 5)
                    └──▶ gradient staleness builds
                                └──▶ loss diverges
```

---

## Phase 1: Prompted Agent

LLaMA-3.3-70B-Instruct via HuggingFace Inference API. Per-task system prompts encode the fault logic, sort nodes by memory risk, and feed the last 5 actions with rewards back to the agent each step. Temperature 0.2.

### What actually happened

Easy and medium ran cleanly. The hard task per-step reward never settled:

![hard/step_reward](https://raw.githubusercontent.com/aarxshi/DistTrainEnv/main/assets/hard_step_reward.jpeg)

Both runs volatile — spiking between 0 and 1 with no stable trend. The root cause fix rate made the problem concrete:

![hard/ep_root_cause_fixed](https://raw.githubusercontent.com/aarxshi/DistTrainEnv/main/assets/hard_ep_root_cause_fixed.jpeg)

Base agent near zero throughout. v2 never registers a causal fix at all. Mean episode reward confirmed prompt iteration wasn't moving the needle:

![hard/ep_mean_reward](https://raw.githubusercontent.com/aarxshi/DistTrainEnv/main/assets/hard_ep_mean_reward.jpeg)

API rate limits and malformed JSON were part of the problem — but fixing those didn't fix the underlying issue. The agent couldn't consistently reason its way to node_2 from a prompt alone.

**[Phase 1 W&B Report →](https://api.wandb.ai/links/srinidhi-rsp-pes-university/e6l9s1km)**

---

## Phase 2: SFT Fine-Tuning (Colab)

We collected correct `(observation, action)` pairs from successful Phase 1 episodes on easy and medium tasks, and fine-tuned the model via Unsloth's `SFTTrainer` on a free Colab T4.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    lora_alpha                 = 32,
    use_gradient_checkpointing = "unsloth",
)
```

LoRA rank 16, ~1% of parameters trainable. 3 epochs, ~250 steps. Loss dropped from 0.15 to 0.12 and stabilized.

### Results

**Easy and Medium:**

![before/after](https://raw.githubusercontent.com/aarxshi/DistTrainEnv/main/assets/before_after_sft.png)

**Hard task:**

![hard task fine-tuned](https://raw.githubusercontent.com/aarxshi/DistTrainEnv/main/assets/fine_tuned_hard.jpeg)

| Task | Baseline | Fine-tuned (SFT) | Delta |
|------|----------|------------------|-------|
| Easy | 0.471 | 0.579 | +0.108 |
| Medium | -0.898 | -0.003 | **+0.895** |
| Hard | -0.800 | -0.800 | +0.000 |

The medium result is the most striking — a broken agent (-0.898) becoming neutral (-0.003) after fine-tuning. The hard task is the honest result: zero transfer from easy/medium trajectories. The cascading OOM scenario needs its own training signal.

**[Phase 2 W&B Report →](https://api.wandb.ai/links/srinidhi-rsp-pes-university/grbkr5bz)**

---

## Evaluation Results

| Task | Score |
|------|-------|
| Easy | 0.9722 |
| Medium | 0.9809 |
| Hard | 0.5754 |
| **Weighted** | **0.7764** |

Weights: easy 0.20 · medium 0.30 · hard 0.50

---

## Folder Structure

```
DistTrainEnv/
├── assets/                    # Phase 1 W&B charts + fine-tuned hard task plot
├── results/                   # SFT before/after plots
├── environment/
│   ├── schema.py              # shared data contract
│   ├── node.py                # node state machine
│   ├── job.py                 # training job + loss curve dynamics
│   ├── faults.py              # fault injection engine
│   ├── ring_cluster.py        # main simulation
│   ├── reward.py              # shaped reward computation
│   ├── models.py              # OpenEnv-compliant Pydantic models
│   └── env.py                 # OpenEnv API (reset / step / state)
├── tasks/
│   └── task_configs.py
├── graders/
│   ├── grader_easy.py
│   ├── grader_medium.py
│   ├── grader_hard.py
│   └── run_graders.py
├── inference.py
├── app.py
├── Dockerfile
└── requirements.txt
```

---

## Running It

```bash
pip install -r requirements.txt

# Dry run — no API key needed
python inference.py --dry-run

# Real LLM
HF_TOKEN=hf_... python inference.py
HF_TOKEN=hf_... python inference.py --task hard

# API server
python app.py

# Docker
docker build -t disttrainenv .
docker run -p 7860:7860 -e HF_TOKEN=hf_... disttrainenv
```

---

## What's Next

- **Curriculum training** — carry the LoRA adapter from easy → medium → hard sequentially
- **GRPO** — online RL with environment reward signals as the natural next step after SFT
- **Real cluster integration** — live NCCL metrics instead of simulation

---

*Built with [LLaMA 3.3](https://ai.meta.com/blog/meta-llama-3/) · [Unsloth](https://github.com/unslothai/unsloth) · [TRL](https://github.com/huggingface/trl) · [Weights & Biases](https://wandb.ai)*
