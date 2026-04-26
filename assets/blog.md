# Teaching a Language Model to Save a Dying Cluster

*How we built a reinforcement learning environment and trained LLaMA-3.3-70B to manage distributed ML training as an autonomous SRE agent.*

---

## 3 AM. The cluster is on fire.

Not literally. But if you've ever run a multi-node training job overnight, you know the specific dread of waking up to a wall of red.

```
node_3: CRASHED
node_7: SLOW  — throughput 30%
gradient_staleness: 0.87
loss_diverging: True
```

A 70B parameter model has been training for six hours. Somewhere in your 8-node ring, node_2's memory has been quietly climbing — 60%, 70%, 80% — while you slept. By the time node_7 slowed to a crawl and the loss started spiking, the damage was already done. The root cause and the symptom were two different nodes. You restarted the wrong one.

We wanted to see if a language model could do better. So we built an environment to find out.

---

## The Environment

**DistTrainEnv** simulates a distributed training cluster: 8 nodes in a ring all-reduce topology. Every step, the agent observes cluster state and picks an action — restart a crashed node, remove a straggler from the ring, reduce batch size on an OOM node, checkpoint, or hold.

Three tasks of increasing difficulty. Easy: node_3 crashes, restart it. Medium: node_5 is running at 30% speed and needs to be removed, but there's a healthy high-memory node sitting nearby that looks suspicious. Hard: a cascading OOM fault where node_2 is the silent root cause and node_7 is the visible symptom that everyone targets first.

The hard task scores 0.30 for fixing node_7. It scores 1.0 for catching node_2 early. The environment rewards correct reasoning, not just visible action.

---

## The Agent: LLaMA-3.3-70B

We used **LLaMA-3.3-70B-Instruct** throughout, accessed via the HuggingFace Inference API. The project ran in two phases: first as a prompted agent, then fine-tuned with SFT using Unsloth.

---

## Phase 1: Prompting

We started with prompt engineering. Per-task system prompts encode the fault logic for each scenario. At every step the agent receives nodes sorted by memory risk, a live analysis block with root-cause hints, and the last 5 actions with their rewards. Temperature at 0.2 for consistency.

For the hard task, the system prompt explicitly lays out the causal chain:

```
FAULT CHAIN:
- node_2 memory climbs ~6% per step from step 1 (SILENT)
- At step 5: node_7 becomes slow (downstream SYMPTOM)

ROOT CAUSE = node_2. SYMPTOM = node_7.
Fixing node_7 only = 0.30 score. Fixing node_2 early = 1.0 score.
```

The agent responds with a JSON action object each step, which the environment parses and executes.

### What the Charts Actually Showed

Easy and medium tasks ran cleanly enough. The hard task was a different story.

The per-step reward on the hard task never settled. Both the base prompted agent and the v2 version with tighter prompts oscillated between 0 and 1 across every episode with no stable trend emerging.

![hard/step_reward — both prompted agents over ~1200 steps, volatile throughout](hard_step_reward.jpeg)

The root cause chart made the problem concrete. The base agent flatlined at 0 for almost the entire run — occasionally spiking toward 0.4 before dropping straight back down. The v2 agent never triggered the causal fix signal at all. Over a thousand steps, neither version reliably identified node_2 as the source of the fault.

![hard/ep_root_cause_fixed — base agent near zero throughout, v2 never registers](hard_ep_root_cause_fixed.jpeg)

Mean episode reward told the same story. The base agent sat around 0.2–0.4 with high noise. The v2 agent, despite the more carefully engineered prompt, landed slightly lower and flatter. Prompt iteration was not moving the needle.

![hard/ep_mean_reward — neither run converges, v2 no better than base](hard_ep_mean_reward.jpeg)

The jank in these curves came from real engineering problems — API rate limits stalling episodes mid-run, malformed JSON responses causing silent fallbacks to `noop` — but fixing those didn't fix the underlying issue. Even in clean runs, the agent couldn't consistently reason its way to node_2. It had the information. It just couldn't reliably act on it from a prompt alone.

That's what pushed us into Phase 2.

---

## Phase 2: Fine-Tuning with SFT + Unsloth

Rather than trying to prompt the model into consistent behavior, we fine-tuned using **SFT** (Supervised Fine-Tuning) via Unsloth on a free Colab T4 GPU.

We generated training data from the environment — correct `(observation, action)` pairs across the easy and medium tasks where the agent had demonstrated the right behavior in Phase 1 — and fine-tuned the model to internalize those decisions rather than derive them from scratch at inference time.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "unsloth/Meta-Llama-3.3-70B-Instruct-bnb-4bit",
    load_in_4bit = True,
    dtype        = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    lora_alpha                 = 32,
    use_gradient_checkpointing = "unsloth",
)
```

LoRA rank 16 keeps roughly 1% of parameters trainable. The full model stays frozen. Training ran for 3 epochs over ~250 steps with the loss descending cleanly from 0.15 to around 0.12.

### Results

The before/after comparison across 5 evaluation episodes per task tells the story.

![Before vs After SFT fine-tuning — easy and medium tasks](1777194243103_image.png)

On the easy task, fine-tuning pushed mean episode reward from 0.471 to 0.579, a +0.108 gain. Consistent and clean.

The medium task is the more striking result. The baseline prompted agent scored -0.898 — it was actively making things worse, likely by triggering the false alarm node and taking repeated penalties. After fine-tuning, the same model scored -0.003. That's a +0.895 swing, going from net harmful to essentially neutral behavior in a task it had previously failed completely.

The hard task results are still being evaluated. The root cause fix rate on the hard task remains the open problem, and it's the one we'd push further with more compute and a curriculum training approach.

---

## What We Took Away

The environment design took longer than the training code. Getting the reward function to separately credit causal fixes, penalize false alarms, and score root-cause resolution at a higher rate than symptom resolution was where most of the iteration happened. That structure is what made the fine-tuning signal meaningful.

Phase 1 failing on the hard task was clarifying. The model had the reasoning capacity — it demonstrated that in the episodes where it got node_2 right — but prompting alone couldn't make that behavior reliable. SFT gave it a way to encode the correct decision pattern rather than re-derive it from a prompt every time.

The medium task result was the most surprising outcome of the project. A model that was actively hurting cluster health under prompting alone became a neutral agent after fine-tuning. That's a bigger behavioral shift than we expected from a LoRA adapter trained on a few hundred examples.

---

## What's Next

- **Hard task fine-tuning** — generate correct trajectories for the cascading OOM scenario and add them to the training set
- **Curriculum training** — carry the LoRA adapter from easy to medium to hard sequentially
- **GRPO** — now that we have a fine-tuned baseline, online RL with environment reward signals is the natural next step
- **Real cluster integration** — replace the simulation with live NCCL metrics from an actual training run

---

## Try It

The full environment, both notebooks, and checkpoints are on GitHub:
**[github.com/aarxshi/DistTrainEnv](https://github.com/aarxshi/DistTrainEnv)**

The SFT notebook runs on a free Colab T4. The LLaMA agent requires a HuggingFace token with Llama access.

---

*Built with LLaMA, Unsloth, TRL, and Weights & Biases.*
