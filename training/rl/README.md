# AlloSpatial GRPO RL

GRPO training for AlloSpatial with **live World2Mind tool calling during rollout** and a
**Harness-Gated Trajectory Reward (HGTR)**. (Internal codename: `raptor`.)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  World2Mind     │    │  vLLM Rollout    │    │   GRPO Trainer      │
│  service        │◄───│  server          │◄───│   (ms-swift)        │
│  (GPU 0,1)      │    │  (GPU 2,3)       │    │   (GPU 4,5,6,7)    │
│  DA3 + SAM3     │    │  model inference │    │   policy update     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
        ▲                       ▲                        │
        └── raptor_tool_scheduler ────────────────────────┘
            (parses <tool_call>, calls World2Mind, injects results)
```

## Files

| File | Description |
|------|-------------|
| `build_grpo_dataset.py` | Build the GRPO prompt set from VSI-590K + MindCube |
| `build_val_dataset.py`  | Build VSI-Bench tiny + MindCube tiny validation set |
| `raptor_rewards.py`     | Reward funcs: `raptor_structure`, `raptor_accuracy`, `raptor_tool_use`, `raptor_length` |
| `raptor_scheduler.py`   | Multi-turn scheduler with real-time tool calling (`raptor_tool_scheduler`) |
| `adaptive_reward_callback.py` | Signal-driven adaptive reward weighting |
| `eval_metrics_callback.py`    | Parses in-training eval completions → W&B metrics |
| `grpo_qwen3vl_v3.sh`    | GRPO launch script (4B) |
| `grpo_qwen3vl_v3-8B.sh` | GRPO launch script (8B) |
| `start_vllm_rollout.sh` | vLLM rollout server |
| `start_world2mind.sh` / `start_world2mind_mp.sh` | World2Mind tool service (single port / multi-port fleet) |

## Quick start

```bash
# 1. Build datasets (set VSI_*/MINDCUBE_* env vars first — see build_grpo_dataset.py)
python build_grpo_dataset.py                                   # ~60K prompts (50K VSI + 10K MindCube)
python build_grpo_dataset.py --vsi-target 1000 --output grpo_dataset_small.jsonl   # small smoke test
python build_val_dataset.py                                    # grpo_val_tiny.jsonl

# 2. Start services (three terminals)
bash start_world2mind.sh        # or start_world2mind_mp.sh for a multi-GPU fleet
bash start_vllm_rollout.sh
SFT_CHECKPOINT=/path/to/AlloSpatial-sft-4B/checkpoint \
  OUTPUT_DIR=./output/AlloSpatial-grpo-4B \
  bash grpo_qwen3vl_v3.sh
```

## Reward functions (Harness-Gated Trajectory Reward)

Default weights `[structure, accuracy, tool_use, length] = [0.15, 0.60, 0.10, 0.15]`. HGTR scores the
**whole trajectory** and applies two gates: accuracy is **structure-gated** (credited only when the
trajectory follows the harness format), and the tool reward is **correctness-tied** (granted only when a
valid tool call contributes to a correct answer).

- **`raptor_structure`** — validates the Step1–5 reasoning protocol; the `<Answer>` tag and Step1 are
  mandatory, Step2–5 are conditional on tool use.
- **`raptor_accuracy`** — MCA: exact letter match (1/0); NA: Mean Relative Accuracy (continuous [0,1],
  aligned with VSI-Bench).
- **`raptor_tool_use`** — bonus when a tool call + correct answer co-occur on tool-beneficial question
  types; penalties for excessive `world2mind` calls (> `RAPTOR_MAX_W2M`) or malformed tool arguments.
- **`raptor_length`** — token-based length penalty on model-generated tokens (tool-returned AST / route /
  visualization metadata is masked out).

## Curriculum / adaptive weighting (optional)

```bash
export RAPTOR_CURRICULUM=1          # progress-based reward multipliers
export RAPTOR_ADAPTIVE_WEIGHTS=1    # signal-driven rescaling of length / tool-use weights only
```

## In-training evaluation

`--val_dataset grpo_val_tiny.jsonl` + `--eval_steps 100` runs the multi-turn scheduler on MindCube tiny
and VSI-Bench tiny during training; `eval_metrics_callback.py` parses the logged completions and writes
`eval/{task}/{metric}` (pass@1) to W&B.

## Key env vars (set in the launch script)

| Var | Meaning |
|---|---|
| `WORLD2MIND_SERVICE_URL` | tool service URL (single-port fallback) |
| `WORLD2MIND_MULTIPORT` / `WORLD2MIND_BASE_PORT` / `WORLD2MIND_GPU_IDS` / `WORLD2MIND_INSTANCES_PER_GPU` | multi-port tool fleet (must match `start_world2mind_mp.sh`) |
| `RAPTOR_MAX_W2M` | hard cap on `world2mind` calls per rollout (default 2) |
| `RAPTOR_MAX_VIEW_IMAGE` | soft cap on `view_image` calls per rollout |
| `RAPTOR_CLEANUP_WORKSPACE` | delete rollout workspaces after each batch |

## GPU allocation (8 GPUs)

| GPUs | Service |
|------|---------|
| 0, 1 | World2Mind (DA3 + SAM3) tool backend |
| 2, 3 | vLLM rollout (model generation) |
| 4–7 | GRPO trainer (ZeRO2) |

Adjust `CUDA_VISIBLE_DEVICES` / `--gpu_ids` in each script to match your machine.
