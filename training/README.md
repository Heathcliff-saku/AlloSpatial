# Training AlloSpatial

AlloSpatial internalizes the Spatial Reasoning Harness into an open-weight model (Qwen3-VL) in two
stages, both built on [ms-swift](https://github.com/modelscope/ms-swift):

1. **SFT cold-start** (`sft/`) — supervised fine-tuning on distilled, harness-following trajectories so
   the model learns the tool-call syntax, the Step1–5 reasoning structure, AST/route parsing, and the
   `<Answer>` format.
2. **GRPO RL** (`rl/`) — Group Sequence/Relative Policy Optimization with **live World2Mind tool calling
   during rollout** and a **Harness-Gated Trajectory Reward** (structure + accuracy + tool-use + length).

> The code uses the internal codename **`raptor`** (reward funcs `raptor_*`, env `RAPTOR_*`,
> `raptor_tool_scheduler`). These identifiers are functional — keep them as-is.

## Prerequisites

```bash
# ms-swift (training framework) + the GRPO multi-turn logging patch
git clone https://github.com/modelscope/ms-swift.git && cd ms-swift && pip install -e .
git apply /path/to/AlloSpatial/training/patches/ms-swift-grpo_trainer-multiturn.patch   # see patches/README.md
```

Datasets used: **VSI-590K** (sampled to ~50K; arkitscenes/scannet/scannetpp) and the **MindCube**
training set (~10K). Set their roots via env vars (`VSI_DATA_PATH`, `VSI_VIDEO_BASE`,
`MINDCUBE_DATA_PATH`, `MINDCUBE_IMAGE_BASE`) or edit `rl/build_grpo_dataset.py`.

## Stage 1 — SFT cold-start

```bash
cd sft
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
SFT_DATASET=/path/to/sft_swift.jsonl \
OUTPUT_DIR=./output/AlloSpatial-sft-8B \
bash qwen3-vl.sh
```

The SFT dataset is an ms-swift *messages*-format JSONL: each line is
`{"messages": [{"role": "system"|"user"|"assistant", "content": ...}], ...}` with multimodal content
(video/images) and assistant turns that contain the Step1–5 reasoning, `<tool_call>` blocks, and the
final `<Answer>…</Answer>`. The cold-start trajectories are distilled from proprietary models and
filtered for answer-correctness + structural validity (the distillation pipeline itself is out of
scope for this release).

## Stage 2 — GRPO RL

GRPO runs three cooperating services (typical 8-GPU layout):

```
GPU 0,1   world2mind tool service (DA3 + SAM3)   → rl/start_world2mind.sh (or _mp.sh for a multi-port fleet)
GPU 2,3   vLLM rollout server (model generation) → rl/start_vllm_rollout.sh
GPU 4-7   GRPO trainer (ms-swift, ZeRO2)         → rl/grpo_qwen3vl_v3.sh  (8B: grpo_qwen3vl_v3-8B.sh)
```

```bash
cd rl
python build_grpo_dataset.py          # build the prompt set (VSI-590K + MindCube)
python build_val_dataset.py           # build the VSI-Bench tiny + MindCube tiny val set

# three terminals:
bash start_world2mind.sh              # or: bash start_world2mind_mp.sh
bash start_vllm_rollout.sh
SFT_CHECKPOINT=/path/to/AlloSpatial-sft-4B/checkpoint OUTPUT_DIR=./output/AlloSpatial-grpo-4B \
  bash grpo_qwen3vl_v3.sh
```

See **`rl/README.md`** for the reward functions, the Harness-Gated Trajectory Reward, curriculum /
adaptive-weight options, in-training evaluation, and the full env-var reference.

## Layout

```
sft/
  qwen3-vl.sh                 # SFT (swift sft), full-parameter, DeepSpeed ZeRO2
rl/
  build_grpo_dataset.py       # training prompts from VSI-590K + MindCube
  build_val_dataset.py        # VSI-Bench tiny + MindCube tiny validation set
  raptor_rewards.py           # reward funcs: structure / accuracy / tool-use / length
  raptor_scheduler.py         # multi-turn scheduler with live World2Mind tool calls
  adaptive_reward_callback.py # signal-driven adaptive reward weighting
  eval_metrics_callback.py    # parses in-training eval completions → W&B metrics
  grpo_qwen3vl_v3.sh / _v3-8B.sh   # GRPO launch scripts (4B / 8B)
  start_world2mind.sh / _mp.sh     # tool service (single / multi-port fleet)
  start_vllm_rollout.sh            # vLLM rollout server
patches/
  ms-swift-grpo_trainer-multiturn.patch
```
