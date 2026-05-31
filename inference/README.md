# Inference: World2Mind + models

Two ways to run AlloSpatial inference:

1. **Single-video demos** (`demo_openai.py`, `demo_vllm.py`) — interactive / one-off queries.
2. **Benchmark evaluation** (`run_*.sh`) — drive the [lmms-eval](../lmms-eval) harness on
   VSI-Bench / MindCube and write full transcripts to the workspace.

All paths/keys are read from the environment. Before anything, start the World2Mind service
(`cd ../world2mind && python start_service.py ...`) and export the tool path:

```bash
export WORLD2MIND_ROOT="$(cd ../world2mind && pwd)"
export OPENAI_API_KEY=...            # for commercial APIs
export OPENAI_BASE_URL=...           # optional: OpenAI-compatible endpoint
```

## Single-video demos

```bash
# World2Mind + a commercial model (OpenAI / Claude / Gemini-compatible endpoint)
python demo_openai.py --video /path/to/video.mp4 \
    --query "How many chairs are in the room?" --model gpt-5.2

# Run the tools directly, without any LLM (inspect the cognitive map)
python demo_openai.py --video /path/to/video.mp4 --direct --categories chair table

# World2Mind + an open-source / trained model served behind an OpenAI-compatible API
python demo_vllm.py --video /path/to/video.mp4 \
    --query "Describe the spatial layout" --server-url http://localhost:8003/v1
```

## Serving a local / trained model

`start_server.sh` brings up an OpenAI-compatible server (vLLM or ms-swift `swift deploy`) for a
local model — e.g. a trained AlloSpatial checkpoint:

```bash
MODEL=/path/to/AlloSpatial-checkpoint bash start_server.sh --port 8003 --backend swift-vllm
```

## Benchmark evaluation

Each script wraps `python -m lmms_eval` with a specific model adapter (defined in
`../lmms-eval/lmms_eval/models/simple/`). `RAPTOR_ROOT`/`LMMS_EVAL_ROOT` default to this repo's
`world2mind/` and `lmms-eval/` (override with `WORLD2MIND_ROOT` / `LMMS_EVAL_ROOT`).

| Script | lmms-eval model | What it evaluates |
|---|---|---|
| `run_benchmark.sh` | `world2mind` | **Commercial model + World2Mind** (the main training-free setting) |
| `run_baseline.sh` | `api_baseline` | Commercial model, **no tool** (baseline) |
| `run_local_benchmark.sh` | `world2mind_local` | **Local / trained model + World2Mind** |
| `run_local_baseline_qwens.sh` | `api_baseline` | Local model served as an API, **no tool** (baseline) |
| `run_blind_benchmark.sh` | `blind_benchmark` | World2Mind + model in the **"blind" setting** (AST text only, no raw frames) |
| `run_blind_baseline.sh` | `blind_baseline` | Blind baseline (no tool, no frames) |

Examples:

```bash
# Commercial model + World2Mind on VSI-Bench (tiny split)
bash run_benchmark.sh --tasks vsibench_tiny --model gpt-5.2 --gpu_ids 0,1,2,3 --num_concurrent 20

# Trained AlloSpatial model + World2Mind on MindCube
bash run_local_benchmark.sh --tasks mindcube_tiny

# "Blind" ablation: AST-only reasoning, visual inputs removed
bash run_blind_benchmark.sh --tasks vsibench_tiny --model gpt-5.2
```

Common flags: `--tasks`, `--model`, `--api_type {openai,claude}`, `--max_frames`, `--fps`,
`--num_concurrent`, `--max_new_tokens`, `--temperature`, `--mca_post_prompt`, `--na_post_prompt`,
`--resume_log`. Pass `--help`-style flags as documented inside each script header.

## Answer extraction convention

Models are prompted to emit the final answer wrapped in `<Answer> ... </Answer>` (a letter for
multiple-choice, a single number for numeric questions). The lmms-eval task utilities parse this tag;
full transcripts (reasoning + tool calls + tool results) are saved under the run's workspace.
