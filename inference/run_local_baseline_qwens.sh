#!/usr/bin/env bash
#
# Local Model Baseline Evaluation (Raw Weights, No System Prompt, No Tools)
#
# Evaluates the local Qwen3-VL model's raw performance via vLLM server.
# No system prompt, no tool calling, no DA3+SAM3 service — pure model inference.
#
# Prerequisites:
#   Start Qwen3-VL model server (vLLM):
#     bash demo/start_server.sh --gpu_ids 6 --port 8003
#
# Usage:
#   bash demo/run_local_baseline.sh --tasks vsibench_tiny
#   bash demo/run_local_baseline.sh --tasks vsibench --limit 10
#   bash demo/run_local_baseline.sh --tasks mindcube_tiny
#   bash demo/run_local_baseline.sh --tasks vsibench --model_server_url http://localhost:8003/v1
#

set -euo pipefail

# ============================================================
# Environment
# ============================================================
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ============================================================
# Defaults
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# RAPTOR_ROOT = the World2Mind tool package (this repo's world2mind/ dir).
RAPTOR_ROOT="${WORLD2MIND_ROOT:-$REPO_ROOT/world2mind}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$REPO_ROOT/lmms-eval}"

# Python interpreter (needs lmms_eval + openai installed). Override with
# --python /path/to/python or by exporting BASELINE_PYTHON.
PYTHON_BIN="${BASELINE_PYTHON:-python}"

# Model server settings (vLLM / swift deploy)
MODEL_SERVER_URL="http://localhost:8003/v1"
MODEL_NAME="qwen3-vl-raptor"

# Evaluation settings
TASKS="mindcube_tiny"
WORKSPACE="./workspace"
MAX_FRAMES=7
FPS=2.0
LIMIT=""
TEMPERATURE=1.0
TIMEOUT=600
MAX_NEW_TOKENS=8192
NUM_CONCURRENT=16
MCA_POST_PROMPT="The result should given in the form of <Answer> [ONLY The option's letter from the given choices] </Answer>."
NA_POST_PROMPT="The result should given in the form of <Answer> [A single number] </Answer>."
PRE_PROMPT=""
DOC_ID=""

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_server_url)  MODEL_SERVER_URL="$2"; shift 2 ;;
        --model_name)    MODEL_NAME="$2"; shift 2 ;;
        --tasks)         TASKS="$2"; shift 2 ;;
        --workspace)     WORKSPACE="$2"; shift 2 ;;
        --max_frames)    MAX_FRAMES="$2"; shift 2 ;;
        --temperature)   TEMPERATURE="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --num_concurrent) NUM_CONCURRENT="$2"; shift 2 ;;
        --fps)           FPS="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --lmms_eval_root) LMMS_EVAL_ROOT="$2"; shift 2 ;;
        --mca_post_prompt) MCA_POST_PROMPT="$2"; shift 2 ;;
        --na_post_prompt)  NA_POST_PROMPT="$2"; shift 2 ;;
        --pre_prompt)      PRE_PROMPT="$2"; shift 2 ;;
        --doc_id)          DOC_ID="$2"; shift 2 ;;
        --python)          PYTHON_BIN="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Sanity: lmms_eval must be importable from the chosen interpreter.
if ! "${PYTHON_BIN}" -c "import lmms_eval" >/dev/null 2>&1; then
    echo "ERROR: '${PYTHON_BIN}' cannot import lmms_eval."
    echo "  Pass --python /path/to/python (an env with lmms_eval installed),"
    echo "  or export BASELINE_PYTHON=/path/to/python."
    exit 1
fi

echo "============================================================"
echo "  Local Model Baseline Evaluation (Raw Weights)"
echo "  (No system prompt, no tools, no DA3+SAM3)"
echo "============================================================"
echo "  Model Server: ${MODEL_SERVER_URL}"
echo "  Model Name:   ${MODEL_NAME}"
echo "  Concurrent:   ${NUM_CONCURRENT}"
echo "  Tasks:        ${TASKS}"
echo "  Workspace:    ${WORKSPACE}"
echo "  Max Frames:   ${MAX_FRAMES}"
echo "  Max Tokens:   ${MAX_NEW_TOKENS}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Timeout:      ${TIMEOUT}s"
echo "  FPS:          ${FPS}"
echo "  Limit:        ${LIMIT:-all}"
echo "  Doc ID:       ${DOC_ID:-none}"
echo "============================================================"

# ============================================================
# Export prompt overrides
# ============================================================
[[ -n "$PRE_PROMPT" ]]      && export VSIBENCH_PRE_PROMPT="$PRE_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export VSIBENCH_MCA_POST_PROMPT="$MCA_POST_PROMPT"
[[ -n "$NA_POST_PROMPT" ]]  && export VSIBENCH_NA_POST_PROMPT="$NA_POST_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export MINDCUBE_MCA_POST_PROMPT="$MCA_POST_PROMPT"

# ============================================================
# Check model server
# ============================================================
echo ""
echo "[Step 1] Checking model server..."

if curl -s "${MODEL_SERVER_URL%/v1}/v1/models" > /dev/null 2>&1; then
    echo "  Model server: OK at ${MODEL_SERVER_URL}"
else
    echo "  WARNING: Model server at ${MODEL_SERVER_URL} is not responding."
    echo "  Start it with: bash ${REPO_ROOT}/inference/start_server.sh --port ${MODEL_SERVER_URL##*:}"
    exit 1
fi

# ============================================================
# Run lmms-eval with api_baseline model (no system prompt)
# ============================================================
echo ""
echo "[Step 2] Running lmms-eval with api_baseline (no system prompt, no tools)..."

MODEL_ARGS="model_version=${MODEL_NAME}"
MODEL_ARGS="${MODEL_ARGS},base_url=${MODEL_SERVER_URL}"
MODEL_ARGS="${MODEL_ARGS},api_key=EMPTY"
MODEL_ARGS="${MODEL_ARGS},api_type=openai"
MODEL_ARGS="${MODEL_ARGS},num_concurrent=${NUM_CONCURRENT}"
MODEL_ARGS="${MODEL_ARGS},max_frames_num=${MAX_FRAMES}"
MODEL_ARGS="${MODEL_ARGS},fps=${FPS}"
MODEL_ARGS="${MODEL_ARGS},workspace_dir=${WORKSPACE}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${MAX_NEW_TOKENS}"
MODEL_ARGS="${MODEL_ARGS},temperature=${TEMPERATURE}"
MODEL_ARGS="${MODEL_ARGS},timeout=${TIMEOUT}"
MODEL_ARGS="${MODEL_ARGS},system_prompt="

if [[ -n "$DOC_ID" ]]; then
    MODEL_ARGS="${MODEL_ARGS},doc_id=${DOC_ID}"
fi

EVAL_CMD="${PYTHON_BIN} -m lmms_eval \
    --model api_baseline \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --batch_size 16 \
    --output_path ${WORKSPACE}/local_baseline_results \
    --log_samples"

if [[ -n "$LIMIT" ]]; then
    EVAL_CMD="${EVAL_CMD} --limit ${LIMIT}"
fi

echo "Command: ${EVAL_CMD}"
echo ""

eval $EVAL_CMD

echo ""
echo "============================================================"
echo "  Local baseline evaluation complete!"
echo "  Results:  ${WORKSPACE}/local_baseline_results"
echo "  Logs dir: ${WORKSPACE}/${TASKS}/"
echo "============================================================"
