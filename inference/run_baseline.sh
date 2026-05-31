#!/usr/bin/env bash
#
# API Baseline Benchmark Evaluation Entry Script
#
# Runs lmms-eval with a pure LLM API (no tool calling, no DA3+SAM3).
# Used to establish baseline performance for comparison with World2Mind.
#
# Usage:
#   bash demo/run_baseline.sh --tasks vsibench
#   bash demo/run_baseline.sh --tasks mindcube_full --model claude-sonnet-4-20250514 --api_type claude
#   bash demo/run_baseline.sh --tasks vsibench --limit 10 --num_concurrent 8
#

set -euo pipefail

# ============================================================
# Python interpreter
# ============================================================
# Uses `python` from the active environment (needs lmms_eval + openai installed).
# Override with --python /path/to/python or by exporting BASELINE_PYTHON.
# Optionally activate a venv by exporting BASELINE_VENV_ACTIVATE=/path/to/venv/bin/activate.
PYTHON_BIN="${BASELINE_PYTHON:-python}"
VENV_ACTIVATE="${BASELINE_VENV_ACTIVATE:-}"

# Optional: set HF_TOKEN (gated datasets) and HF_ENDPOINT (e.g. https://hf-mirror.com) in your env.
export HF_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$(cd "$SCRIPT_DIR/../lmms-eval" && pwd)}"

# ============================================================
# Defaults
# ============================================================
TASKS="vsibench_tiny"
MODEL=""
# Provider endpoint + key are read from the environment (override with --base_url / --api_key).
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
API_KEY="${OPENAI_API_KEY:-}"
API_TYPE="auto"
NUM_CONCURRENT=40
WORKSPACE="./workspace"
MAX_FRAMES=7
FPS=1.0
LIMIT=""
MAX_NEW_TOKENS=16384
TEMPERATURE=0.0
TIMEOUT=1200
MCA_POST_PROMPT="The result should given in the form of <Answer> [ONLY The option's letter from the given choices] </Answer>."
NA_POST_PROMPT="The result should given in the form of <Answer> [A single number] </Answer>."
PRE_PROMPT=""
DOC_ID=""

# ============================================================
# Activate the chosen venv (unless explicitly skipped). Done BEFORE arg-parse
# so that --skip_venv / --venv_activate can override the default after parse.
# ============================================================
_activate_now() {
    # Optional legacy path: only activate if VENV_ACTIVATE was set explicitly.
    if [[ -n "${VENV_ACTIVATE}" && -f "${VENV_ACTIVATE}" ]]; then
        echo "[venv] activating ${VENV_ACTIVATE}"
        # shellcheck disable=SC1090
        source "${VENV_ACTIVATE}"
    fi
}

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)           TASKS="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --base_url)        BASE_URL="$2"; shift 2 ;;
        --api_key)         API_KEY="$2"; shift 2 ;;
        --api_type)        API_TYPE="$2"; shift 2 ;;
        --num_concurrent)  NUM_CONCURRENT="$2"; shift 2 ;;
        --workspace)       WORKSPACE="$2"; shift 2 ;;
        --max_frames)      MAX_FRAMES="$2"; shift 2 ;;
        --fps)             FPS="$2"; shift 2 ;;
        --limit)           LIMIT="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature)     TEMPERATURE="$2"; shift 2 ;;
        --timeout)         TIMEOUT="$2"; shift 2 ;;
        --lmms_eval_root)  LMMS_EVAL_ROOT="$2"; shift 2 ;;
        --mca_post_prompt) MCA_POST_PROMPT="$2"; shift 2 ;;
        --na_post_prompt)  NA_POST_PROMPT="$2"; shift 2 ;;
        --pre_prompt)      PRE_PROMPT="$2"; shift 2 ;;
        --doc_id)          DOC_ID="$2"; shift 2 ;;
        --venv_activate)   VENV_ACTIVATE="$2"; shift 2 ;;
        --python)          PYTHON_BIN="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

_activate_now

echo "============================================================"
echo "  API Baseline Benchmark Evaluation"
echo "============================================================"
echo "  Tasks:        ${TASKS}"
echo "  LLM Model:    ${MODEL}"
echo "  API Type:     ${API_TYPE}"
echo "  Base URL:     ${BASE_URL}"
echo "  Concurrent:   ${NUM_CONCURRENT}"
echo "  Workspace:    ${WORKSPACE}"
echo "  Max Frames:   ${MAX_FRAMES}"
echo "  FPS:          ${FPS}"
echo "  Max Tokens:   ${MAX_NEW_TOKENS}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Timeout:      ${TIMEOUT}s"
echo "  Limit:        ${LIMIT:-all}"
echo "  Doc ID:       ${DOC_ID:-none}"
echo "  Pre Prompt:   ${PRE_PROMPT:-default}"
echo "  MCA Prompt:   ${MCA_POST_PROMPT:-default}"
echo "  NA Prompt:    ${NA_POST_PROMPT:-default}"
echo "  Python:       ${PYTHON_BIN}"
echo "============================================================"

# Sanity: lmms_eval must be importable from the chosen interpreter.
if ! "${PYTHON_BIN}" -c "import lmms_eval" >/dev/null 2>&1; then
    echo "ERROR: '${PYTHON_BIN}' cannot import lmms_eval."
    echo "  Pass --python /path/to/python (an env with lmms_eval installed),"
    echo "  or export BASELINE_PYTHON=/path/to/python."
    exit 1
fi

# ============================================================
# Export prompt overrides as environment variables
# ============================================================
[[ -n "$PRE_PROMPT" ]]      && export VSIBENCH_PRE_PROMPT="$PRE_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export VSIBENCH_MCA_POST_PROMPT="$MCA_POST_PROMPT"
[[ -n "$NA_POST_PROMPT" ]]  && export VSIBENCH_NA_POST_PROMPT="$NA_POST_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export MINDCUBE_MCA_POST_PROMPT="$MCA_POST_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export MMSI_MCA_POST_PROMPT="$MCA_POST_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export MMSI_BENCH_MCA_POST_PROMPT="$MCA_POST_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export VIEWSPATIAL_MCA_POST_PROMPT="$MCA_POST_PROMPT"

# ============================================================
# Run lmms-eval with api_baseline model
# ============================================================
MODEL_ARGS="model_version=${MODEL}"
MODEL_ARGS="${MODEL_ARGS},base_url=${BASE_URL}"
MODEL_ARGS="${MODEL_ARGS},api_key=${API_KEY}"
MODEL_ARGS="${MODEL_ARGS},api_type=${API_TYPE}"
MODEL_ARGS="${MODEL_ARGS},num_concurrent=${NUM_CONCURRENT}"
MODEL_ARGS="${MODEL_ARGS},max_frames_num=${MAX_FRAMES}"
MODEL_ARGS="${MODEL_ARGS},fps=${FPS}"
MODEL_ARGS="${MODEL_ARGS},workspace_dir=${WORKSPACE}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${MAX_NEW_TOKENS}"
MODEL_ARGS="${MODEL_ARGS},temperature=${TEMPERATURE}"
MODEL_ARGS="${MODEL_ARGS},timeout=${TIMEOUT}"

if [[ -n "$DOC_ID" ]]; then
    MODEL_ARGS="${MODEL_ARGS},doc_id=${DOC_ID}"
fi

EVAL_CMD="${PYTHON_BIN} -m lmms_eval \
    --model api_baseline \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --batch_size 1 \
    --output_path ${WORKSPACE}/baseline_results \
    --log_samples"

if [[ -n "$LIMIT" ]]; then
    EVAL_CMD="${EVAL_CMD} --limit ${LIMIT}"
fi

echo ""
echo "Command: ${EVAL_CMD}"
echo ""

eval $EVAL_CMD

echo ""
echo "============================================================"
echo "  Baseline evaluation complete!"
echo "  Results:  ${WORKSPACE}/baseline_results"
echo "  Full logs: ${WORKSPACE}/${TASKS}/baseline_eval_log.jsonl"
echo "============================================================"
