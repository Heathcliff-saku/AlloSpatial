#!/usr/bin/env bash
#
# Blind LLM Baseline Benchmark Evaluation Entry Script
#
# Runs lmms-eval with a pure text-only LLM (no images, no tool calling).
# Used as a blind-guessing baseline to measure how much visual information
# and spatial intelligence tools contribute to performance.
#
# Usage:
#   bash demo/run_blind_baseline.sh --tasks vsibench
#   bash demo/run_blind_baseline.sh --tasks mindcube_full --model claude-sonnet-4-20250514 --api_type claude
#   bash demo/run_blind_baseline.sh --tasks vsibench --limit 10 --num_concurrent 8
#

set -euo pipefail

# ============================================================
# Activate your lmms-eval environment here if needed, e.g.:
#   source /path/to/venv/bin/activate
# ============================================================
export HF_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$(cd "$SCRIPT_DIR/../lmms-eval" && pwd)}"

# ============================================================
# Defaults
# ============================================================
TASKS="vsibench"
MODEL="gpt-5.2"
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
API_KEY="${OPENAI_API_KEY:-}"
API_TYPE="auto"
NUM_CONCURRENT=40
WORKSPACE="./workspace"
LIMIT=""
MAX_NEW_TOKENS=16384
TEMPERATURE=0.0
TIMEOUT=1200
MCA_POST_PROMPT=""
NA_POST_PROMPT=""
PRE_PROMPT=""
DOC_ID=""

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
        --limit)           LIMIT="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature)     TEMPERATURE="$2"; shift 2 ;;
        --timeout)         TIMEOUT="$2"; shift 2 ;;
        --lmms_eval_root)  LMMS_EVAL_ROOT="$2"; shift 2 ;;
        --mca_post_prompt) MCA_POST_PROMPT="$2"; shift 2 ;;
        --na_post_prompt)  NA_POST_PROMPT="$2"; shift 2 ;;
        --pre_prompt)      PRE_PROMPT="$2"; shift 2 ;;
        --doc_id)          DOC_ID="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  Blind LLM Baseline Benchmark Evaluation"
echo "  (No images, no tools — pure text blind guessing)"
echo "============================================================"
echo "  Tasks:        ${TASKS}"
echo "  LLM Model:    ${MODEL}"
echo "  API Type:     ${API_TYPE}"
echo "  Base URL:     ${BASE_URL}"
echo "  Concurrent:   ${NUM_CONCURRENT}"
echo "  Workspace:    ${WORKSPACE}"
echo "  Max Tokens:   ${MAX_NEW_TOKENS}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Timeout:      ${TIMEOUT}s"
echo "  Limit:        ${LIMIT:-all}"
echo "  Doc ID:       ${DOC_ID:-none}"
echo "  Pre Prompt:   ${PRE_PROMPT:-default}"
echo "  MCA Prompt:   ${MCA_POST_PROMPT:-default}"
echo "  NA Prompt:    ${NA_POST_PROMPT:-default}"
echo "============================================================"

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
# Run lmms-eval with blind_baseline model
# ============================================================
MODEL_ARGS="model_version=${MODEL}"
MODEL_ARGS="${MODEL_ARGS},base_url=${BASE_URL}"
MODEL_ARGS="${MODEL_ARGS},api_key=${API_KEY}"
MODEL_ARGS="${MODEL_ARGS},api_type=${API_TYPE}"
MODEL_ARGS="${MODEL_ARGS},num_concurrent=${NUM_CONCURRENT}"
MODEL_ARGS="${MODEL_ARGS},workspace_dir=${WORKSPACE}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${MAX_NEW_TOKENS}"
MODEL_ARGS="${MODEL_ARGS},temperature=${TEMPERATURE}"
MODEL_ARGS="${MODEL_ARGS},timeout=${TIMEOUT}"

if [[ -n "$DOC_ID" ]]; then
    MODEL_ARGS="${MODEL_ARGS},doc_id=${DOC_ID}"
fi

EVAL_CMD="python -m lmms_eval \
    --model blind_baseline \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --batch_size 1 \
    --output_path ${WORKSPACE}/blind_baseline_results \
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
echo "  Blind baseline evaluation complete!"
echo "  Results:  ${WORKSPACE}/blind_baseline_results"
echo "  Full logs: ${WORKSPACE}/${TASKS}/blind_baseline_eval_log.jsonl"
echo "============================================================"
