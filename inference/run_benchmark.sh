#!/usr/bin/env bash
#
# World2Mind Benchmark Evaluation Entry Script
#
# Assumes DA3+SAM3 model service is already running externally
# (started via start_service.py in its own environment).
# This script activates the lmms-eval venv and runs evaluation.
#
# Usage:
#   bash demo/run_benchmark.sh --gpu_ids 0,1 --tasks vsibench
#   bash demo/run_benchmark.sh --gpu_ids 0,1,2,3 --tasks mindcube_full --model claude-sonnet-4-20250514 --api_type claude
#   bash demo/run_benchmark.sh --gpu_ids 0 --tasks vsibench --limit 10
#   bash demo/run_benchmark.sh --tasks vsibench --resume_log workspace/vsibench/vsibench_claude-opus-4-6_20260227_143052.jsonl
#
# NOTE: Start the model service separately first:
#   conda activate raptor && python start_service.py --gpu_ids 0,1,2,3
#

set -euo pipefail

# ============================================================
# Activate your lmms-eval environment here if needed, e.g.:
#   source /path/to/venv/bin/activate
# ============================================================
# Optional: set HF_TOKEN / HF_ENDPOINT in your env if needed.
export HF_TOKEN="${HF_TOKEN:-}"

# ============================================================
# Defaults
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# RAPTOR_ROOT = the World2Mind tool package (this repo's world2mind/ dir).
RAPTOR_ROOT="${WORLD2MIND_ROOT:-$REPO_ROOT/world2mind}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$REPO_ROOT/lmms-eval}"

GPU_IDS="0,1,2,3"
TASKS="vsibench"
MODEL="doubao-seed-1-8-251228"
# Provider endpoint + key are read from the environment (override with --base_url / --api_key).
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
API_KEY="${OPENAI_API_KEY:-}"
API_TYPE="auto"
SERVICE_PORT=8100
WORKSPACE="./workspace"
MAX_TURNS=10
MAX_FRAMES=32
FPS=1.0
LIMIT=""
TEMPERATURE=0.0
TIMEOUT=600
MAX_NEW_TOKENS=16384
NUM_CONCURRENT=20
CONFIG="${RAPTOR_ROOT}/config/default_config.yaml"
MCA_POST_PROMPT="Following the instruction, use the tools appropriately for careful reasoning. After reasoning, the result should given in the form of <Answer> [The option's letter from the given choices, DO NOT provide any other information here.] </Answer>"
NA_POST_PROMPT="Following the instruction, use the tools appropriately for careful reasoning. After reasoning, the result should given in the form of <Answer> [A single number, DO NOT provide any other information here.] </Answer>"
PRE_PROMPT=""
RESUME_LOG=""
DOC_ID=""

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_ids)       GPU_IDS="$2"; shift 2 ;;
        --tasks)         TASKS="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --base_url)      BASE_URL="$2"; shift 2 ;;
        --api_key)       API_KEY="$2"; shift 2 ;;
        --api_type)      API_TYPE="$2"; shift 2 ;;
        --port)          SERVICE_PORT="$2"; shift 2 ;;
        --workspace)     WORKSPACE="$2"; shift 2 ;;
        --max_turns)     MAX_TURNS="$2"; shift 2 ;;
        --max_frames)    MAX_FRAMES="$2"; shift 2 ;;
        --temperature)     TEMPERATURE="$2"; shift 2 ;;
        --timeout)         TIMEOUT="$2"; shift 2 ;;
        --max_new_tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --num_concurrent)  NUM_CONCURRENT="$2"; shift 2 ;;
        --fps)           FPS="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --lmms_eval_root) LMMS_EVAL_ROOT="$2"; shift 2 ;;
        --mca_post_prompt) MCA_POST_PROMPT="$2"; shift 2 ;;
        --na_post_prompt)  NA_POST_PROMPT="$2"; shift 2 ;;
        --pre_prompt)      PRE_PROMPT="$2"; shift 2 ;;
        --resume_log)      RESUME_LOG="$2"; shift 2 ;;
        --doc_id)          DOC_ID="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

SERVICE_URL="http://localhost:${SERVICE_PORT}"

# Auto-detect total worker count from service health endpoint
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPU=${#GPU_ARRAY[@]}
HEALTH_JSON=$(curl -s "${SERVICE_URL}/health" 2>/dev/null || echo "")
if [[ -n "$HEALTH_JSON" ]] && command -v python3 &>/dev/null; then
    DETECTED=$(echo "$HEALTH_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('workers',[])))" 2>/dev/null || echo "")
    if [[ -n "$DETECTED" && "$DETECTED" -gt 0 ]]; then
        NUM_WORKERS=$DETECTED
    else
        NUM_WORKERS=$NUM_GPU
    fi
else
    NUM_WORKERS=$NUM_GPU
fi

# Auto-cap NUM_CONCURRENT to avoid excessive memory usage
# Allow a small buffer (+2) above worker count for request pipelining
MAX_CONCURRENT=$((NUM_WORKERS + 20))
if [[ "$NUM_CONCURRENT" -gt "$MAX_CONCURRENT" ]]; then
    echo "NOTE: Capping NUM_CONCURRENT from ${NUM_CONCURRENT} to ${MAX_CONCURRENT} (workers=${NUM_WORKERS} + 2 buffer)"
    NUM_CONCURRENT=$MAX_CONCURRENT
fi

echo "============================================================"
echo "  World2Mind Benchmark Evaluation"
echo "============================================================"
echo "  GPUs:         ${GPU_IDS} (${NUM_GPU} GPUs, ${NUM_WORKERS} workers)"
echo "  Concurrent:   ${NUM_CONCURRENT}"
echo "  Tasks:        ${TASKS}"
echo "  LLM Model:    ${MODEL}"
echo "  API Type:     ${API_TYPE}"
echo "  Base URL:     ${BASE_URL}"
echo "  Service URL:  ${SERVICE_URL}"
echo "  Workspace:    ${WORKSPACE}"
echo "  Max Turns:    ${MAX_TURNS}"
echo "  Max Frames:   ${MAX_FRAMES}"
echo "  Max Tokens:   ${MAX_NEW_TOKENS}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Timeout:      ${TIMEOUT}s"
echo "  FPS:          ${FPS}"
echo "  Limit:        ${LIMIT:-all}"
echo "  Resume Log:   ${RESUME_LOG:-none}"
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
# Check that model service is running
# ============================================================
echo ""
echo "[Step 1] Checking DA3+SAM3 model service..."
if curl -s "${SERVICE_URL}/health" > /dev/null 2>&1; then
    echo "Service is reachable at ${SERVICE_URL}"
else
    echo "WARNING: Service at ${SERVICE_URL} is not responding."
    echo "Please start it separately:"
    echo "  python ${RAPTOR_ROOT}/start_service.py --gpu_ids ${GPU_IDS} --port ${SERVICE_PORT}"
    exit 1
fi

# ============================================================
# Run lmms-eval
# ============================================================
echo ""
echo "[Step 2] Running lmms-eval..."

MODEL_ARGS="model_version=${MODEL}"
MODEL_ARGS="${MODEL_ARGS},base_url=${BASE_URL}"
MODEL_ARGS="${MODEL_ARGS},api_key=${API_KEY}"
MODEL_ARGS="${MODEL_ARGS},api_type=${API_TYPE}"
MODEL_ARGS="${MODEL_ARGS},service_url=${SERVICE_URL}"
MODEL_ARGS="${MODEL_ARGS},raptor_root=${RAPTOR_ROOT}"
MODEL_ARGS="${MODEL_ARGS},num_workers=${NUM_WORKERS}"
MODEL_ARGS="${MODEL_ARGS},num_concurrent=${NUM_CONCURRENT}"
MODEL_ARGS="${MODEL_ARGS},max_turns=${MAX_TURNS}"
MODEL_ARGS="${MODEL_ARGS},max_frames_num=${MAX_FRAMES}"
MODEL_ARGS="${MODEL_ARGS},fps=${FPS}"
MODEL_ARGS="${MODEL_ARGS},workspace_dir=${WORKSPACE}"
MODEL_ARGS="${MODEL_ARGS},config_path=${CONFIG}"
MODEL_ARGS="${MODEL_ARGS},temperature=${TEMPERATURE}"
MODEL_ARGS="${MODEL_ARGS},timeout=${TIMEOUT}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${MAX_NEW_TOKENS}"

if [[ -n "$RESUME_LOG" ]]; then
    MODEL_ARGS="${MODEL_ARGS},resume_log=${RESUME_LOG}"
fi

if [[ -n "$DOC_ID" ]]; then
    MODEL_ARGS="${MODEL_ARGS},doc_id=${DOC_ID}"
fi

EVAL_CMD="python -m lmms_eval \
    --model world2mind \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --batch_size 1 \
    --output_path ${WORKSPACE}/results \
    --log_samples"

if [[ -n "$LIMIT" ]]; then
    EVAL_CMD="${EVAL_CMD} --limit ${LIMIT}"
fi

echo "Command: ${EVAL_CMD}"
echo ""

eval $EVAL_CMD

echo ""
echo "============================================================"
echo "  Evaluation complete!"
echo "  Results:  ${WORKSPACE}/results"
echo "  Logs dir: ${WORKSPACE}/${TASKS}/"
echo "============================================================"
