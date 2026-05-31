#!/usr/bin/env bash
#
# World2Mind Benchmark Evaluation with Local Qwen3-VL Model (Multi-Port)
#
# Uses vLLM / swift deploy server for LLM inference and a fleet of
# DA3+SAM3 model service ports (one per GPU, started by
# sft_scripts/grpo/start_world2mind_mp.sh) for spatial intelligence.
# w2m calls are dispatched via MultiPortModelClient (least in-flight, with
# silent failover to surviving ports) — same scheduler used by GRPO rollout.
#
# Prerequisites:
#   1. Start the multi-port DA3+SAM3 fleet (ports BASE_PORT .. BASE_PORT+N-1):
#      bash demo/sft_scripts/grpo/start_world2mind_mp.sh
#
#   2. Start Qwen3-VL model server (vLLM):
#      bash demo/start_server.sh --gpu_ids <id> --port 8003
#
# Usage:
#   bash demo/run_local_benchmark.sh --tasks vsibench_tiny
#   bash demo/run_local_benchmark.sh --tasks vsibench --limit 10
#   bash demo/run_local_benchmark.sh --tasks mindcube_tiny
#   bash demo/run_local_benchmark.sh --tasks vsibench --resume_log workspace/vsibench/xxx.jsonl
#

set -euo pipefail

# ============================================================
# Activate lmms-eval environment
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

# Model server settings (vLLM / swift deploy)
MODEL_SERVER_URL="http://localhost:8003/v1"
MODEL_NAME="qwen3-vl-raptor"

# DA3+SAM3 multi-port fleet (must match start_world2mind_mp.sh)
W2M_BASE_PORT=9100
W2M_GPU_IDS="0,1,2,3,4,5,6,7"
W2M_INSTANCES_PER_GPU=1

# Evaluation settings
TASKS="vsibench_tiny"
WORKSPACE="./workspace"
MAX_TURNS=8
MAX_FRAMES=7
FPS=2.0
LIMIT=""
TEMPERATURE=1.0
TIMEOUT=600
MAX_NEW_TOKENS=8192
NUM_CONCURRENT=32
CONFIG="${RAPTOR_ROOT}/config/default_config.yaml"
# MCA_POST_PROMPT="Following the instruction, use the tools appropriately for careful reasoning. After reasoning, the result should given in the form of <Answer> [The option's letter from the given choices, DO NOT provide any other information here.] </Answer>"
# NA_POST_PROMPT="Following the instruction, use the tools appropriately for careful reasoning. After reasoning, the result should given in the form of <Answer> [A single number, DO NOT provide any other information here.] </Answer>"
MCA_POST_PROMPT=""
NA_POST_PROMPT=""
PRE_PROMPT=""
RESUME_LOG=""
DOC_ID=""

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_server_url)  MODEL_SERVER_URL="$2"; shift 2 ;;
        --model_name)    MODEL_NAME="$2"; shift 2 ;;
        --w2m_base_port) W2M_BASE_PORT="$2"; shift 2 ;;
        --w2m_gpu_ids)   W2M_GPU_IDS="$2"; shift 2 ;;
        --w2m_instances_per_gpu) W2M_INSTANCES_PER_GPU="$2"; shift 2 ;;
        --tasks)         TASKS="$2"; shift 2 ;;
        --workspace)     WORKSPACE="$2"; shift 2 ;;
        --max_turns)     MAX_TURNS="$2"; shift 2 ;;
        --max_frames)    MAX_FRAMES="$2"; shift 2 ;;
        --temperature)   TEMPERATURE="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --num_concurrent) NUM_CONCURRENT="$2"; shift 2 ;;
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

# Compute the multi-port fleet layout — mirrors train_multiproc_service.py:
#   port = base_port + gpu_id * instances_per_gpu + inst_idx
IFS=',' read -ra GPU_ARRAY <<< "$W2M_GPU_IDS"
NUM_GPU=${#GPU_ARRAY[@]}
NUM_PORTS=$(( NUM_GPU * W2M_INSTANCES_PER_GPU ))
W2M_PORTS=()
for gid in "${GPU_ARRAY[@]}"; do
    for ((inst=0; inst<W2M_INSTANCES_PER_GPU; inst++)); do
        W2M_PORTS+=( $(( W2M_BASE_PORT + gid * W2M_INSTANCES_PER_GPU + inst )) )
    done
done
PORTS_STR="${W2M_PORTS[*]}"

# Primary service URL (fallback target if multi-port env is later disabled)
SERVICE_URL="http://localhost:${W2M_BASE_PORT}"
NUM_WORKERS=${NUM_PORTS}

echo "============================================================"
echo "  World2Mind Local Benchmark (Qwen3-VL, Multi-Port)"
echo "============================================================"
echo "  Model Server: ${MODEL_SERVER_URL}"
echo "  Model Name:   ${MODEL_NAME}"
echo "  W2M Ports:    ${PORTS_STR} (${NUM_PORTS} total, ${NUM_GPU} GPU × ${W2M_INSTANCES_PER_GPU})"
echo "  Concurrent:   ${NUM_CONCURRENT}"
echo "  Tasks:        ${TASKS}"
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
echo "============================================================"

# Export multi-port env vars consumed by world2mind_local._maybe_build_multiport_client
# (same convention as demo/sft_scripts/grpo/grpo_qwen3vl*.sh)
export WORLD2MIND_MULTIPORT=1
export WORLD2MIND_BASE_PORT="${W2M_BASE_PORT}"
export WORLD2MIND_GPU_IDS="${W2M_GPU_IDS}"
export WORLD2MIND_INSTANCES_PER_GPU="${W2M_INSTANCES_PER_GPU}"
export WORLD2MIND_SERVICE_URL="${SERVICE_URL}"

# ============================================================
# Export prompt overrides
# ============================================================
[[ -n "$PRE_PROMPT" ]]      && export VSIBENCH_PRE_PROMPT="$PRE_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export VSIBENCH_MCA_POST_PROMPT="$MCA_POST_PROMPT"
[[ -n "$NA_POST_PROMPT" ]]  && export VSIBENCH_NA_POST_PROMPT="$NA_POST_PROMPT"
[[ -n "$MCA_POST_PROMPT" ]] && export MINDCUBE_MCA_POST_PROMPT="$MCA_POST_PROMPT"

# ============================================================
# Check services
# ============================================================
echo ""
echo "[Step 1] Checking services..."

# Probe every w2m port; require >=1 reachable. MultiPortModelClient handles
# silent failover for any others that go down later.
W2M_OK=0
W2M_DOWN=()
for p in "${W2M_PORTS[@]}"; do
    if curl -s --max-time 3 "http://localhost:${p}/health" > /dev/null 2>&1; then
        W2M_OK=$((W2M_OK + 1))
        echo "  DA3+SAM3 :${p} OK"
    else
        W2M_DOWN+=( "$p" )
        echo "  DA3+SAM3 :${p} DOWN"
    fi
done
if [[ $W2M_OK -eq 0 ]]; then
    echo "  ERROR: no DA3+SAM3 ports reachable. Start the fleet with:"
    echo "    bash ${REPO_ROOT}/training/rl/start_world2mind_mp.sh"
    exit 1
fi
echo "  DA3+SAM3 fleet: ${W2M_OK}/${NUM_PORTS} ports reachable"
if [[ ${#W2M_DOWN[@]} -gt 0 ]]; then
    echo "  (down: ${W2M_DOWN[*]} — MultiPortClient will skip these)"
fi

# Check model server
if curl -s "${MODEL_SERVER_URL%/v1}/v1/models" > /dev/null 2>&1; then
    echo "  Model server: OK at ${MODEL_SERVER_URL}"
else
    echo "  WARNING: Model server at ${MODEL_SERVER_URL} is not responding."
    echo "  Start it with: bash ${REPO_ROOT}/inference/start_server.sh --port ${MODEL_SERVER_URL##*:}"
    exit 1
fi

# ============================================================
# Run lmms-eval with world2mind_local model
# ============================================================
echo ""
echo "[Step 2] Running lmms-eval with world2mind_local..."

MODEL_ARGS="model_version=${MODEL_NAME}"
MODEL_ARGS="${MODEL_ARGS},base_url=${MODEL_SERVER_URL}"
MODEL_ARGS="${MODEL_ARGS},api_key=EMPTY"
MODEL_ARGS="${MODEL_ARGS},api_type=openai"
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
    --model world2mind_local \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --batch_size 8 \
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
