#!/usr/bin/env bash
#
# Start the trained Qwen3-VL model as an OpenAI-compatible API server.
#
# Supports two backends:
#   - vllm (default): faster inference, native Qwen3-VL support
#   - transformers:   slower but always works, via ms-swift deploy
#
# Usage:
#   bash demo/start_server.sh                                        # vllm (default)
#   bash demo/start_server.sh --backend transformers                 # transformers fallback
#   bash demo/start_server.sh --model /path/to/ckpt --gpu_ids 6 --port 8001
#   bash demo/start_server.sh --tp 2 --gpu_ids 6,7
#
export VLLM_USE_V1=0
set -euo pipefail

# ============================================================
# Defaults
# ============================================================
MODEL="${MODEL:-/path/to/AlloSpatial-checkpoint}"  # trained AlloSpatial checkpoint, or any Qwen3-VL model
GPU_IDS="6,7"
PORT=8003
BACKEND="swift-vllm"
MAX_NEW_TOKENS=32768
MAX_MODEL_LEN=32768
SERVED_MODEL_NAME="qwen3-vl-raptor"

# vLLM-specific
TP=2
GPU_MEMORY_UTILIZATION=0.65
LIMIT_MM_PER_PROMPT='{"image": 32, "video": 1}'

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --gpu_ids)        GPU_IDS="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --backend)        BACKEND="$2"; shift 2 ;;
        --tp)             TP="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --max_model_len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --served_model_name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --limit_mm_per_prompt) LIMIT_MM_PER_PROMPT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  Starting Qwen3-VL Model Server"
echo "============================================================"
echo "  Model:          ${MODEL}"
echo "  GPUs:           ${GPU_IDS}"
echo "  Port:           ${PORT}"
echo "  Backend:        ${BACKEND}"
echo "  Model Name:     ${SERVED_MODEL_NAME}"
echo "  Max Model Len:  ${MAX_MODEL_LEN}"
echo "  Max New Tokens: ${MAX_NEW_TOKENS}"
if [[ "$BACKEND" == "vllm" || "$BACKEND" == "swift-vllm" ]]; then
echo "  TP:             ${TP}"
echo "  GPU Mem Util:   ${GPU_MEMORY_UTILIZATION}"
echo "  MM Per Prompt:  ${LIMIT_MM_PER_PROMPT}"
fi
echo "============================================================"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

if [[ "$BACKEND" == "vllm" ]]; then
    # ============================================================
    # vLLM backend: native vllm serve (recommended for Qwen3-VL)
    # ============================================================
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    vllm serve "${MODEL}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --port ${PORT} \
        --dtype bfloat16 \
        --max-model-len ${MAX_MODEL_LEN} \
        --tensor-parallel-size ${TP} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
        --trust-remote-code \

elif [[ "$BACKEND" == "swift-vllm" ]]; then
    # ============================================================
    # swift deploy + vLLM: ms-swift handles template/tokenizer
    # ============================================================
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    IMAGE_MAX_TOKEN_NUM=1024 \
    VIDEO_MAX_TOKEN_NUM=768 \
    FPS_MAX_FRAMES=32 \
    swift deploy \
        --model "${MODEL}" \
        --infer_backend vllm \
        --torch_dtype bfloat16 \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --vllm_max_model_len ${MAX_MODEL_LEN} \
        --vllm_gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --vllm_tensor_parallel_size ${TP} \
        --vllm_limit_mm_per_prompt "{\"image\": 32, \"video\": 1}" \
        --port ${PORT} \
        --served_model_name "${SERVED_MODEL_NAME}" \
        --temperature 0

elif [[ "$BACKEND" == "transformers" ]]; then
    # ============================================================
    # Transformers backend: fallback, always works
    # ============================================================
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    IMAGE_MAX_TOKEN_NUM=1024 \
    VIDEO_MAX_TOKEN_NUM=768 \
    FPS_MAX_FRAMES=32 \
    swift deploy \
        --model "${MODEL}" \
        --infer_backend transformers \
        --torch_dtype bfloat16 \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_length ${MAX_MODEL_LEN} \
        --port ${PORT} \
        --served_model_name "${SERVED_MODEL_NAME}" \
        --temperature 0
else
    echo "ERROR: Unknown backend '${BACKEND}'. Use 'vllm', 'swift-vllm', or 'transformers'."
    exit 1
fi
