#!/bin/bash
# Start vLLM rollout server for GRPO generation
# Runs on GPU 2,3 — must be started BEFORE GRPO training
#
# This server handles model inference during rollout (generating completions).
# The GRPO trainer connects to it via --vllm_mode server.
ulimit -n 1048576
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# world2mind service URL for the scheduler (single-port fallback)
export WORLD2MIND_SERVICE_URL="http://localhost:9100"

# Multi-port W2M client (must match start_world2mind_mp.sh).
# The scheduler plugin is loaded in THIS server process (server-mode rollout),
# so the multi-port env MUST be set here, not only in the train script.
export WORLD2MIND_MULTIPORT=1
export WORLD2MIND_BASE_PORT=9100
export WORLD2MIND_GPU_IDS=0,1,2,3,4,5,6,7
export WORLD2MIND_INSTANCES_PER_GPU=1   # must match --instances_per_gpu in start_world2mind_mp.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=768 \
FPS_MAX_FRAMES=36 \
swift rollout \
    --model "${POLICY_MODEL:-/path/to/AlloSpatial-sft-checkpoint}" \
    --port 9200 \
    --external_plugins ${SCRIPT_DIR}/raptor_scheduler.py \
    --multi_turn_scheduler raptor_tool_scheduler \
    --max_turns 5 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.65
