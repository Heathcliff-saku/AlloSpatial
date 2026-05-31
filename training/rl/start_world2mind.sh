#!/bin/bash
# Start world2mind tool service (DA3 depth + SAM3 segmentation)
# Runs on GPU 0,1 — must be started BEFORE GRPO training
#
# The service provides the cognitive map pipeline via HTTP API.
# GRPO's RaptorToolScheduler calls this service during rollout.

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_ENDPOINT=https://hf-mirror.com   # optional mainland-China mirror

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# World2Mind tool package (this repo's world2mind/ dir).
W2M_ROOT="${WORLD2MIND_ROOT:-$(cd "$SCRIPT_DIR/../../world2mind" && pwd)}"
# Python interpreter (needs torchcodec for in-process frame decoding).
PYTHON="${PYTHON:-python}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
$PYTHON ${W2M_ROOT}/start_service.py \
    --gpu_ids 0,1,2,3,4,5 \
    --port 9100 \
    --workers_per_gpu 2 \
    --max_cpu_concurrent 12
