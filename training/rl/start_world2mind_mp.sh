#!/bin/bash
# Multi-process / multi-port world2mind launcher (training use only).
#
# Spawns gpu_count * instances_per_gpu independent subprocesses of start_service.py.
# Each subprocess owns exactly one GPU via CUDA_VISIBLE_DEVICES.
#
# Port assignment: base_port + gpu_id * instances_per_gpu + inst_idx
#   instances_per_gpu=1 (default): ports 9100..9105  (same as before)
#   instances_per_gpu=2:           ports 9100..9111  (GPU0→9100,9101; GPU1→9102,9103; ...)
#
# Why multi-instance: CPU phases (mapping/AST/route ~5s) run truly in parallel;
# queue depth per port halved → lower tail latency under 32-concurrent rollout.
# GPU phases are time-sliced by the CUDA driver (Default compute mode, no MPS needed).
# VRAM requirement: ~14GB per instance; verify free VRAM before increasing N.
#
# IMPORTANT: keep WORLD2MIND_INSTANCES_PER_GPU in grpo_qwen3vl.sh /
# grpo_qwen3vl_colocate.sh / start_vllm_rollout.sh in sync with --instances_per_gpu here.
#
# Child full logs:
#   instances_per_gpu=1: logs/w2m_gpu{X}.log
#   instances_per_gpu>1: logs/w2m_gpu{X}_inst{Y}.log
# Main tmux shows: [gpu X] / [gpu X.Y] READY / START scene=... / DONE duration=...s
ulimit -n 1048576
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_ENDPOINT=https://hf-mirror.com   # optional mainland-China mirror

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# World2Mind tool package (this repo's world2mind/ dir).
PROJECT_ROOT="${WORLD2MIND_ROOT:-$(cd "$SCRIPT_DIR/../../world2mind" && pwd)}"

# Python interpreter (needs torchcodec for in-process frame decoding).
PYTHON="${PYTHON:-python}"

$PYTHON ${PROJECT_ROOT}/tools/train_multiproc_service.py \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --base_port 9100 \
    --instances_per_gpu 1 \
    --config ${PROJECT_ROOT}/config/default_config.yaml \
    --log_dir ${SCRIPT_DIR}/logs \
    --ready_timeout_s 600 \
    --max_restart_fails 20 \
    --restart_cooldown_s 5 \
    --flap_window_s 120
