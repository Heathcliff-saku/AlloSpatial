#!/bin/bash
# Raptor-R1 GRPO Training Script v2
# Qwen3-VL-4B with real-time tool calling via RaptorToolScheduler
#
# Changes vs grpo_qwen3vl.sh:
#   1. raptor_length reward (char-based length penalty, excludes tool returns)
#      reward_weights adjusted to [0.20, 0.55, 0.15, 0.10]
#   2. GSPO: --importance_sampling_level sequence
#      (sequence-level importance sampling, reduces high-variance token-level noise)
#   3. Off-policy diagnostic logging: --log_rollout_offpolicy_metrics true
#      (rollout_importance_sampling_mode removed: multi-turn tool tokens cause logprob
#       count mismatch → IS correction is skipped for nearly every batch anyway)
#   5. Resume from checkpoint-100 (healthy metrics, before reward collapse at step 125)
#
# GPU allocation (8 GPUs total):
#   GPU 0,1 — world2mind service (start_world2mind.sh)
#   GPU 2,3 — vLLM rollout server (start_vllm_rollout.sh)
#   GPU 4,5,6,7 — GRPO training (this script)
#
# Prerequisites:
#   1. world2mind service running on port 9100
#   2. vLLM rollout server running
#   3. GRPO dataset built: python build_grpo_dataset.py
#
# Startup order:
#   bash start_world2mind.sh   # Terminal 1
#   bash start_vllm_rollout.sh # Terminal 2
#   bash grpo_qwen3vl_v2.sh    # Terminal 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
# HF cache: datasets live at $HF_HOME/hub/datasets--*
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# If your CUDA libraries are not already on the loader path, prepend them, e.g.:
# export LD_LIBRARY_PATH=/path/to/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# world2mind service URL for the scheduler (single-port fallback)
export WORLD2MIND_SERVICE_URL="http://localhost:9100"

# Multi-port W2M client (training-only, requires start_world2mind_mp.sh):
#   - Spawns 6 independent single-GPU W2M processes on ports 9100..9105.
#   - Removes cross-GPU GIL/lock contention for true 6-way parallelism.
# Set to 0 to fall back to single-port (start_world2mind.sh).
export WORLD2MIND_MULTIPORT=1
export WORLD2MIND_BASE_PORT=9100
export WORLD2MIND_GPU_IDS=0,1,2,3,4,5,6,7
export WORLD2MIND_INSTANCES_PER_GPU=1   # must match --instances_per_gpu in start_world2mind_mp.sh

# Curriculum scheduling: set to 1 to enable dynamic reward weight adjustment
export RAPTOR_CURRICULUM=0

# Adaptive reward weighting (problem 2): signal-driven rescaling of
# raptor_length / raptor_tool_use only; structure & accuracy stay frozen at
# base weights. Callback is auto-installed by adaptive_reward_callback.py
# when RAPTOR_ADAPTIVE_WEIGHTS=1.
export RAPTOR_ADAPTIVE_WEIGHTS=0
export RAPTOR_ADAPTIVE_WARMUP=50
export RAPTOR_ADAPTIVE_EVERY=10
export RAPTOR_ADAPTIVE_EMA=0.2
export RAPTOR_ADAPTIVE_WINDOW=50
export RAPTOR_ADAPTIVE_SIGMA_LENGTH=0.15
export RAPTOR_ADAPTIVE_SIGMA_TOOL=0.10

# To resume an existing W&B run, set these to your own run id:
# export WANDB_RESUME=allow
# export WANDB_RUN_ID=<your-run-id>

# Inline evaluation (problem 3) is now handled natively by swift:
#   1. --val_dataset points at grpo_val_tiny.jsonl (MindCube tiny + VSI-Bench tiny)
#   2. swift eval_rollouts every --eval_steps 100 via multi_turn_scheduler + vLLM(9200)
#   3. log_completions appends to {output_dir}/completions.jsonl
#   4. eval_metrics_callback.py (rank-0, on_evaluate) parses those rows and
#      writes eval/{task}/{metric} to wandb — no subprocess, no extra env.

# Workspace cleanup: set to 1 to auto-delete workspace dirs after each rollout batch
export RAPTOR_CLEANUP_WORKSPACE=1

# Hard cap on world2mind calls per rollout.
export RAPTOR_MAX_W2M=2

# Soft cap on view_image calls per rollout.
export RAPTOR_MAX_VIEW_IMAGE=4

# Resume the same wandb run as the previous training session so the new
# curves are appended to the original run instead of starting a fresh one.
# Run ID is taken from wandb/run-20260417_130610-85o1kiai/.
# export WANDB_RESUME=allow
# export WANDB_RUN_ID=85o1kiai

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=2,3,4,5 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=512 \
FPS_MAX_FRAMES=32 \
swift rlhf \
    --rlhf_type grpo \
    --model "${SFT_CHECKPOINT:-/path/to/AlloSpatial-sft-4B/checkpoint}" \
    --external_plugins ${SCRIPT_DIR}/raptor_rewards.py ${SCRIPT_DIR}/raptor_scheduler.py ${SCRIPT_DIR}/adaptive_reward_callback.py ${SCRIPT_DIR}/eval_metrics_callback.py \
    --reward_funcs raptor_structure raptor_accuracy raptor_tool_use raptor_length raptor_val_collector \
    --reward_weights 0.15 0.60 0.10 0.15 0.0 \
    --multi_turn_scheduler raptor_tool_scheduler \
    --max_turns 5 \
    --completion_length_limit_scope total \
    --dataset ${SCRIPT_DIR}/grpo_dataset.jsonl \
    --val_dataset ${SCRIPT_DIR}/grpo_val_tiny.jsonl \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.005 \
    --lr_scheduler_type cosine \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --output_dir "${OUTPUT_DIR:-./output/AlloSpatial-grpo-4B}" \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 32768 \
    --max_completion_length 8192 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --tuner_type full \
    --deepspeed zero2 \
    --report_to wandb \
    --save_only_model false \
    --num_generations 8 \
    --num_generations_eval 1 \
    --temperature 1.0 \
    --beta 0.0 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host localhost \
    --vllm_server_port 9200 \
    --vllm_server_pass_dataset true \
    --truncation_strategy delete \
    --log_completions true \
    --loss_type dapo \
    --dynamic_sample true \
    --epsilon_high 0.28 \
    --max_resample_times 3 \
    --overlong_filter true \
    --importance_sampling_level sequence
    # To resume from a previous GRPO checkpoint, uncomment and set the path:
    # --resume_from_checkpoint /path/to/AlloSpatial-grpo-4B/checkpoint

    # --log_rollout_offpolicy_metrics true \

        # --eval_on_start true \
