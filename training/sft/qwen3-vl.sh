# Qwen3-VL-4B-Instruct SFT on Raptor spatial intelligence dataset
# 6 GPUs, full-parameter training with DeepSpeed ZeRO2
#
# Key differences from Qwen3.5-4B:
#   - Standard transformer attention (no GatedDeltaNet) → vLLM compatible
#   - Supports packing + padding_free → more efficient than group_by_length
#   - No thinking mode in Instruct variant → no add_non_thinking_prefix needed

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
# If your CUDA libraries are not already on the loader path, prepend them, e.g.:
# export LD_LIBRARY_PATH=/path/to/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=6 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=768 \
FPS_MAX_FRAMES=32 \
swift sft \
    --model "${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
    --dataset "${SFT_DATASET:-/path/to/sft_swift.jsonl}" \

    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --freeze_vit true \
    --freeze_aligner true \
    --packing false \
    --padding_free true \
    --gradient_checkpointing true \
    --output_dir "${OUTPUT_DIR:-./output/AlloSpatial-sft-8B}" \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 32768 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --tuner_type full \
    --deepspeed zero2 \
    --report_to wandb \
    --save_only_model true
