# export HF_HOME="/data/shouwei"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)
# export HF_ENDPOINT=https://hf-mirror.com   # optional mainland-China mirror
export HF_TOKEN="${HF_TOKEN:-}"              # set if you need gated HuggingFace datasets
# Example with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000

python3 -m lmms_eval \
    --model vllm \
    --model_args model=/path/to/Qwen3-VL-4B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=12800,disable_mm_preprocessor_cache=True \
    --tasks vsibench_tiny \
    --batch_size 16 \
    --output_path ./logs \
    --log_samples \
    --verbosity=DEBUG

