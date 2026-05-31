#!/bin/bash
# Launch the World2Mind model service (DA3 depth + SAM3 segmentation) over HTTP.
#
# Optional: set a proxy / HF endpoint mirror if your environment needs them.
#   export http_proxy="..."; export https_proxy="..."
#   export HF_ENDPOINT="https://hf-mirror.com"   # mainland-China mirror
#
# HF_HOME controls where HuggingFace weights are cached.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python start_service.py --gpu_ids 0,1 --port 9100 --workers_per_gpu 1 --max_cpu_concurrent 2
