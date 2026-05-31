#!/usr/bin/env python3
"""
Start the World2Mind Model Service.

Reads configuration and launches the FastAPI server with DA3+SAM3
models loaded on specified GPUs.

Usage:
    python start_service.py
    python start_service.py --config config/default_config.yaml
    python start_service.py --gpu_ids 0,1 --port 8100
"""

import sys
import argparse
import yaml
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Start World2Mind Model Service")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--host", type=str, default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs (overrides config)")
    parser.add_argument("--workers_per_gpu", type=int, default=None,
                        help="Number of DA3+SAM3 replicas per GPU (overrides config)")
    parser.add_argument("--max_cpu_concurrent", type=int, default=None,
                        help="Max concurrent CPU-phase requests (overrides config)")
    args = parser.parse_args()

    # Resolve config path relative to project root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    svc_cfg = cfg.get('service', {})
    host = args.host or svc_cfg.get('host', '0.0.0.0')
    port = args.port or svc_cfg.get('port', 8100)
    gpu_ids = (
        [int(x) for x in args.gpu_ids.split(',')]
        if args.gpu_ids
        else svc_cfg.get('gpu_ids', [0])
    )
    workers_per_gpu = args.workers_per_gpu or svc_cfg.get('workers_per_gpu', 1)
    max_cpu_concurrent = args.max_cpu_concurrent or svc_cfg.get('max_cpu_concurrent', 2)

    print("=" * 60)
    print("World2Mind Model Service")
    print("=" * 60)
    print(f"  Config:          {config_path}")
    print(f"  Host:            {host}")
    print(f"  Port:            {port}")
    print(f"  GPUs:            {gpu_ids}")
    print(f"  Workers/GPU:     {workers_per_gpu}")
    print(f"  Total workers:   {len(gpu_ids) * workers_per_gpu}")
    print(f"  Max CPU concur:  {max_cpu_concurrent}")
    print("=" * 60)

    from services.model_service import init_workers, run_server

    init_workers(
        gpu_ids=gpu_ids,
        config_path=str(config_path),
        workers_per_gpu=workers_per_gpu,
        max_cpu_concurrent=max_cpu_concurrent,
    )

    print(f"\nService ready at http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/health")
    print()

    run_server(host=host, port=port)


if __name__ == "__main__":
    main()
