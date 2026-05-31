# World2Mind — Allocentric Cognitive-Mapping Sandbox

World2Mind turns egocentric video / multi-view images into structured **allocentric spatial knowledge**
(cognitive maps) that a foundation model can query as a tool. It exposes two LLM tools — `world2mind`
(build a cognitive map) and `view_image` (inspect a generated visualization).

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────────┐
│ run_pipeline.py │ ───────────►  │  Model Service        │
│ demo_*.py (api) │   REST API    │  (FastAPI)            │
│ lmms-eval models│               │  GPU i: DA3 + SAM3    │
└─────────────────┘               │  round-robin balance  │
                                  └──────────────────────┘
```

- The model service preloads **Depth Anything 3** (depth + camera pose) and **SAM 3** (open-vocabulary
  segmentation) onto one or more GPUs and serves a `/cognitive_map` endpoint that runs the full pipeline:
  `depth → segmentation → point-cloud mapping → AST → route knowledge`.
- Clients (the CLI, the demos, the lmms-eval models) call it over HTTP, so heavy models load once.

## Layout

```
config/
  default_config.yaml    # unified config — set da3.model, sam3.model_path, gpu_ids here
  category_eps.yaml       # per-category DBSCAN clustering parameters
scripts/
  config.py               # config dataclasses
  frame_extraction.py     # video → frames
  depth_estimation.py     # DA3 depth + pose
  segmentation.py         # SAM 3 open-vocab masks
  mapping.py              # semantic point-cloud construction
  ast_generation.py       # Allocentric-Spatial Tree (landmark map)
  route_knowledge.py      # traversability grid + camera trajectory (route map)
  memory_cache.py         # in-memory pipeline cache (skips intermediate disk I/O)
services/
  model_service.py        # FastAPI service (DA3 + SAM3)
  client.py               # HTTP client
tools/
  tool_definitions.py     # LLM tool schemas (world2mind, view_image)
  spatial_tools.py        # tool handlers
  prompts.py              # system / user prompts (API + vLLM variants)
  blind_prompts.py        # AST-only ("blind") prompts
  train_multiproc_{service,client}.py  # multi-GPU service fleet used during RL rollout
run_pipeline.py           # CLI entry point
start_service.py          # launch the model service
world2mind_service.sh     # convenience launcher
```

## Setup

1. Install the perception models and point the config at the weights:

   ```yaml
   # config/default_config.yaml
   da3:
     model: "/path/to/DA3NESTED-GIANT-LARGE-1.1"   # HF repo id or local path
   sam3:
     model_path: "/path/to/sam3/sam3.pt"
   service:
     gpu_ids: [0]          # one DA3+SAM3 set per GPU; e.g. [0,1,2] for 3 GPUs
     port: 8100
   ```

2. Start the service:

   ```bash
   python start_service.py --gpu_ids 0,1 --port 8100
   curl http://localhost:8100/health
   ```

## Run the pipeline directly

```bash
# video input
python run_pipeline.py --video_path /path/to/video.mp4 \
    --categories "car,building,tree" --scene_type outdoor --knowledge_type both

# image-list input
python run_pipeline.py --image_paths "/a.jpg,/b.jpg,/c.jpg" \
    --categories "chair,table,floor" --scene_type indoor
```

Outputs (AST YAML + visualizations) are written under `output_base` (default `./workspace`).

## Tools

### `world2mind`
Builds the cognitive map and returns YAML spatial knowledge + visualization file paths.

| Argument | Meaning |
|---|---|
| `video_path` / `image_paths` | input (one of) |
| `categories` | object categories to detect |
| `scene_type` | `indoor` / `outdoor` |
| `knowledge_type` | `landmark` / `route` / `both` |
| `output_format` | `grid` / `rectangle` / `ellipse` |
| `traversable_categories` | traversable classes (for route knowledge) |

### `view_image`
Views a generated visualization to aid spatial judgement:
`landmark_vis`, `route_vis`, `pointcloud_rgb_topdown`, `pointcloud_semantic_topdown`.

## Notes

- The internal codename for this tool/project is **`raptor`**; the `WORLD2MIND_ROOT` env var (or the
  `raptor_root=` model arg in lmms-eval) should point at *this* directory.
- Configuration knobs (depth thresholds, DBSCAN `eps`/`min_samples`, voxel sizes, denoising, core
  extraction, etc.) are documented inline in `config/default_config.yaml`.
