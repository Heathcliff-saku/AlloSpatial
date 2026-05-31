"""
FastAPI Model Service for World2Mind Cognitive Map Pipeline.

Loads DA3+SAM3 models onto specified GPUs and exposes a single /cognitive_map
endpoint that runs the full pipeline: depth → segmentation → mapping → AST → route.

GPU operations (depth, segmentation) are serialized per-GPU via locks.
CPU operations (mapping, AST, route) are bounded by a semaphore to prevent OOM.

Usage:
    python -m services.model_service --config config/default_config.yaml
"""

import gc
import os
import sys
import shutil
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ['HF_HOME'] = '/data2/hf_home'

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import PipelineConfig, INDOOR_TRAVERSABLE_CATEGORIES, OUTDOOR_TRAVERSABLE_CATEGORIES


# ============================================================
# Request / Response Models
# ============================================================

class CognitiveMapRequest(BaseModel):
    image_dir: str              # Path to extracted frames
    workspace: str              # Workspace directory for this request
    categories: List[str]
    scene_type: str = "outdoor"
    run_landmark: bool = True
    run_route: bool = True
    output_format: str = "grid"
    traversable_categories: List[str] = []
    config_path: str = ""       # Path to config YAML
    scene_id: str = ""


class ServiceResponse(BaseModel):
    success: bool
    message: str = ""
    data: Dict[str, Any] = {}


# ============================================================
# GPU Worker
# ============================================================

class GPUWorker:
    """
    A worker that holds DA3+SAM3 models on a single GPU.
    Uses a per-GPU lock to serialize CUDA operations.
    """

    def __init__(self, gpu_id: int, da3_model_path: str, sam3_model_path: str,
                 sam3_conf: float = 0.76, half_precision: bool = True,
                 hf_endpoint: str = ""):
        self.gpu_id = gpu_id
        self.da3_model = None
        self.sam_predictor = None
        self._lock = None  # set to shared per-GPU lock after init
        self.busy = False

        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint

        logger.info(f"[GPU {gpu_id}] Loading DA3 model: {da3_model_path}")
        from depth_anything_3.api import DepthAnything3
        self.da3_model = DepthAnything3.from_pretrained(da3_model_path).to(self.device)
        logger.info(f"[GPU {gpu_id}] DA3 loaded")

        logger.info(f"[GPU {gpu_id}] Loading SAM3 model: {sam3_model_path}")
        from ultralytics.models.sam import SAM3SemanticPredictor
        overrides = dict(
            conf=sam3_conf,
            task="segment",
            mode="predict",
            model=sam3_model_path,
            half=half_precision,
            save=False,
            verbose=False,
            device=str(gpu_id),
        )
        self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)
        logger.info(f"[GPU {gpu_id}] SAM3 loaded")

        # Warm up SAM3: initialises predictor.imgsz, predictor.mean/std so that
        # the first batched-encoding call does not fail with
        # "'NoneType' object is not subscriptable" (imgsz is None until set_image
        # is called at least once via the standard __call__ path).
        logger.info(f"[GPU {gpu_id}] Warming up SAM3 ...")
        import numpy as _np_warmup
        _dummy_rgb = _np_warmup.zeros((480, 640, 3), dtype=_np_warmup.uint8)
        self.sam_predictor.set_image(_dummy_rgb)
        self.sam_predictor(text=["object"])
        self.sam_predictor.reset_image()
        del _dummy_rgb, _np_warmup
        logger.info(f"[GPU {gpu_id}] SAM3 warm-up done")

    def run_cognitive_map(self, request: CognitiveMapRequest) -> Dict[str, Any]:
        """
        Run the full cognitive map pipeline.

        GPU operations (depth + segmentation) hold the per-GPU lock.
        CPU operations (mapping + AST + route) run without the lock.
        """
        from scripts.depth_estimation import run_depth_estimation, load_conf_summary, calculate_skip_frames
        from scripts.segmentation import run_segmentation
        from scripts.mapping import run_mapping
        from scripts.ast_generation import run_ast_generation
        from scripts.route_knowledge import generate_route_knowledge

        config = PipelineConfig.from_yaml(request.config_path)
        scene_id = request.scene_id
        workspace = Path(request.workspace)
        frames_dir = Path(request.image_dir)

        # Merge traversable categories into detection categories
        traversable_cats = list(request.traversable_categories) if request.traversable_categories else []
        all_categories = list(request.categories)
        for cat in traversable_cats:
            if cat not in all_categories:
                all_categories.append(cat)

        # Output directories
        da3_output = workspace / config.da3.output_folder
        sam_output = workspace / config.sam3.output_folder
        mapping_output = workspace / config.mapping.output_folder

        # In-memory pipeline: skip intermediate disk I/O (DA3/SAM3/mapping arrays)
        use_memory_cache = getattr(config, 'use_memory_cache', True)
        memory_cache = None
        if use_memory_cache:
            from scripts.memory_cache import SceneMemoryCache
            memory_cache = SceneMemoryCache()
            logger.info(f"[GPU {self.gpu_id}][{scene_id}] In-memory pipeline enabled")

        # ============================================================
        # GPU Phase (hold per-GPU lock)
        # ============================================================
        conf_summary = None
        skip_frames = set()

        with self._lock:
            self.busy = True
            try:
                # ---- Phase 1: DA3 Depth Estimation ----
                if config.da3.enable:
                    logger.info(f"[GPU {self.gpu_id}][{scene_id}] Phase 1: Depth Estimation")
                    # When using memory cache we always run (no disk state to resume from)
                    if memory_cache is not None or not da3_output.exists() or not (da3_output / "depth").exists():
                        conf_summary = run_depth_estimation(
                            image_dir=str(frames_dir),
                            output_dir=str(da3_output),
                            save_rgb=config.da3.save_rgb,
                            ref_view_strategy=config.da3.ref_view_strategy,
                            use_ray_pose=config.da3.use_ray_pose,
                            model_obj=self.da3_model,
                            device=self.device,
                            memory_cache=memory_cache,
                        )
                    else:
                        conf_summary = load_conf_summary(str(da3_output))

                    # Prefer conf_summary from cache when available
                    if memory_cache is not None and memory_cache.conf_summary is not None:
                        conf_summary = memory_cache.conf_summary

                    skip_frames, skip_scene = calculate_skip_frames(
                        conf_summary,
                        conf_threshold=config.da3.conf_threshold,
                        scene_conf_threshold=config.da3.scene_conf_threshold,
                    )
                    if skip_scene:
                        if memory_cache is not None:
                            memory_cache.clear()
                        return {"success": True, "error": "Scene skipped due to low confidence"}
                    if skip_frames:
                        logger.info(f"[GPU {self.gpu_id}][{scene_id}] Skipping {len(skip_frames)} low-confidence frames")

                # ---- Phase 2: SAM3 Segmentation ----
                if config.sam3.enable:
                    logger.info(f"[GPU {self.gpu_id}][{scene_id}] Phase 2: Segmentation")
                    if memory_cache is not None or not sam_output.exists() or not (sam_output / "class_info.npy").exists():
                        run_segmentation(
                            image_dir=str(frames_dir),
                            categories=all_categories,
                            output_dir=str(sam_output),
                            save_vis=config.sam3.save_vis,
                            skip_frames=skip_frames,
                            show_progress=False,
                            predictor_obj=self.sam_predictor,
                            seg_batch_size=getattr(config.sam3, "seg_batch_size", 1),
                            memory_cache=memory_cache,
                        )
            finally:
                self.busy = False

        # Free GPU-phase temporaries before CPU-heavy work
        gc.collect()
        import torch
        torch.cuda.empty_cache()

        # ============================================================
        # Extract camera orientations (few-frame scenes only)
        # ============================================================
        camera_orientations = None
        if memory_cache is not None and memory_cache.has_da3():
            valid_frame_count = sum(1 for n in memory_cache.da3_poses if n not in skip_frames)
            if valid_frame_count <= config.camera_orientation_max_frames:
                camera_orientations = self._extract_camera_orientations_from_cache(
                    memory_cache.da3_poses, skip_frames
                )
                if camera_orientations:
                    logger.info(
                        f"[GPU {self.gpu_id}][{scene_id}] Extracted camera orientations "
                        f"for {len(camera_orientations)} frames (from cache)"
                    )
        elif da3_output.exists() and (da3_output / "pose").exists():
            pose_count = len([
                f for f in sorted((da3_output / "pose").glob("*.npy"))
                if f.stem not in skip_frames
            ])
            if pose_count <= config.camera_orientation_max_frames:
                camera_orientations = self._extract_camera_orientations(
                    da3_output, skip_frames
                )
                if camera_orientations:
                    logger.info(
                        f"[GPU {self.gpu_id}][{scene_id}] Extracted camera orientations "
                        f"for {len(camera_orientations)} frames"
                    )

        # ============================================================
        # CPU Phase (bounded by semaphore to prevent OOM)
        # ============================================================
        with _cpu_semaphore:
            # ---- Phase 3: Mapping + AST ----
            if config.mapping.enable and request.run_landmark:
                logger.info(f"[GPU {self.gpu_id}][{scene_id}] Phase 3: Mapping + AST")
                mapping_output.mkdir(parents=True, exist_ok=True)

                if request.scene_type == "indoor":
                    effective_max_depth = config.max_depth_indoor
                    if memory_cache is not None and memory_cache.has_da3():
                        avg_h = self._calculate_avg_camera_height_from_cache(
                            memory_cache.da3_poses, skip_frames
                        )
                    else:
                        avg_h = self._calculate_avg_camera_height(da3_output, skip_frames)
                    effective_ceiling = avg_h + config.ceiling_height_offset
                else:
                    effective_max_depth = config.max_depth_outdoor
                    effective_ceiling = None

                success = run_mapping(
                    da3_dir=str(da3_output),
                    image_dir=str(frames_dir),
                    sam_dir=str(sam_output),
                    output_dir=str(mapping_output),
                    mode=config.mapping.mode,
                    grid_size=config.mapping.grid_size,
                    ceiling_height=effective_ceiling,
                    min_height=config.mapping.min_height,
                    min_depth=config.mapping.min_depth,
                    max_depth=effective_max_depth,
                    pixel_conf_threshold=config.mapping.pixel_conf_threshold,
                    denoise_k=config.mapping.denoise_k,
                    min_points_per_voxel=config.mapping.min_points_per_voxel,
                    point_skip=config.mapping.point_skip,
                    flip_y=config.mapping.flip_y,
                    skip_frames=skip_frames,
                    show_progress=False,
                    camera_orientations=camera_orientations,
                    use_gpu=getattr(config.mapping, 'use_gpu', True),
                    memory_cache=memory_cache,
                )
                gc.collect()
                if not success:
                    if memory_cache is not None:
                        memory_cache.clear()
                    return {"success": False, "error": "Mapping failed"}

                exclude_cats = traversable_cats if request.run_route else []
                success = run_ast_generation(
                    output_dir=str(mapping_output),
                    ast_filename=config.mapping.ast_filename,
                    merge_dist=config.mapping.merge_dist,
                    min_voxels=config.mapping.min_voxels,
                    eps=config.mapping.eps,
                    min_samples=config.mapping.min_samples,
                    save_visualization=True,
                    ast_format=request.output_format,
                    grid_divisions=config.mapping.grid_divisions,
                    exclude_categories=exclude_cats,
                    use_core_extraction=config.mapping.use_core_extraction,
                    core_percentile=config.mapping.core_percentile,
                    use_downsampling=config.mapping.use_downsampling,
                    downsample_voxel_size=config.mapping.downsample_voxel_size,
                    large_class_threshold=config.mapping.large_class_threshold,
                    camera_orientations=camera_orientations,
                    use_gpu=getattr(config.mapping, 'use_gpu', True),
                    memory_cache=memory_cache,
                )
                gc.collect()
                if not success:
                    if memory_cache is not None:
                        memory_cache.clear()
                    return {"success": False, "error": "AST generation failed"}

            # ---- Phase 4: Route Knowledge ----
            if config.route_knowledge.enable and request.run_route:
                logger.info(f"[GPU {self.gpu_id}][{scene_id}] Phase 4: Route Knowledge")
                route_output = mapping_output
                route_output.mkdir(parents=True, exist_ok=True)

                route_max_depth = config.max_depth_indoor if request.scene_type == "indoor" else config.max_depth_outdoor

                # Load scene bounds from cache or disk
                route_scene_bounds = None
                if memory_cache is not None and memory_cache.pc_scene_bounds is not None:
                    route_scene_bounds = memory_cache.pc_scene_bounds
                else:
                    pc_meta_path = mapping_output / "point_cloud_meta.npy"
                    if pc_meta_path.exists():
                        try:
                            pc_meta = np.load(pc_meta_path, allow_pickle=True).item()
                            route_scene_bounds = pc_meta.get('scene_bounds')
                        except Exception:
                            pass

                config.route_knowledge.traversable_categories = traversable_cats
                try:
                    generate_route_knowledge(
                        da3_dir=str(da3_output),
                        image_dir=str(frames_dir),
                        sam_dir=str(sam_output),
                        output_dir=str(route_output),
                        scene_name=scene_id,
                        config=config.route_knowledge,
                        skip_frames=skip_frames,
                        max_depth=route_max_depth,
                        scene_bounds=route_scene_bounds,
                        pixel_conf_threshold=config.mapping.pixel_conf_threshold,
                        camera_orientations=camera_orientations,
                        memory_cache=memory_cache,
                    )
                except Exception as e:
                    logger.warning(f"[GPU {self.gpu_id}][{scene_id}] Route knowledge failed: {e}")
                gc.collect()

        # ---- Release in-memory cache now that all stages are done ----
        if memory_cache is not None:
            memory_cache.clear()
            memory_cache = None
            gc.collect()

        # ---- Cleanup ----
        if not config.mapping.save_grid_npy and mapping_output.exists():
            for npy_name in ["voxel_grid.npy", "voxel_grid_meta.npy", "point_cloud.npy", "point_cloud_meta.npy"]:
                npy_file = mapping_output / npy_name
                if npy_file.exists():
                    npy_file.unlink()

        if not config.keep_extracted_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)
        if not config.keep_da3_output and da3_output.exists():
            shutil.rmtree(da3_output)
        if not config.keep_sam3_output and sam_output.exists():
            shutil.rmtree(sam_output)

        # ---- Collect outputs ----
        outputs = {"success": True, "scene_id": scene_id, "output_dir": str(workspace)}

        if request.run_landmark:
            if request.output_format == "all":
                outputs["landmark_yaml_path"] = str(mapping_output / "cognitive_ast_ellipse.yaml")
                outputs["landmark_visualization_path"] = str(mapping_output / "ast_visualization_ellipse.png")
            else:
                ast_file = mapping_output / f"cognitive_ast_{request.output_format}.yaml"
                vis_file = mapping_output / f"ast_visualization_{request.output_format}.png"
                if not ast_file.exists():
                    ast_file = mapping_output / "cognitive_ast.yaml"
                if not vis_file.exists():
                    vis_file = mapping_output / "ast_visualization.png"
                outputs["landmark_yaml_path"] = str(ast_file)
                outputs["landmark_visualization_path"] = str(vis_file)

        if request.run_route:
            outputs["route_yaml_path"] = str(mapping_output / "route_knowledge.yaml")
            outputs["route_visualization_path"] = str(mapping_output / "route_knowledge_visualization.png")

        # Point cloud visualizations
        pc_rgb = mapping_output / "point_cloud_rgb_topdown.png"
        pc_sem = mapping_output / "point_cloud_semantic_topdown.png"
        if pc_rgb.exists():
            outputs["pointcloud_rgb_vis_path"] = str(pc_rgb)
        if pc_sem.exists():
            outputs["pointcloud_semantic_vis_path"] = str(pc_sem)

        return outputs

    def _calculate_avg_camera_height(self, da3_dir: Path, skip_frames: Set[str] = None) -> float:
        """Calculate average camera height from pose files."""
        pose_dir = da3_dir / "pose"
        if not pose_dir.exists():
            return 0.0
        heights = []
        for pose_file in sorted(pose_dir.glob("*.npy")):
            frame_name = pose_file.stem
            if skip_frames and frame_name in skip_frames:
                continue
            try:
                c2w = np.load(pose_file)
                heights.append(float(c2w[1, 3]))
            except Exception:
                continue
        return float(np.mean(heights)) if heights else 0.0

    def _calculate_avg_camera_height_from_cache(
        self, poses_dict: dict, skip_frames: Set[str] = None
    ) -> float:
        """Calculate average camera height from in-memory pose arrays."""
        heights = []
        for name, c2w in poses_dict.items():
            if skip_frames and name in skip_frames:
                continue
            try:
                heights.append(float(c2w[1, 3]))
            except Exception:
                continue
        return float(np.mean(heights)) if heights else 0.0

    @staticmethod
    def _extract_camera_orientations_from_cache(
        poses_dict: dict,
        skip_frames: Set[str] = None,
    ) -> list:
        """Extract per-frame camera position and heading from in-memory c2w pose arrays."""
        orientations = []
        idx = 0
        for frame_name in sorted(poses_dict.keys()):
            if skip_frames and frame_name in skip_frames:
                continue
            c2w = poses_dict[frame_name]
            idx += 1
            cam_x = float(c2w[0, 3])
            cam_z = float(c2w[2, 3])
            R = c2w[:3, :3]
            forward = R[:, 2]
            fx, fz = float(forward[0]), float(forward[2])
            heading_rad = np.arctan2(fx, fz)
            heading_deg = round(float(np.degrees(heading_rad)), 1)
            orientations.append({
                "frame_name": frame_name,
                "image_index": idx,
                "position": (round(cam_x, 3), round(cam_z, 3)),
                "heading_deg": heading_deg,
            })
        return orientations

    @staticmethod
    def _extract_camera_orientations(
        da3_dir: Path,
        skip_frames: Set[str] = None,
    ) -> list:
        """Extract per-frame camera position and heading from c2w pose matrices.

        For each valid frame, computes:
        - position (x, z) in world metres
        - heading angle in degrees (0°=+Z, clockwise positive)

        Returns:
            List of dicts with keys: frame_name, image_index, position, heading_deg
        """
        pose_dir = da3_dir / "pose"
        if not pose_dir.exists():
            return []

        orientations = []
        pose_files = sorted(pose_dir.glob("*.npy"))
        idx = 0
        for pose_file in pose_files:
            frame_name = pose_file.stem
            if skip_frames and frame_name in skip_frames:
                continue
            try:
                c2w = np.load(pose_file)
            except Exception:
                continue

            idx += 1
            # Camera position in world coords
            cam_x = float(c2w[0, 3])
            cam_z = float(c2w[2, 3])

            # Camera forward direction: +Z axis of the camera in world coords
            # DA3 (DUSt3R-based) uses OpenCV convention where camera looks along +Z
            R = c2w[:3, :3]
            forward = R[:, 2]
            fx, fz = float(forward[0]), float(forward[2])

            # Heading: angle from +Z axis, clockwise positive
            heading_rad = np.arctan2(fx, fz)
            heading_deg = round(float(np.degrees(heading_rad)), 1)

            orientations.append({
                "frame_name": frame_name,
                "image_index": idx,
                "position": (round(cam_x, 3), round(cam_z, 3)),
                "heading_deg": heading_deg,
            })

        return orientations

    def release(self):
        """Release GPU resources."""
        if self.da3_model is not None:
            del self.da3_model
            self.da3_model = None
        if self.sam_predictor is not None:
            del self.sam_predictor
            self.sam_predictor = None
        self.torch.cuda.empty_cache()
        logger.info(f"[GPU {self.gpu_id}] Models released")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="World2Mind Model Service", version="2.0.0")

# Global state
workers: List[GPUWorker] = []
_round_robin_counter = 0
_counter_lock = threading.Lock()
_gpu_locks: Dict[int, threading.Lock] = {}
_cpu_semaphore: threading.Semaphore = threading.Semaphore(2)  # default, overridden by init_workers


def _get_next_worker() -> GPUWorker:
    """Round-robin worker selection."""
    global _round_robin_counter
    with _counter_lock:
        worker = workers[_round_robin_counter % len(workers)]
        _round_robin_counter += 1
    return worker


@app.post("/cognitive_map", response_model=ServiceResponse)
def cognitive_map(request: CognitiveMapRequest):
    if not workers:
        raise HTTPException(status_code=503, detail="No GPU workers available")
    worker = _get_next_worker()
    try:
        logger.info(f"[GPU {worker.gpu_id}] Cognitive map request: {request.scene_id}")
        result = worker.run_cognitive_map(request)
        if result.get("success"):
            return ServiceResponse(success=True, message="Cognitive map completed", data=result)
        else:
            return ServiceResponse(success=False, message=result.get("error", "Unknown error"), data=result)
    except Exception as e:
        logger.error(f"[GPU {worker.gpu_id}] Cognitive map failed: {e}")
        import traceback
        traceback.print_exc()
        return ServiceResponse(success=False, message=str(e))


@app.get("/health")
def health():
    status = []
    for w in workers:
        status.append({
            "gpu_id": w.gpu_id,
            "device": str(w.device),
            "busy": w.busy,
            "da3_loaded": w.da3_model is not None,
            "sam3_loaded": w.sam_predictor is not None,
        })
    return {"status": "ok", "workers": status}


def init_workers(gpu_ids: List[int], config_path: str, workers_per_gpu: int = 1,
                  max_cpu_concurrent: int = 2):
    """Initialize GPU workers from config. Called before starting the server."""
    global workers, _cpu_semaphore
    _cpu_semaphore = threading.Semaphore(max(1, max_cpu_concurrent))
    logger.info(f"CPU semaphore initialized: max_cpu_concurrent={max_cpu_concurrent}")

    config = PipelineConfig.from_yaml(config_path)
    da3_model_path = config.da3.model
    sam3_model_path = config.sam3.model_path
    sam3_conf = config.sam3.conf
    half_precision = config.sam3.half_precision
    hf_endpoint = config.hf_endpoint

    workers_per_gpu = max(1, workers_per_gpu)
    # Create per-GPU locks
    for gpu_id in gpu_ids:
        if gpu_id not in _gpu_locks:
            _gpu_locks[gpu_id] = threading.Lock()
    for gpu_id in gpu_ids:
        for replica in range(workers_per_gpu):
            suffix = f" replica {replica+1}/{workers_per_gpu}" if workers_per_gpu > 1 else ""
            logger.info(f"Initializing worker on GPU {gpu_id}{suffix}...")
            w = GPUWorker(
                gpu_id=gpu_id,
                da3_model_path=da3_model_path,
                sam3_model_path=sam3_model_path,
                sam3_conf=sam3_conf,
                half_precision=half_precision,
                hf_endpoint=hf_endpoint,
            )
            w._lock = _gpu_locks[gpu_id]
            workers.append(w)
    logger.info(f"All {len(workers)} workers initialized ({len(gpu_ids)} GPUs x {workers_per_gpu} replicas)")


def run_server(host: str = "0.0.0.0", port: int = 8100):
    """Start the uvicorn server."""
    uvicorn.run(app, host=host, port=port)

