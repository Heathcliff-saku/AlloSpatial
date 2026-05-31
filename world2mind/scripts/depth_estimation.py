"""
Depth estimation module using Depth Anything 3.

Refactored to support both direct model inference and HTTP service calls.
"""

import os
import gc
import glob
import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def run_depth_estimation(
    image_dir: str,
    output_dir: str,
    model_path: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    save_rgb: bool = False,
    model_obj: Optional[Any] = None,
    ref_view_strategy: str = "middle",
    use_ray_pose: bool = True,
    device: Optional[torch.device] = None,
    service_client: Optional[Any] = None,
    memory_cache: Optional[Any] = None,   # SceneMemoryCache | None
) -> Optional[Dict]:
    """
    Run DA3 depth estimation on images.

    Supports three modes:
    1. service_client provided → call HTTP model service
    2. model_obj provided → use pre-loaded model directly
    3. neither → load model from model_path

    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save outputs
        model_path: DA3 model path or HuggingFace model ID
        save_rgb: Whether to save RGB images
        model_obj: Pre-loaded model object (optional)
        ref_view_strategy: Reference view selection strategy
        use_ray_pose: Whether to use ray-based pose estimation
        device: Torch device to use
        service_client: ModelServiceClient instance (optional)

    Returns:
        Dictionary with confidence statistics, or None if failed
    """
    # Mode 1: HTTP service
    if service_client is not None:
        logger.info(f"Running depth estimation via HTTP service...")
        conf_data = service_client.run_depth_estimation(
            image_dir=image_dir,
            output_dir=output_dir,
            save_rgb=save_rgb,
            ref_view_strategy=ref_view_strategy,
            use_ray_pose=use_ray_pose,
        )
        # Reload conf_summary from saved file (service saves it)
        return load_conf_summary(output_dir) or conf_data

    # Mode 2 & 3: Direct inference
    from depth_anything_3.api import DepthAnything3

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_obj is not None:
        model = model_obj
    else:
        logger.info(f"Loading DA3 model: {model_path}")
        model = DepthAnything3.from_pretrained(model_path)
        model = model.to(device=device)

    # Get images
    if os.path.isfile(image_dir):
        images_path = [image_dir]
    else:
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.BMP']
        images_path = []
        for ext in extensions:
            images_path.extend(glob.glob(os.path.join(image_dir, ext)))

    images_path = sorted(images_path)
    if not images_path:
        logger.warning(f"No images found in {image_dir}")
        return None

    logger.info(f"Found {len(images_path)} images. Starting inference...")
    logger.info(f"  ref_view_strategy: {ref_view_strategy}, use_ray_pose: {use_ray_pose}")

    # Run inference
    prediction = model.inference(
        images_path,
        ref_view_strategy=ref_view_strategy,
        use_ray_pose=use_ray_pose
    )

    use_disk = memory_cache is None

    # Create output directories only when writing to disk
    if use_disk:
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intrinsics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pose"), exist_ok=True)
        if save_rgb:
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Save / cache results
    num_images = len(images_path)
    conf_means = []

    for i in range(num_images):
        img_name = os.path.basename(images_path[i])
        name_no_ext = os.path.splitext(img_name)[0]

        depth_map = prediction.depth[i]
        intrinsics = prediction.intrinsics[i]

        # Confidence
        conf_map = None
        if prediction.conf is not None:
            conf_map = prediction.conf[i]
            conf_mean = float(np.mean(conf_map))
            conf_means.append(conf_mean)
            logger.debug(f"  Frame {name_no_ext}: conf mean = {conf_mean:.4f}")
        else:
            conf_means.append(None)

        # Pose (w2c → c2w)
        w2c_3x4 = prediction.extrinsics[i]
        w2c = np.eye(4)
        w2c[:3, :] = w2c_3x4
        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix for frame {i}, pose might be invalid.")
            c2w = np.eye(4)

        if use_disk:
            np.save(os.path.join(output_dir, "depth", f"{name_no_ext}.npy"), depth_map)
            np.save(os.path.join(output_dir, "intrinsics", f"{name_no_ext}.npy"), intrinsics)
            np.save(os.path.join(output_dir, "pose", f"{name_no_ext}.npy"), c2w)
            if conf_map is not None:
                np.save(os.path.join(output_dir, "depth", f"{name_no_ext}_conf.npy"), conf_map)
            if save_rgb:
                if hasattr(prediction, 'processed_images'):
                    cv2.imwrite(
                        os.path.join(output_dir, "images", f"{name_no_ext}.jpg"),
                        cv2.cvtColor(prediction.processed_images[i], cv2.COLOR_RGB2BGR)
                    )
                else:
                    img = cv2.imread(images_path[i])
                    cv2.imwrite(os.path.join(output_dir, "images", f"{name_no_ext}.jpg"), img)
        else:
            memory_cache.da3_depths[name_no_ext]     = depth_map.astype(np.float32)
            memory_cache.da3_confs[name_no_ext]      = (conf_map.astype(np.float32)
                                                         if conf_map is not None else None)
            memory_cache.da3_intrinsics[name_no_ext] = intrinsics.astype(np.float32)
            memory_cache.da3_poses[name_no_ext]      = c2w

    # Calculate overall confidence statistics
    valid_confs = [c for c in conf_means if c is not None]
    conf_mean_overall = float(np.mean(valid_confs)) if valid_confs else None

    # Release DA3 prediction object to free GPU/CPU memory
    del prediction
    gc.collect()
    torch.cuda.empty_cache()

    if conf_mean_overall is not None:
        logger.info(f"Overall confidence mean: {conf_mean_overall:.4f}")

    conf_summary = {
        'conf_means': conf_means,
        'conf_mean_overall': conf_mean_overall,
        'image_names': [os.path.splitext(os.path.basename(p))[0] for p in images_path]
    }

    if use_disk:
        logger.info(f"Results saved to {output_dir}")
        np.save(os.path.join(output_dir, "conf_summary.npy"), conf_summary)
    else:
        memory_cache.conf_summary = conf_summary
        logger.info(f"DA3 results cached in memory ({len(memory_cache.da3_depths)} frames)")

    return conf_summary


def load_conf_summary(da3_output_dir: str) -> Optional[Dict]:
    """Load confidence summary from DA3 output directory."""
    conf_summary_path = Path(da3_output_dir) / "conf_summary.npy"
    if conf_summary_path.exists():
        return np.load(str(conf_summary_path), allow_pickle=True).item()
    return None


def calculate_skip_frames(
    conf_summary: Optional[Dict],
    conf_threshold: float = 0.0,
    scene_conf_threshold: float = 0.0
) -> tuple:
    """
    Calculate frames to skip based on confidence thresholds.

    Returns:
        Tuple of (skip_frames set, skip_scene_due_to_low_conf bool)
    """
    skip_frames = set()
    skip_scene = False

    if conf_summary is None:
        return skip_frames, skip_scene

    if scene_conf_threshold > 0:
        conf_mean_overall = conf_summary.get('conf_mean_overall')
        if conf_mean_overall is not None and conf_mean_overall < scene_conf_threshold:
            logger.warning(f"Scene overall conf mean ({conf_mean_overall:.4f}) < {scene_conf_threshold}")
            skip_scene = True
            return skip_frames, skip_scene

    if conf_threshold > 0:
        conf_means = conf_summary.get('conf_means', [])
        image_names = conf_summary.get('image_names', [])
        for name, conf in zip(image_names, conf_means):
            if conf is not None and conf < conf_threshold:
                skip_frames.add(name)
        if skip_frames:
            logger.info(f"Will skip {len(skip_frames)} frames with conf < {conf_threshold}")

    return skip_frames, skip_scene
