"""
Semantic segmentation module using SAM3.

Refactored to support both direct model inference and HTTP service calls.

Batched encoding (seg_batch_size > 1):
    The ViT image encoder is the bottleneck (~80% of per-frame time).
    With seg_batch_size > 1 we call the encoder once per batch of B frames
    and only run the lightweight decoder N times.  This mirrors the approach
    proposed in ultralytics PR #23341 (not yet merged as of v8.4.15), but
    implemented directly here so we don't need to wait for the upstream merge.

    Implementation notes:
    - We preprocess the batch manually to bypass the `assert len(im) == 1`
      that exists in the base Predictor.preprocess() of v8.4.15.
    - `predictor.get_im_features(batch_tensor)` accepts (B, 3, H, W) – the
      ViT backbone has no batch-size restriction at all.
    - `predictor.inference_features(single_features, src_shape, text=…)` is
      already public in v8.4.15 and returns (pred_masks, pred_boxes) where
      pred_boxes is (N, 6) = [x1, y1, x2, y2, score, cls_id].
    - `_extract_single_features` slices the batch feature dict frame-by-frame.
    - On any error the loop falls back to the original single-frame path for
      the affected batch so that a single problematic frame cannot abort the
      entire scene.

    seg_batch_size == 1 → original frame-by-frame path, no code change.
"""

import os
import gc
import cv2
import glob
import logging
import colorsys
import numpy as np
from pathlib import Path
from typing import List, Optional, Set, Any, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batched-encoding helpers (used when seg_batch_size > 1)
# ---------------------------------------------------------------------------

def _preprocess_batch_for_sam3(predictor, frames_rgb: list) -> "torch.Tensor":
    """Convert a list of RGB numpy arrays into a batched (B, 3, H, W) tensor.

    Replicates SAM3SemanticPredictor.pre_transform + preprocess without the
    ``assert len(im) == 1`` restriction present in ultralytics ≤ 8.4.15.
    """
    import torch
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(predictor.imgsz, auto=False, center=False, scale_fill=True)
    transformed = [lb(image=f) for f in frames_rgb]   # list of (H, W, 3) RGB uint8

    stacked = np.stack(transformed)                    # (B, H, W, 3)
    stacked = stacked[..., ::-1]                       # RGB → BGR (predictor convention)
    stacked = stacked.transpose(0, 3, 1, 2)            # (B, 3, H, W)
    stacked = np.ascontiguousarray(stacked)

    im = torch.from_numpy(stacked).to(predictor.device)
    im = (im - predictor.mean) / predictor.std
    return im.half() if predictor.model.fp16 else im.float()


def _extract_single_features(batched_features: dict, idx: int) -> dict:
    """Slice per-frame features from a batch feature dict.

    Handles the nested structure returned by SAM3's backbone.forward_image():
      {
        'vision_features':  Tensor (B, C, H', W'),
        'vision_pos_enc':   list[Tensor (B, …)],
        'backbone_fpn':     list[Tensor (B, …)],
        'sam2_backbone_out': dict | None,
      }
    """
    import torch

    out = {}
    for key, value in batched_features.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[idx: idx + 1]
        elif isinstance(value, list):
            out[key] = [
                v[idx: idx + 1] if isinstance(v, torch.Tensor) else v
                for v in value
            ]
        elif isinstance(value, dict):
            out[key] = _extract_single_features(value, idx)
        else:
            out[key] = value
    return out


def _results_from_batched_api(
    pred_masks_t,   # Tensor (N, H, W) bool  or  None
    pred_boxes_t,   # Tensor (N, 6) [x1,y1,x2,y2,score,cls_id]  or  zeros(0,6)
    h: int, w: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Convert inference_features() output to (masks_np, cls_ids_np, bboxes_np).

    Returns:
        masks_np:   (N, H, W) bool numpy  or  None when no detections
        cls_ids_np: (N,) int64 numpy
        bboxes_np:  (N, 4) float32 numpy  [x1,y1,x2,y2]
    """
    if pred_masks_t is None or pred_masks_t.shape[0] == 0:
        return None, np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32)

    masks_np = pred_masks_t.cpu().numpy()                          # (N, H, W) bool
    cls_ids_np = pred_boxes_t[:, 5].long().cpu().numpy()           # (N,) int64
    bboxes_np = pred_boxes_t[:, :4].cpu().numpy().astype(np.float32)  # (N, 4)
    return masks_np, cls_ids_np, bboxes_np


def _results_from_legacy_api(r) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Convert single-frame predictor result to (masks_np, cls_ids_np, bboxes_np).

    Args:
        r: results[0] from predictor(text=categories)
    """
    if r.masks is None:
        return None, np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32)

    masks_np = r.masks.data.cpu().numpy() > 0.5                   # (N, H', W') bool
    cls_ids_np = r.boxes.cls.long().cpu().numpy()                  # (N,) int64
    bboxes_np = r.boxes.xyxy.cpu().numpy().astype(np.float32)      # (N, 4)
    return masks_np, cls_ids_np, bboxes_np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def generate_instance_colors(num_instances: int = 256) -> List[tuple]:
    """Generate distinct colors for instance visualization."""
    colors = []
    for i in range(num_instances):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.7 + (i % 2) * 0.2
        value = 0.85 + (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_segmentation(
    image_dir: str,
    categories: List[str],
    output_dir: str,
    model_path: str = "sam3.pt",
    conf: float = 0.65,
    device: str = "cuda",
    save_vis: bool = False,
    half_precision: bool = True,
    predictor_obj: Optional[Any] = None,
    skip_frames: Optional[Set[str]] = None,
    show_progress: bool = True,
    service_client: Optional[Any] = None,
    seg_batch_size: int = 8,
    memory_cache: Optional[Any] = None,   # SceneMemoryCache | None
) -> bool:
    """
    Process images with SAM3 semantic segmentation.

    Supports three modes:
    1. service_client provided → call HTTP model service
    2. predictor_obj provided → use pre-loaded predictor directly
    3. neither → load model from model_path

    Args:
        image_dir: Directory containing input images
        categories: List of category names to detect
        output_dir: Directory to save outputs
        model_path: Path to SAM3 model
        conf: Confidence threshold
        device: Device to use
        save_vis: Whether to save visualization images
        half_precision: Whether to use half precision
        predictor_obj: Pre-loaded predictor object (optional)
        skip_frames: Set of frame names to skip
        show_progress: Whether to show progress bar
        service_client: ModelServiceClient instance (optional)
        seg_batch_size: Frames per encoder batch.  >1 enables batched
            encoding (one ViT forward pass per batch instead of per frame),
            giving ~3-5x speedup.  Set to 1 to use the original
            frame-by-frame path (safe fallback for debugging).

    Returns:
        True if successful
    """
    # Mode 1: HTTP service
    if service_client is not None:
        logger.info("Running segmentation via HTTP service...")
        service_client.run_segmentation(
            image_dir=image_dir,
            categories=categories,
            output_dir=output_dir,
            save_vis=save_vis,
            half_precision=half_precision,
            conf=conf,
            skip_frames=skip_frames,
        )
        return True

    # Mode 2 & 3: Direct inference
    from ultralytics.models.sam import SAM3SemanticPredictor

    if skip_frames is None:
        skip_frames = set()

    if predictor_obj is not None:
        predictor = predictor_obj
    else:
        logger.info(f"Initializing SAM3 Semantic Predictor: {model_path}")
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=(half_precision and device == "cuda"),
            save=False,
            verbose=False,
        )
        try:
            predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            logger.error(f"Error initializing SAM3: {e}")
            return False

    use_disk = memory_cache is None
    if use_disk:
        os.makedirs(output_dir, exist_ok=True)

    # Gather images
    if os.path.isfile(image_dir):
        image_paths = [image_dir]
    else:
        exts = ["*.jpg", "*.png", "*.jpeg", "*.BMP"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.sort()

    if not image_paths:
        logger.warning("No images found.")
        return False

    num_classes = len(categories)
    instance_colors = generate_instance_colors(max(256, num_classes))

    first_img = cv2.imread(image_paths[0])
    h, w = first_img.shape[:2]

    class_info = {"num_classes": num_classes, "class_names": categories,
                  "width": w, "height": h}
    if use_disk:
        np.save(os.path.join(output_dir, "class_info.npy"), class_info)
    else:
        memory_cache.sam3_class_info = class_info

    # Pre-filter skipped frames
    active_paths = [
        p for p in image_paths
        if os.path.splitext(os.path.basename(p))[0] not in skip_frames
    ]
    n_total = len(active_paths)
    n_skip = len(image_paths) - n_total
    logger.info(
        f"Segmenting {n_total} images with SAM3"
        f"{f' (skipping {n_skip} low-confidence frames)' if n_skip else ''}"
        f"  [seg_batch_size={seg_batch_size}]"
    )

    mask_store = memory_cache.sam3_masks if memory_cache is not None else None

    if seg_batch_size > 1:
        frame_count = _run_batched(
            predictor=predictor,
            active_paths=active_paths,
            categories=categories,
            num_classes=num_classes,
            h=h, w=w,
            instance_colors=instance_colors,
            output_dir=output_dir,
            save_vis=save_vis,
            seg_batch_size=seg_batch_size,
            show_progress=show_progress,
            mask_store=mask_store,
        )
    else:
        frame_count = _run_sequential(
            predictor=predictor,
            active_paths=active_paths,
            categories=categories,
            num_classes=num_classes,
            h=h, w=w,
            instance_colors=instance_colors,
            output_dir=output_dir,
            save_vis=save_vis,
            show_progress=show_progress,
            mask_store=mask_store,
        )

    if memory_cache is not None:
        logger.info(f"Segmentation finished ({frame_count} frames). Masks cached in memory.")
    else:
        logger.info(f"Segmentation finished ({frame_count} frames). Results at {output_dir}")
    return True


# ---------------------------------------------------------------------------
# Batched inference path  (seg_batch_size > 1)
# ---------------------------------------------------------------------------

def _run_batched(
    predictor,
    active_paths: List[str],
    categories: List[str],
    num_classes: int,
    h: int, w: int,
    instance_colors: List[tuple],
    output_dir: str,
    save_vis: bool,
    seg_batch_size: int,
    show_progress: bool,
    mask_store: Optional[dict] = None,
) -> int:
    """One ViT encoder call per batch + per-frame decoder calls."""
    import torch

    # Warm-up model (mirrors what stream_inference does internally).
    # SAM3SemanticModel does not expose a warmup() method, so guard with hasattr.
    if not predictor.done_warmup:
        predictor.setup_model()
        if hasattr(predictor.model, "warmup"):
            predictor.model.warmup(imgsz=(1, predictor.model.ch, *predictor.imgsz))
        predictor.done_warmup = True

    # Safety: predictor.imgsz is None until set_image() / __call__() initialises
    # it via setup_source() → check_imgsz().  The batched path bypasses that
    # chain, so if imgsz is still None the first call to LetterBox(None, …)
    # raises "'NoneType' object is not subscriptable".  A tiny dummy run forces
    # the standard initialisation path without wasting significant time.
    if getattr(predictor, 'imgsz', None) is None:
        import numpy as np_init
        logger.warning(
            "predictor.imgsz is None — running dummy set_image() to initialise it"
        )
        _dummy = np_init.zeros((480, 640, 3), dtype=np_init.uint8)
        predictor.set_image(_dummy)
        predictor.reset_image()
        del _dummy, np_init

    n_total = len(active_paths)
    batch_indices = range(0, n_total, seg_batch_size)
    if show_progress:
        batch_indices = tqdm(
            batch_indices, desc="Segmenting (batched)",
            total=(n_total + seg_batch_size - 1) // seg_batch_size,
        )

    frame_count = 0
    for batch_start in batch_indices:
        batch_paths = active_paths[batch_start: batch_start + seg_batch_size]

        # Load frames
        batch_bgr, batch_rgb, batch_names = [], [], []
        for p in batch_paths:
            bgr = cv2.imread(p)
            if bgr is None:
                logger.warning(f"Cannot read image: {p}, skipping")
                continue
            batch_bgr.append(bgr)
            batch_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            batch_names.append(os.path.splitext(os.path.basename(p))[0])

        if not batch_rgb:
            continue

        # Batched encoder: one forward pass for the whole batch
        batch_features = None
        try:
            im_tensor = _preprocess_batch_for_sam3(predictor, batch_rgb)
            batch_features = predictor.get_im_features(im_tensor)
        except Exception as e:
            logger.warning(
                f"Batched encoding failed at batch_start={batch_start}: {e} "
                f"— falling back to frame-by-frame for this batch"
            )

        # Per-frame decoder
        for j, (bgr, rgb, name) in enumerate(zip(batch_bgr, batch_rgb, batch_names)):
            try:
                if batch_features is not None:
                    single_feats = _extract_single_features(batch_features, j)
                    pred_masks_t, pred_boxes_t = predictor.inference_features(
                        single_feats, src_shape=rgb.shape[:2], text=categories
                    )
                    masks_np, cls_ids_np, bboxes_np = _results_from_batched_api(
                        pred_masks_t, pred_boxes_t, h, w
                    )
                else:
                    # Fallback: single-frame encode+decode
                    predictor.set_image(rgb)
                    results = predictor(text=categories)
                    masks_np, cls_ids_np, bboxes_np = _results_from_legacy_api(results[0])

                _save_frame_masks(
                    name=name, frame_bgr=bgr,
                    masks_np=masks_np, cls_ids_np=cls_ids_np, bboxes_np=bboxes_np,
                    num_classes=num_classes, h=h, w=w,
                    categories=categories, instance_colors=instance_colors,
                    output_dir=output_dir, save_vis=save_vis,
                    mask_store=mask_store,
                )
            except Exception as e:
                logger.warning(f"Failed to process frame {name}: {e}")

            frame_count += 1
            if frame_count % 20 == 0:
                gc.collect()

        # Free batch tensors
        if batch_features is not None:
            del im_tensor, batch_features
        del batch_bgr, batch_rgb
        gc.collect()

    return frame_count


# ---------------------------------------------------------------------------
# Original sequential inference path  (seg_batch_size == 1)
# ---------------------------------------------------------------------------

def _run_sequential(
    predictor,
    active_paths: List[str],
    categories: List[str],
    num_classes: int,
    h: int, w: int,
    instance_colors: List[tuple],
    output_dir: str,
    save_vis: bool,
    show_progress: bool,
    mask_store: Optional[dict] = None,
) -> int:
    """Original frame-by-frame path — unchanged from v1."""
    iterator = tqdm(active_paths, desc="Segmenting") if show_progress else active_paths
    frame_count = 0
    for img_path in iterator:
        name = os.path.splitext(os.path.basename(img_path))[0]

        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Cannot read image: {img_path}, skipping")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predictor.set_image(frame_rgb)
        results = predictor(text=categories)
        masks_np, cls_ids_np, bboxes_np = _results_from_legacy_api(results[0])

        _save_frame_masks(
            name=name, frame_bgr=frame,
            masks_np=masks_np, cls_ids_np=cls_ids_np, bboxes_np=bboxes_np,
            num_classes=num_classes, h=h, w=w,
            categories=categories, instance_colors=instance_colors,
            output_dir=output_dir, save_vis=save_vis,
            mask_store=mask_store,
        )

        del frame, frame_rgb, results
        frame_count += 1
        if frame_count % 10 == 0:
            gc.collect()

    return frame_count


# ---------------------------------------------------------------------------
# Shared per-frame save helper
# ---------------------------------------------------------------------------

def _save_frame_masks(
    name: str,
    frame_bgr: np.ndarray,
    masks_np: Optional[np.ndarray],   # (N, H, W) bool or None
    cls_ids_np: np.ndarray,           # (N,) int64
    bboxes_np: np.ndarray,            # (N, 4) float32  [x1,y1,x2,y2]
    num_classes: int,
    h: int, w: int,
    categories: List[str],
    instance_colors: List[tuple],
    output_dir: str,
    save_vis: bool,
    mask_store: Optional[dict] = None,  # when not None: store mask here instead of disk
) -> None:
    """Write mask_{name}.npy (or store in mask_store) for a single frame."""
    frame_masks = np.zeros((num_classes, h, w), dtype=np.uint8)

    if save_vis:
        vis_img = frame_bgr.copy()
        vis_clean = frame_bgr.copy()
        overlay_layer = np.zeros_like(frame_bgr)
        mask_accum = np.zeros((h, w), dtype=bool)

    if masks_np is not None and masks_np.shape[0] > 0:
        for c_id in np.unique(cls_ids_np):
            c_id = int(c_id)
            if c_id >= num_classes:
                continue
            indices = np.where(cls_ids_np == c_id)[0]

            color = instance_colors[c_id % len(instance_colors)] if save_vis else None
            class_mask_accum = np.zeros((h, w), dtype=bool)

            for idx in indices:
                m = masks_np[idx]
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                class_mask_accum = class_mask_accum | m

                if save_vis:
                    overlay_layer[m] = color
                    mask_accum = mask_accum | m

                    x1, y1, x2, y2 = map(int, bboxes_np[idx][:4])
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

                    label = categories[c_id]
                    (tw, _), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(
                        vis_img, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    )

            frame_masks[c_id][class_mask_accum] = 1

    if mask_store is not None:
        mask_store[name] = frame_masks          # store in-memory (caller owns dict)
    else:
        np.save(os.path.join(output_dir, f"mask_{name}.npy"), frame_masks)
        del frame_masks

    if save_vis:
        if np.any(mask_accum):
            alpha = 0.5
            vis_clean[mask_accum] = cv2.addWeighted(
                vis_clean[mask_accum], 1 - alpha, overlay_layer[mask_accum], alpha, 0
            )
            vis_img[mask_accum] = cv2.addWeighted(
                vis_img[mask_accum], 1 - alpha, overlay_layer[mask_accum], alpha, 0
            )

        vis_dir = os.path.join(output_dir, "vis")
        vis_clean_dir = os.path.join(output_dir, "vis_clean")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(vis_clean_dir, exist_ok=True)

        cv2.imwrite(os.path.join(vis_dir, f"{name}.jpg"), vis_img)
        cv2.imwrite(os.path.join(vis_clean_dir, f"{name}.jpg"), vis_clean)
