"""
Point cloud mapping and AST generation module.
Supports two modes:
- voxel: Build voxel grid from point cloud
- direct: Save point cloud directly for AST generation (no voxelization)
"""

import os
import gc
import glob
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Set, List, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """Generate n distinct colors."""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


class DepthDataWrapper:
    """Wrapper for loading depth estimation outputs (disk or in-memory cache)."""

    def __init__(self, da3_dir: Path, image_dir: Path,
                 skip_frames: Optional[Set[str]] = None,
                 memory_cache=None):   # SceneMemoryCache | None
        self.da3_dir = Path(da3_dir)
        self.image_dir = Path(image_dir)
        self.skip_frames = skip_frames or set()
        self.memory_cache = memory_cache

        if memory_cache is not None and memory_cache.has_da3():
            # Build frame list from cache keys
            all_names = sorted(memory_cache.da3_depths.keys())
            self.frame_names = [n for n in all_names if n not in self.skip_frames]
        else:
            # Find all depth files on disk
            all_depth_files = sorted(glob.glob(os.path.join(da3_dir, "depth", "*.npy")))
            depth_files = [f for f in all_depth_files if not f.endswith("_conf.npy")]
            self.frame_names = []
            for f in depth_files:
                name = os.path.splitext(os.path.basename(f))[0]
                if name not in self.skip_frames:
                    self.frame_names.append(name)

        self.num_frames = len(self.frame_names)
        src = "cache" if (memory_cache is not None and memory_cache.has_da3()) else "disk"
        logger.info(f"Found {self.num_frames} frames in DA3 output ({src}).")

    def get_data(self, idx: int) -> Tuple:
        """Get depth, confidence, intrinsics, pose, and image for a frame."""
        frame_name = self.frame_names[idx]

        if self.memory_cache is not None and self.memory_cache.has_da3():
            depth = self.memory_cache.da3_depths[frame_name]
            conf  = self.memory_cache.da3_confs.get(frame_name)
            K     = self.memory_cache.da3_intrinsics[frame_name]
            c2w   = self.memory_cache.da3_poses[frame_name]
        else:
            depth_path = self.da3_dir / "depth" / f"{frame_name}.npy"
            depth = np.load(depth_path)
            conf_path = self.da3_dir / "depth" / f"{frame_name}_conf.npy"
            conf = np.load(conf_path) if conf_path.exists() else None
            K    = np.load(self.da3_dir / "intrinsics" / f"{frame_name}.npy")
            c2w  = np.load(self.da3_dir / "pose" / f"{frame_name}.npy")

        # Image always loaded from disk (original extracted frames, cheap)
        img = None
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = self.image_dir / f"{frame_name}{ext}"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                break

        return frame_name, depth, conf, K, c2w, img


class SemanticProjector:
    """Projects semantic masks to 3D (disk or in-memory cache)."""

    def __init__(self, sam_dir: Path, num_classes: int, memory_cache=None):
        self.sam_dir = Path(sam_dir)
        self.num_classes = num_classes
        self.memory_cache = memory_cache

    def load_semantic_mask(self, frame_name: str) -> Optional[np.ndarray]:
        """Load semantic mask for a frame."""
        if self.memory_cache is not None and self.memory_cache.has_sam3():
            return self.memory_cache.sam3_masks.get(frame_name)
        mask_path = self.sam_dir / f"mask_{frame_name}.npy"
        if mask_path.exists():
            return np.load(mask_path)
        return None


class GlobalMapping:
    """Global point cloud mapping with semantic labels."""

    def __init__(
        self,
        da3_dir: str,
        image_dir: str,
        sam_dir: str,
        output_dir: str,
        mode: str = "direct",
        grid_size: float = 0.05,
        ceiling_height: Optional[float] = None,
        min_height: Optional[float] = None,
        min_depth: float = 0.1,
        max_depth: float = 50.0,
        pixel_conf_threshold: float = 1.1,
        denoise_k: int = 10,
        min_points_per_voxel: int = 3,
        point_skip: int = 5,
        flip_y: bool = False,
        skip_frames: Optional[Set[str]] = None,
        dyn_mask_loader=None,
        show_progress: bool = True,
        camera_orientations: list = None,
        use_gpu: bool = True,
        memory_cache=None,   # SceneMemoryCache | None
    ):
        """
        Initialize global mapping.

        Args:
            da3_dir: DA3 output directory
            image_dir: Original images directory
            sam_dir: SAM3 output directory
            output_dir: Output directory for mapping results
            mode: "direct" (save point cloud) or "voxel" (build voxel grid)
            grid_size: Voxel grid resolution in meters (used for both modes)
            ceiling_height: Maximum height to include (None = no limit)
            min_height: Minimum height to include (None = no limit)
            min_depth: Minimum depth distance to include
            max_depth: Maximum depth distance to include
            pixel_conf_threshold: Minimum confidence for each pixel (skip pixels below this)
            denoise_k: Remove connected components smaller than this
            min_points_per_voxel: Minimum points required per voxel
            point_skip: Point sampling rate (1 = keep all)
            flip_y: Whether to flip Y axis
            skip_frames: Set of frame names to skip
            dyn_mask_loader: DynamicMaskLoader instance for excluding moving objects
            show_progress: Whether to show progress bar
            camera_orientations: Optional list of camera orientation dicts (for few-frame scenes)
        """
        self.da3_dir = Path(da3_dir)
        self.image_dir = Path(image_dir)
        self.sam_dir = Path(sam_dir)
        self.output_dir = Path(output_dir)

        self.mode = mode
        self.grid_size = grid_size
        self.ceiling_height = ceiling_height
        self.min_height = min_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pixel_conf_threshold = pixel_conf_threshold
        self.denoise_k = denoise_k
        self.min_points_per_voxel = min_points_per_voxel
        self.point_skip = max(1, point_skip)
        self.flip_y = flip_y
        self.skip_frames = skip_frames or set()
        self.dyn_mask_loader = dyn_mask_loader
        self.show_progress = show_progress
        self.camera_orientations = camera_orientations
        self.use_gpu = use_gpu
        self.memory_cache = memory_cache

    def run(self):
        """Run the mapping pipeline."""
        logger.info(f"Starting Global Mapping (mode={self.mode})...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._load_metadata()

        if self.use_gpu:
            try:
                points, colors, labels = self._generate_semantic_point_cloud_gpu()
            except Exception as e:
                logger.warning(f"GPU back-projection failed ({e}), falling back to CPU")
                points, colors, labels = self._generate_semantic_point_cloud_cpu()
        else:
            points, colors, labels = self._generate_semantic_point_cloud_cpu()

        if len(points) == 0:
            logger.error("No points generated!")
            return

        # Apply height filtering
        points, colors, labels = self._filter_by_height(points, colors, labels)

        if len(points) == 0:
            logger.error("No points after height filtering!")
            return

        # Save point cloud visualization
        self._save_point_cloud_images(points, colors, labels)

        if self.mode == "direct":
            if self.memory_cache is not None:
                logger.info("Storing point cloud in memory cache...")
                self._cache_point_cloud_data(points, colors, labels)
            else:
                logger.info("Saving point cloud data (direct mode)...")
                self._save_point_cloud_data(points, colors, labels)
        else:
            # Voxel mode: build and save voxel grid
            logger.info("Building Voxel Grid...")
            grid, origin, grid_shape = self._build_voxel_grid(points, labels)

            if grid is not None:
                logger.info("Refining Voxel Grid...")
                cleaned_grid = self._refine_grid(grid)
                self._save_grid_data(cleaned_grid, origin)
                logger.info("Rendering Voxel Grid Views...")
                self._render_grid_views(cleaned_grid)
            else:
                logger.error("Voxel grid generation failed!")

        logger.info("Mapping done.")

    def _load_metadata(self):
        """Load class information from SAM output (disk or cache)."""
        if self.memory_cache is not None and self.memory_cache.sam3_class_info is not None:
            self.class_info = self.memory_cache.sam3_class_info
        else:
            class_info_path = self.sam_dir / "class_info.npy"
            if not class_info_path.exists():
                raise FileNotFoundError(f"Class info not found at {class_info_path}")
            self.class_info = np.load(class_info_path, allow_pickle=True).item()

        self.num_classes = self.class_info['num_classes']
        self.class_names = self.class_info['class_names']

        self.wrapper = DepthDataWrapper(
            self.da3_dir, self.image_dir,
            skip_frames=self.skip_frames,
            memory_cache=self.memory_cache,
        )
        self.projector = SemanticProjector(
            self.sam_dir, self.num_classes,
            memory_cache=self.memory_cache,
        )
        self.colors = generate_distinct_colors(self.num_classes)

    def _generate_semantic_point_cloud_gpu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GPU-accelerated back-projection using PyTorch.

        Key design: batch-load ALL frame arrays first (CPU I/O phase), then do a
        single large GPU transfer + batched projection kernel.  This avoids the
        per-frame host→device transfer overhead that made the naïve frame-loop
        version slower than CPU.
        """
        import torch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            logger.warning("CUDA not available — falling back to CPU back-projection")
            return self._generate_semantic_point_cloud_cpu()

        n_frames = self.wrapper.num_frames
        logger.info(f"Back-projecting frames (GPU batch)  n={n_frames}")
        logger.info(f"Pixel confidence threshold: {self.pixel_conf_threshold}")

        # ── Phase A: CPU I/O — load all arrays ───────────────────────────────
        depths_list, confs_list, Ks_list, c2ws_list, imgs_list, masks_list = \
            [], [], [], [], [], []
        valid_frame_indices: List[int] = []
        dyn_masks_list: List[Optional[np.ndarray]] = []

        for i in range(n_frames):
            frame_name, depth, conf, K, c2w, img = self.wrapper.get_data(i)
            if img is None:
                continue
            h, w = depth.shape
            mask = self.projector.load_semantic_mask(frame_name)

            # Resize on CPU (cheap)
            if conf is not None and (conf.shape[0] != h or conf.shape[1] != w):
                conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            if mask is not None and (mask.shape[1] != h or mask.shape[2] != w):
                mt = mask.transpose(1, 2, 0).astype(np.uint8)
                mr = cv2.resize(mt, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = mr[np.newaxis] if mr.ndim == 2 else mr.transpose(2, 0, 1)

            # Dynamic mask
            dyn = None
            if self.dyn_mask_loader is not None:
                try:
                    fidx = int(frame_name.split('_')[-1])
                    dyn = self.dyn_mask_loader.get_mask(fidx)
                    if dyn is not None:
                        if dyn.shape[0] != h or dyn.shape[1] != w:
                            dyn = cv2.resize(dyn.astype(np.uint8), (w, h),
                                             interpolation=cv2.INTER_NEAREST)
                except (ValueError, IndexError):
                    pass

            valid_frame_indices.append(i)
            depths_list.append(depth.astype(np.float32))
            confs_list.append(conf.astype(np.float32) if conf is not None else None)
            Ks_list.append(K.astype(np.float32))
            c2ws_list.append(c2w.astype(np.float32))
            imgs_list.append(img.astype(np.float32) / 255.0)
            masks_list.append(mask)
            dyn_masks_list.append(dyn)

        if not valid_frame_indices:
            return np.array([]), np.array([]), np.array([])

        h, w = depths_list[0].shape
        F = len(valid_frame_indices)

        # ── Phase B: single batch transfer to GPU ────────────────────────────
        # Stack fixed-shape arrays; masks may be None or variable C — handle separately
        depths_np  = np.stack(depths_list,  axis=0)          # [F,H,W]
        imgs_np    = np.stack(imgs_list,    axis=0)          # [F,H,W,3]
        Ks_np      = np.stack(Ks_list,      axis=0)          # [F,3,3]
        c2ws_np    = np.stack(c2ws_list,    axis=0)          # [F,4,4]
        del depths_list, imgs_list, Ks_list, c2ws_list

        depths_t = torch.from_numpy(depths_np).to(device)    # [F,H,W]
        imgs_t   = torch.from_numpy(imgs_np).to(device)      # [F,H,W,3]
        Ks_t     = torch.from_numpy(Ks_np).to(device)        # [F,3,3]
        c2ws_t   = torch.from_numpy(c2ws_np).to(device)      # [F,4,4]
        del depths_np, imgs_np, Ks_np, c2ws_np

        # Confidence: stack only frames that have it; others leave as None flag
        has_conf = [c is not None for c in confs_list]
        if any(has_conf):
            confs_padded = np.stack([
                c if c is not None else np.ones((h, w), dtype=np.float32) * 1e9
                for c in confs_list
            ])
            confs_t = torch.from_numpy(confs_padded).to(device)   # [F,H,W]
            has_conf_t = torch.tensor(has_conf, device=device)
            del confs_padded
        else:
            confs_t = None
            has_conf_t = None
        del confs_list

        # Masks: stack if all frames have same C; otherwise process per-frame
        C = self.num_classes
        valid_masks = [m for m in masks_list if m is not None]
        if len(valid_masks) == F:
            masks_np = np.stack([m.astype(np.float32) for m in masks_list])  # [F,C,H,W]
            masks_t  = torch.from_numpy(masks_np).to(device)
            del masks_np
        else:
            masks_t = None  # fall back to per-frame below
        del masks_list, valid_masks

        # ── Phase C: batched GPU computation ─────────────────────────────────
        # Build pixel grid once [H,W]
        uu = torch.arange(w, device=device, dtype=torch.float32)
        vv = torch.arange(h, device=device, dtype=torch.float32)
        grid_v, grid_u = torch.meshgrid(vv, uu, indexing='ij')   # [H,W]
        grid_u_flat = grid_u.reshape(-1)   # [H*W]
        grid_v_flat = grid_v.reshape(-1)

        all_xyz:    List[np.ndarray] = []
        all_rgb:    List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        total_pixels    = 0
        filtered_pixels = 0
        dynamic_filtered_pixels = 0

        for fi in range(F):
            d_t   = depths_t[fi]          # [H,W]
            img_t = imgs_t[fi]            # [H,W,3]
            K_t   = Ks_t[fi]             # [3,3]
            c2w_t = c2ws_t[fi]           # [4,4]

            valid = (d_t > self.min_depth) & (d_t < self.max_depth)
            total_pixels += int(valid.sum())

            if confs_t is not None and has_conf_t[fi]:
                conf_ok = confs_t[fi] >= self.pixel_conf_threshold
                filtered_pixels += int((~conf_ok & valid).sum())
                valid = valid & conf_ok

            if dyn_masks_list[fi] is not None:
                static_t = torch.from_numpy(
                    (dyn_masks_list[fi] == 0).reshape(h, w)
                ).to(device)
                before_dyn = int(valid.sum())
                valid = valid & static_t
                dynamic_filtered_pixels += before_dyn - int(valid.sum())

            valid_1d = valid.reshape(-1)

            u_v = grid_u_flat[valid_1d]
            v_v = grid_v_flat[valid_1d]
            d_v = d_t.reshape(-1)[valid_1d]

            fx, fy = K_t[0, 0], K_t[1, 1]
            cx, cy = K_t[0, 2], K_t[1, 2]
            cam_pts = torch.stack([
                (u_v - cx) * d_v / fx,
                (v_v - cy) * d_v / fy,
                d_v,
            ], dim=1)                                               # [N,3]
            world_pts = cam_pts @ c2w_t[:3, :3].T + c2w_t[:3, 3]  # [N,3]
            colors_arr = img_t.reshape(-1, 3)[valid_1d]             # [N,3]

            frame_labels = torch.full((world_pts.shape[0],), -1,
                                      dtype=torch.int32, device=device)
            if masks_t is not None:
                mask_2d = masks_t[fi].reshape(C, -1)[:, valid_1d]  # [C,N]
                has_cls = mask_2d.any(dim=0)
                if has_cls.any():
                    frame_labels[has_cls] = mask_2d[:, has_cls].argmax(dim=0).int()

            skip = self.point_skip
            all_xyz.append(world_pts[::skip].cpu().numpy().astype(np.float32))
            all_rgb.append(colors_arr[::skip].cpu().numpy().astype(np.float32))
            all_labels.append(frame_labels[::skip].cpu().numpy())

            del d_t, img_t, valid, valid_1d, u_v, v_v, d_v
            del cam_pts, world_pts, colors_arr, frame_labels

        del depths_t, imgs_t, Ks_t, c2ws_t, confs_t, masks_t
        torch.cuda.empty_cache()

        if filtered_pixels > 0:
            ratio = filtered_pixels / total_pixels * 100 if total_pixels > 0 else 0
            logger.info(f"Pixel-level confidence filtering: {filtered_pixels:,} pixels filtered ({ratio:.1f}%)")
        if dynamic_filtered_pixels > 0:
            dyn_ratio = dynamic_filtered_pixels / total_pixels * 100 if total_pixels > 0 else 0
            logger.info(f"Dynamic mask filtering: {dynamic_filtered_pixels:,} pixels filtered ({dyn_ratio:.1f}%)")

        if all_xyz:
            points = np.concatenate(all_xyz); del all_xyz
            colors = np.concatenate(all_rgb); del all_rgb
            labels = np.concatenate(all_labels); del all_labels
            gc.collect()
            return points, colors, labels
        return np.array([]), np.array([]), np.array([])

    def _generate_semantic_point_cloud_cpu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate semantic point cloud from all frames with pixel-level confidence filtering."""
        all_xyz = []
        all_rgb = []
        all_labels = []

        logger.info("Back-projecting frames...")
        logger.info(f"Pixel confidence threshold: {self.pixel_conf_threshold}")
        if self.dyn_mask_loader is not None:
            logger.info("Dynamic mask filtering enabled")

        total_pixels = 0
        filtered_pixels = 0
        dynamic_filtered_pixels = 0

        iterator = tqdm(range(self.wrapper.num_frames), desc="Projecting") if self.show_progress else range(self.wrapper.num_frames)
        for i in iterator:
            frame_name, depth, conf, K, c2w, img = self.wrapper.get_data(i)
            if img is None:
                continue

            mask = self.projector.load_semantic_mask(frame_name)
            h, w = depth.shape

            if mask is not None:
                if mask.shape[1] != h or mask.shape[2] != w:
                    mask_t = mask.transpose(1, 2, 0).astype(np.uint8)
                    mask_resized = cv2.resize(mask_t, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = mask_resized[np.newaxis, :, :] if len(mask_resized.shape) == 2 else mask_resized.transpose(2, 0, 1)

            if img is not None:
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            # Resize confidence map if needed
            if conf is not None and (conf.shape[0] != h or conf.shape[1] != w):
                conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

            # Load dynamic mask if available
            dyn_mask = None
            if self.dyn_mask_loader is not None:
                # Extract frame index from frame name (e.g., "frame_000012" -> 12)
                try:
                    frame_idx = int(frame_name.split('_')[-1])
                    dyn_mask = self.dyn_mask_loader.get_mask(frame_idx)
                    if dyn_mask is not None and (dyn_mask.shape[0] != h or dyn_mask.shape[1] != w):
                        dyn_mask = cv2.resize(dyn_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                except (ValueError, IndexError):
                    pass

            u, v = np.meshgrid(np.arange(w), np.arange(h))
            u, v, d = u.flatten(), v.flatten(), depth.flatten()

            # Depth-based validity mask
            valid = (d > self.min_depth) & (d < self.max_depth)
            total_pixels += np.sum(valid)

            # Apply pixel-level confidence filtering
            if conf is not None:
                conf_flat = conf.flatten()
                conf_valid = conf_flat >= self.pixel_conf_threshold
                valid = valid & conf_valid
                filtered_pixels += np.sum(~conf_valid & (d > self.min_depth) & (d < self.max_depth))

            # Apply dynamic mask filtering (exclude moving objects)
            if dyn_mask is not None:
                dyn_flat = dyn_mask.flatten()
                static_mask = dyn_flat == 0  # 0 = static, 1 = dynamic
                before_dyn = np.sum(valid)
                valid = valid & static_mask
                dynamic_filtered_pixels += before_dyn - np.sum(valid)

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x = (u[valid] - cx) * d[valid] / fx
            y = (v[valid] - cy) * d[valid] / fy
            z = d[valid]

            cam_points = np.stack([x, y, z], axis=-1)
            R, t = c2w[:3, :3], c2w[:3, 3]
            world_points = (R @ cam_points.T).T + t

            colors_arr = img.reshape(-1, 3)[valid] / 255.0 if img is not None else np.zeros_like(world_points)
            frame_labels = np.full(len(world_points), -1, dtype=np.int32)

            if mask is not None:
                mask_valid = mask.reshape(self.num_classes, -1)[:, valid]
                has_class = np.any(mask_valid, axis=0)
                if np.any(has_class):
                    frame_labels[has_class] = np.argmax(mask_valid[:, has_class], axis=0)

            skip = self.point_skip
            all_xyz.append(world_points[::skip])
            all_rgb.append(colors_arr[::skip])
            all_labels.append(frame_labels[::skip])

            # Free per-frame intermediate arrays
            del depth, conf, K, c2w, img, mask, dyn_mask
            del u, v, d, valid
            del cam_points, world_points, colors_arr, frame_labels
            if (i + 1) % 10 == 0:
                gc.collect()

        if filtered_pixels > 0:
            filter_ratio = filtered_pixels / total_pixels * 100 if total_pixels > 0 else 0
            logger.info(f"Pixel-level confidence filtering: {filtered_pixels:,} pixels filtered ({filter_ratio:.1f}%)")

        if dynamic_filtered_pixels > 0:
            dyn_ratio = dynamic_filtered_pixels / total_pixels * 100 if total_pixels > 0 else 0
            logger.info(f"Dynamic mask filtering: {dynamic_filtered_pixels:,} pixels filtered ({dyn_ratio:.1f}%)")

        if all_xyz:
            points = np.concatenate(all_xyz); del all_xyz
            colors = np.concatenate(all_rgb); del all_rgb
            labels = np.concatenate(all_labels); del all_labels
            gc.collect()
            return points, colors, labels
        return np.array([]), np.array([]), np.array([])

    def _filter_by_height(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> Tuple:
        """Filter points by height."""
        heights = -points[:, 1] if self.flip_y else points[:, 1]

        if self.ceiling_height is not None or self.min_height is not None:
            ceil_mask = heights <= self.ceiling_height if self.ceiling_height is not None else np.ones(len(heights), dtype=bool)
            floor_mask = heights >= self.min_height if self.min_height is not None else np.ones(len(heights), dtype=bool)
            valid_mask = ceil_mask & floor_mask
            return points[valid_mask], colors[valid_mask], labels[valid_mask]
        return points, colors, labels

    def _save_point_cloud_data(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
        """Save point cloud data for direct mode AST generation."""
        # Calculate scene bounds from all points
        scene_bounds = {
            'min_x': float(np.min(points[:, 0])),
            'max_x': float(np.max(points[:, 0])),
            'min_y': float(np.min(points[:, 1])),
            'max_y': float(np.max(points[:, 1])),
            'min_z': float(np.min(points[:, 2])),
            'max_z': float(np.max(points[:, 2]))
        }

        # Save raw point cloud
        np.save(self.output_dir / "point_cloud.npy", {
            'points': points,
            'colors': colors,
            'labels': labels
        })

        # Save metadata with scene bounds
        np.save(self.output_dir / "point_cloud_meta.npy", {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'grid_size': self.grid_size,
            'mode': 'direct',
            'scene_bounds': scene_bounds
        })

        logger.info(f"Saved point cloud: {len(points)} points, {self.num_classes} classes")
        logger.info(f"Scene bounds: X=[{scene_bounds['min_x']:.2f}, {scene_bounds['max_x']:.2f}], "
                   f"Z=[{scene_bounds['min_z']:.2f}, {scene_bounds['max_z']:.2f}]")

    def _cache_point_cloud_data(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
        """Store point cloud in memory cache (skips disk write)."""
        scene_bounds = {
            'min_x': float(np.min(points[:, 0])), 'max_x': float(np.max(points[:, 0])),
            'min_y': float(np.min(points[:, 1])), 'max_y': float(np.max(points[:, 1])),
            'min_z': float(np.min(points[:, 2])), 'max_z': float(np.max(points[:, 2])),
        }
        self.memory_cache.pc_points      = points
        self.memory_cache.pc_colors      = colors
        self.memory_cache.pc_labels      = labels
        self.memory_cache.pc_class_names = self.class_names
        self.memory_cache.pc_scene_bounds = scene_bounds
        logger.info(f"Point cloud cached: {len(points)} points, {self.num_classes} classes")
        logger.info(f"Scene bounds: X=[{scene_bounds['min_x']:.2f}, {scene_bounds['max_x']:.2f}], "
                    f"Z=[{scene_bounds['min_z']:.2f}, {scene_bounds['max_z']:.2f}]")

    def _save_point_cloud_images(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
        """Save point cloud visualization images."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # RGB top-down view
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(points[:, 0], points[:, 2], c=np.clip(colors, 0, 1), s=0.5, alpha=0.8)
        self._draw_camera_markers(ax, points)
        ax.set_title("Point Cloud RGB (Top-Down)")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Z (meters)")
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(self.output_dir / "point_cloud_rgb_topdown.png", dpi=150)
        plt.close(fig)

        # Semantic top-down view
        fig, ax = plt.subplots(figsize=(14, 12))
        unlabeled = labels < 0
        if np.any(unlabeled):
            ax.scatter(points[unlabeled, 0], points[unlabeled, 2], c='lightgray', s=0.1, alpha=0.2, label='unlabeled')

        for i in range(self.num_classes):
            mask = labels == i
            if np.any(mask):
                ax.scatter(points[mask, 0], points[mask, 2], c=[self.colors[i]], s=0.8, alpha=0.7, label=self.class_names[i])

        self._draw_camera_markers(ax, points)
        ax.set_title("Point Cloud Semantic (Top-Down)")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Z (meters)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(self.output_dir / "point_cloud_semantic_topdown.png", dpi=150)
        plt.close(fig)

    def _draw_camera_markers(self, ax, points: np.ndarray):
        """Draw camera FOV wedges and heading arrows on a point cloud plot."""
        if not self.camera_orientations:
            return

        from matplotlib.patches import Wedge
        import matplotlib.colors as mcolors

        cam_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                       '#9467bd', '#8c564b', '#e377c2', '#17becf']

        x_range = points[:, 0].max() - points[:, 0].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        fov_radius = max(x_range, z_range) * 0.07
        arrow_len = fov_radius * 1.15
        fov_half_angle = 30  # degrees

        for i, cam in enumerate(self.camera_orientations):
            cx, cz = cam['position']
            heading_deg = cam['heading_deg']
            img_idx = cam['image_index']
            color = cam_colors[i % len(cam_colors)]
            dark_color = self._darken_color(color, 0.6)

            # FOV wedge: heading is 0°=+Z clockwise, matplotlib Wedge uses
            # 0°=+X counterclockwise, so convert: mpl_angle = 90 - heading
            mpl_center_angle = 90 - heading_deg
            wedge = Wedge(
                (cx, cz), fov_radius,
                mpl_center_angle - fov_half_angle,
                mpl_center_angle + fov_half_angle,
                facecolor=color, alpha=0.25,
                edgecolor=color, linewidth=1.2, zorder=9,
            )
            ax.add_patch(wedge)

            # Heading arrow (darker, from camera origin)
            heading_rad = np.radians(heading_deg)
            dx = np.sin(heading_rad) * arrow_len
            dz = np.cos(heading_rad) * arrow_len
            ax.annotate(
                '', xy=(cx + dx, cz + dz), xytext=(cx, cz),
                arrowprops=dict(
                    arrowstyle='->', color=dark_color, lw=2.0,
                    mutation_scale=14,
                ),
                zorder=11,
            )

            # Camera origin dot
            ax.plot(cx, cz, 'o', markersize=5, color=dark_color,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=12)

            # Label
            ax.annotate(
                f"img {img_idx}", (cx, cz),
                xytext=(6, 6), textcoords='offset points',
                fontsize=7, fontweight='bold', color=dark_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.85),
                zorder=13,
            )

    @staticmethod
    def _darken_color(hex_color, factor=0.6):
        """Darken a hex color by the given factor."""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(hex_color)
        return tuple(c * factor for c in rgb)

    def _build_voxel_grid(self, points: np.ndarray, labels: np.ndarray) -> Tuple:
        """Build voxel grid from point cloud."""
        if len(points) == 0:
            return None, None, None

        min_bound = np.floor(np.min(points, axis=0) / self.grid_size) * self.grid_size
        max_bound = np.ceil(np.max(points, axis=0) / self.grid_size) * self.grid_size

        grid_dims = ((max_bound - min_bound) / self.grid_size).astype(int) + 1
        voxel_grid = np.zeros((self.num_classes + 1, *grid_dims), dtype=np.float32)

        voxel_indices = ((points - min_bound) / self.grid_size).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)

        for vi, label in zip(voxel_indices, labels):
            if label >= 0:
                voxel_grid[label, vi[0], vi[1], vi[2]] += 1
            voxel_grid[self.num_classes, vi[0], vi[1], vi[2]] += 1

        return voxel_grid, min_bound, grid_dims

    def _refine_grid(self, grid: np.ndarray) -> np.ndarray:
        """Refine voxel grid by removing noise."""
        from scipy import ndimage

        cleaned = grid.copy()

        # Apply minimum points threshold
        total_counts = cleaned[self.num_classes]
        low_count_mask = total_counts < self.min_points_per_voxel
        cleaned[:, low_count_mask] = 0

        # Remove small connected components for each class
        for c in range(self.num_classes):
            binary = cleaned[c] > 0
            labeled, num_features = ndimage.label(binary)
            for region_id in range(1, num_features + 1):
                region_mask = labeled == region_id
                if np.sum(region_mask) < self.denoise_k:
                    cleaned[c][region_mask] = 0

        return cleaned

    def _save_grid_data(self, grid: np.ndarray, origin: np.ndarray):
        """Save voxel grid data."""
        np.save(self.output_dir / "voxel_grid.npy", grid)
        np.save(self.output_dir / "voxel_grid_meta.npy", {
            'origin': origin,
            'grid_size': self.grid_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'mode': self.mode
        })
        logger.info(f"Saved voxel grid: shape {grid.shape}")

    def _render_grid_views(self, grid: np.ndarray):
        """Render voxel grid visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Top-down view
        semantic_grid = grid[:self.num_classes]
        occupancy = np.sum(semantic_grid, axis=0) > 0
        top_down = np.any(occupancy, axis=1)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(top_down.T, origin='lower', cmap='gray')
        ax.set_title("Voxel Grid Top-Down View")
        fig.savefig(self.output_dir / "grid_view_topdown.png", dpi=100)
        plt.close(fig)


def run_mapping(
    da3_dir: str,
    image_dir: str,
    sam_dir: str,
    output_dir: str,
    mode: str = "direct",
    grid_size: float = 0.05,
    ceiling_height: Optional[float] = None,
    min_height: Optional[float] = None,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    pixel_conf_threshold: float = 1.1,
    denoise_k: int = 10,
    min_points_per_voxel: int = 3,
    point_skip: int = 5,
    flip_y: bool = False,
    skip_frames: Optional[Set[str]] = None,
    dyn_mask_loader=None,
    show_progress: bool = True,
    camera_orientations: list = None,
    use_gpu: bool = True,
    memory_cache=None,   # SceneMemoryCache | None
) -> bool:
    """
    Run global mapping pipeline.

    Args:
        mode: "direct" (save point cloud) or "voxel" (build voxel grid only)
        pixel_conf_threshold: Minimum confidence for each pixel (skip pixels below this)
        dyn_mask_loader: DynamicMaskLoader instance for excluding moving objects
        show_progress: Whether to show progress bar
        memory_cache: SceneMemoryCache for in-memory mode (skips disk I/O for intermediates)

    Returns:
        True if successful
    """
    try:
        mapper = GlobalMapping(
            da3_dir=da3_dir,
            image_dir=image_dir,
            sam_dir=sam_dir,
            output_dir=output_dir,
            mode=mode,
            grid_size=grid_size,
            ceiling_height=ceiling_height,
            min_height=min_height,
            min_depth=min_depth,
            max_depth=max_depth,
            pixel_conf_threshold=pixel_conf_threshold,
            denoise_k=denoise_k,
            min_points_per_voxel=min_points_per_voxel,
            point_skip=point_skip,
            flip_y=flip_y,
            skip_frames=skip_frames,
            dyn_mask_loader=dyn_mask_loader,
            show_progress=show_progress,
            camera_orientations=camera_orientations,
            use_gpu=use_gpu,
            memory_cache=memory_cache,
        )
        mapper.run()
        return True
    except Exception as e:
        logger.error(f"Mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False
