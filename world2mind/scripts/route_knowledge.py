"""
Route Knowledge Cognitive Map Generation Module.

This module generates a grid-based cognitive map representing:
1. Traversable areas (floor, ground, road, sidewalk, etc.)
2. Camera trajectory through the scene

The output is a grid map showing which cells are traversable and
the path the camera took through the scene.
"""

import os
import glob
import logging
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Optional, Set, List, Tuple, Dict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class RouteKnowledgeGenerator:
    """
    Generates route knowledge cognitive map from semantic segmentation
    and camera pose data.
    """

    def __init__(
        self,
        da3_dir: str,
        image_dir: str,
        sam_dir: str,
        output_dir: str,
        traversable_categories: List[str],
        grid_divisions: int = 10,
        min_points_per_cell: int = 50,
        simplify_trajectory: bool = True,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        pixel_conf_threshold: float = 1.1,
        point_skip: int = 10,
        voxel_size: float = 0.1,
        skip_frames: Optional[Set[str]] = None,
        scene_bounds: Optional[Dict] = None,
        dyn_mask_loader=None,
        memory_cache=None,
    ):
        """
        Initialize route knowledge generator.

        Args:
            da3_dir: DA3 output directory (contains depth, pose, intrinsics)
            image_dir: Original images directory
            sam_dir: SAM3 output directory (contains semantic masks)
            output_dir: Output directory for route knowledge results
            traversable_categories: List of category names considered traversable
            grid_divisions: Number of grid divisions (NxN)
            min_points_per_cell: Minimum points to consider a cell traversable
            simplify_trajectory: Whether to remove duplicate consecutive cells
            min_depth: Minimum depth distance to include
            max_depth: Maximum depth distance to include
            pixel_conf_threshold: Minimum confidence for each pixel (skip pixels below this)
            point_skip: Point sampling rate for each frame
            voxel_size: Voxel size for downsampling accumulated points (meters)
            skip_frames: Set of frame names to skip
            scene_bounds: Full scene bounds dict with min_x, max_x, min_z, max_z
            dyn_mask_loader: DynamicMaskLoader instance for excluding moving objects
            memory_cache: SceneMemoryCache instance (skips all disk reads when set)
        """
        self.da3_dir = Path(da3_dir)
        self.image_dir = Path(image_dir)
        self.sam_dir = Path(sam_dir)
        self.output_dir = Path(output_dir)
        self.memory_cache = memory_cache

        self.traversable_categories = [c.lower() for c in traversable_categories]
        self.grid_divisions = grid_divisions
        self.min_points_per_cell = min_points_per_cell
        self.simplify_trajectory = simplify_trajectory
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pixel_conf_threshold = pixel_conf_threshold
        self.point_skip = max(1, point_skip)
        self.voxel_size = voxel_size
        self.skip_frames = skip_frames or set()
        self.scene_bounds = scene_bounds
        self.dyn_mask_loader = dyn_mask_loader

        # Data storage
        self.class_names = []
        self.traversable_indices = []
        self.camera_positions = []  # List of (x, z) camera positions
        self.traversable_points_set = set()  # Use set with voxel keys for deduplication

    def run(self) -> Optional[Dict]:
        """
        Run the route knowledge generation pipeline.

        Returns:
            Route knowledge dictionary or None if failed
        """
        logger.info("Starting Route Knowledge Generation...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load class information
        has_traversable = self._load_class_info()
        if not has_traversable:
            logger.info("No traversable categories found in scene, will only generate camera trajectory")

        # Load frame data
        frame_names = self._get_frame_names()
        if not frame_names:
            logger.error("No frames found")
            return None

        # Process frames to extract traversable points and camera trajectory
        self._process_frames(frame_names, extract_traversable=has_traversable)

        if not self.traversable_points_set and not self.camera_positions:
            logger.warning("No traversable points or camera positions found")
            return None

        # Generate route knowledge map
        route_data = self._generate_route_map()

        return route_data

    def _load_class_info(self) -> bool:
        """Load class information and identify traversable categories."""
        # Mode A: in-memory cache
        if self.memory_cache is not None and self.memory_cache.sam3_class_info is not None:
            class_info = self.memory_cache.sam3_class_info
            self.class_names = class_info.get('class_names', [])
            self.mask_height = class_info.get('height', 720)
            self.mask_width = class_info.get('width', 1280)
        else:
            # Mode B: disk
            class_info_path = self.sam_dir / "class_info.npy"
            if not class_info_path.exists():
                logger.error(f"Class info not found at {class_info_path}")
                return False
            class_info = np.load(class_info_path, allow_pickle=True).item()
            self.class_names = class_info.get('class_names', [])
            self.mask_height = class_info.get('height', 720)
            self.mask_width = class_info.get('width', 1280)

        # Find indices of traversable categories
        self.traversable_indices = []
        for idx, name in enumerate(self.class_names):
            name_lower = name.lower()
            for trav_cat in self.traversable_categories:
                if trav_cat in name_lower or name_lower in trav_cat:
                    self.traversable_indices.append(idx)
                    logger.info(f"Found traversable category: {name} (index {idx})")
                    break

        return len(self.traversable_indices) > 0

    def _get_frame_names(self) -> List[str]:
        """Get list of frame names to process."""
        # Mode A: in-memory cache
        if self.memory_cache is not None and self.memory_cache.has_da3():
            frame_names = [
                name for name in sorted(self.memory_cache.da3_depths.keys())
                if name not in self.skip_frames
            ]
            logger.info(f"Found {len(frame_names)} frames to process (from memory cache)")
            return frame_names

        # Mode B: disk
        depth_files = sorted(glob.glob(os.path.join(self.da3_dir, "depth", "*.npy")))
        depth_files = [f for f in depth_files if not f.endswith("_conf.npy")]

        frame_names = []
        for f in depth_files:
            name = os.path.splitext(os.path.basename(f))[0]
            if name not in self.skip_frames:
                frame_names.append(name)

        logger.info(f"Found {len(frame_names)} frames to process")
        return frame_names

    def _process_frames(self, frame_names: List[str], extract_traversable: bool = True):
        """Process all frames to extract traversable points and camera positions."""
        from tqdm import tqdm

        logger.info(f"Pixel confidence threshold: {self.pixel_conf_threshold}")
        if self.dyn_mask_loader is not None:
            logger.info("Dynamic mask filtering enabled")
        total_pixels = 0
        filtered_pixels = 0
        dynamic_filtered_pixels = 0

        use_cache = self.memory_cache is not None and self.memory_cache.has_da3()

        for frame_name in tqdm(frame_names, desc="Processing frames for route knowledge"):
            # Load pose (camera position)
            if use_cache:
                c2w_raw = self.memory_cache.da3_poses.get(frame_name)
                if c2w_raw is not None:
                    c2w = c2w_raw
                    cam_x = c2w[0, 3]
                    cam_z = c2w[2, 3]
                    self.camera_positions.append((cam_x, cam_z, frame_name))
            else:
                pose_path = self.da3_dir / "pose" / f"{frame_name}.npy"
                if pose_path.exists():
                    c2w = np.load(pose_path)
                    cam_x = c2w[0, 3]
                    cam_z = c2w[2, 3]
                    self.camera_positions.append((cam_x, cam_z, frame_name))

            # Skip traversable extraction if not needed
            if not extract_traversable:
                continue

            if use_cache:
                # --- Mode A: read all arrays from memory cache ---
                depth = self.memory_cache.da3_depths.get(frame_name)
                if depth is None:
                    continue
                K = self.memory_cache.da3_intrinsics.get(frame_name)
                if K is None:
                    continue
                conf = self.memory_cache.da3_confs.get(frame_name)
                semantic_mask = self.memory_cache.sam3_masks.get(frame_name)
                if semantic_mask is None:
                    continue
                c2w = self.memory_cache.da3_poses.get(frame_name)
                if c2w is None:
                    continue
            else:
                # --- Mode B: read all arrays from disk ---
                # Load depth
                depth_path = self.da3_dir / "depth" / f"{frame_name}.npy"
                if not depth_path.exists():
                    continue
                depth = np.load(depth_path)

                # Load intrinsics
                intrinsics_path = self.da3_dir / "intrinsics" / f"{frame_name}.npy"
                if not intrinsics_path.exists():
                    continue
                K = np.load(intrinsics_path)

                # Load confidence map if available
                conf_path = self.da3_dir / "depth" / f"{frame_name}_conf.npy"
                conf = np.load(conf_path) if conf_path.exists() else None

                # Load semantic mask
                mask_path = self.sam_dir / f"mask_{frame_name}.npy"
                if not mask_path.exists():
                    continue
                semantic_mask = np.load(mask_path)

                # Load pose for projection
                pose_path = self.da3_dir / "pose" / f"{frame_name}.npy"
                if not pose_path.exists():
                    continue
                c2w = np.load(pose_path)

            # Load dynamic mask if available
            dyn_mask = None
            if self.dyn_mask_loader is not None:
                try:
                    frame_idx = int(frame_name.split('_')[-1])
                    dyn_mask = self.dyn_mask_loader.get_mask(frame_idx)
                except (ValueError, IndexError):
                    pass

            # Extract traversable points from this frame
            frame_total, frame_filtered, frame_dyn_filtered = self._extract_traversable_points(
                depth, K, c2w, semantic_mask, conf, dyn_mask
            )
            total_pixels += frame_total
            filtered_pixels += frame_filtered
            dynamic_filtered_pixels += frame_dyn_filtered

        if filtered_pixels > 0:
            filter_ratio = filtered_pixels / total_pixels * 100 if total_pixels > 0 else 0
            logger.info(f"Pixel-level confidence filtering: {filtered_pixels:,} pixels filtered ({filter_ratio:.1f}%)")

        if dynamic_filtered_pixels > 0:
            dyn_ratio = dynamic_filtered_pixels / total_pixels * 100 if total_pixels > 0 else 0
            logger.info(f"Dynamic mask filtering: {dynamic_filtered_pixels:,} pixels filtered ({dyn_ratio:.1f}%)")

    def _extract_traversable_points(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        c2w: np.ndarray,
        semantic_mask: np.ndarray,
        conf: Optional[np.ndarray] = None,
        dyn_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, int, int]:
        """
        Extract 3D points for traversable areas and project to ground plane.

        Args:
            depth: Depth map (H_depth, W_depth)
            K: Camera intrinsics (3x3)
            c2w: Camera-to-world transformation (4x4)
            semantic_mask: Semantic segmentation mask (num_classes, H_mask, W_mask)
            conf: Confidence map (H_conf, W_conf), optional
            dyn_mask: Dynamic mask (H, W) where 1=dynamic, 0=static, optional

        Returns:
            Tuple of (total_valid_pixels, conf_filtered_pixels, dyn_filtered_pixels) for statistics
        """
        H_depth, W_depth = depth.shape[:2]

        # Resize confidence map if needed
        if conf is not None and (conf.shape[0] != H_depth or conf.shape[1] != W_depth):
            conf = cv2.resize(conf, (W_depth, H_depth), interpolation=cv2.INTER_LINEAR)

        # Resize dynamic mask if needed
        if dyn_mask is not None and (dyn_mask.shape[0] != H_depth or dyn_mask.shape[1] != W_depth):
            dyn_mask = cv2.resize(dyn_mask.astype(np.uint8), (W_depth, H_depth), interpolation=cv2.INTER_NEAREST)

        # Handle multi-channel mask format (num_classes, H, W)
        if semantic_mask.ndim == 3:
            num_classes, H_mask, W_mask = semantic_mask.shape

            # Create combined traversable mask from relevant channels
            traversable_mask_full = np.zeros((H_mask, W_mask), dtype=bool)
            for idx in self.traversable_indices:
                if idx < num_classes:
                    traversable_mask_full |= (semantic_mask[idx] > 0)

            # Resize traversable mask to match depth resolution
            if (H_mask, W_mask) != (H_depth, W_depth):
                traversable_mask_resized = cv2.resize(
                    traversable_mask_full.astype(np.uint8),
                    (W_depth, H_depth),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                traversable_mask_resized = traversable_mask_full
        else:
            # Single channel mask (H, W) with class indices
            H_mask, W_mask = semantic_mask.shape
            traversable_mask_full = np.zeros((H_mask, W_mask), dtype=bool)
            for idx in self.traversable_indices:
                traversable_mask_full |= (semantic_mask == idx)

            if (H_mask, W_mask) != (H_depth, W_depth):
                traversable_mask_resized = cv2.resize(
                    traversable_mask_full.astype(np.uint8),
                    (W_depth, H_depth),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                traversable_mask_resized = traversable_mask_full

        # Create pixel grid
        u = np.arange(W_depth)
        v = np.arange(H_depth)
        u, v = np.meshgrid(u, v)

        # Subsample
        u = u[::self.point_skip, ::self.point_skip]
        v = v[::self.point_skip, ::self.point_skip]
        d = depth[::self.point_skip, ::self.point_skip]
        mask = traversable_mask_resized[::self.point_skip, ::self.point_skip]
        conf_sub = conf[::self.point_skip, ::self.point_skip] if conf is not None else None
        dyn_sub = dyn_mask[::self.point_skip, ::self.point_skip] if dyn_mask is not None else None

        # Filter by depth
        valid_depth = (d > self.min_depth) & (d < self.max_depth)

        # Combine with traversable mask
        valid = valid_depth & mask

        # Count total valid pixels before confidence filtering
        total_valid = np.sum(valid)
        filtered_count = 0
        dyn_filtered_count = 0

        # Apply pixel-level confidence filtering
        if conf_sub is not None:
            conf_valid = conf_sub >= self.pixel_conf_threshold
            before_conf = np.sum(valid)
            valid = valid & conf_valid
            filtered_count = before_conf - np.sum(valid)

        # Apply dynamic mask filtering (exclude moving objects)
        if dyn_sub is not None:
            static_mask = dyn_sub == 0  # 0 = static, 1 = dynamic
            before_dyn = np.sum(valid)
            valid = valid & static_mask
            dyn_filtered_count = before_dyn - np.sum(valid)

        if not np.any(valid):
            return total_valid, filtered_count, dyn_filtered_count

        u_valid = u[valid].flatten()
        v_valid = v[valid].flatten()
        d_valid = d[valid].flatten()

        # Unproject to camera coordinates
        # Note: K (intrinsics) is already scaled to depth resolution by DA3,
        # so we use it directly without additional scaling
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (u_valid - cx) * d_valid / fx
        y_cam = (v_valid - cy) * d_valid / fy
        z_cam = d_valid

        # Stack as homogeneous coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)

        # Transform to world coordinates
        points_world = (c2w @ points_cam.T).T[:, :3]

        # Directly trust segmentation results without height filtering
        # This allows traversable areas with height variations (stairs, slopes, etc.)

        # Use voxel-based deduplication to reduce memory usage
        # Vectorized voxel key computation (replaces slow Python for-loop)
        voxel_x = (points_world[:, 0] / self.voxel_size).astype(np.int64)
        voxel_z = (points_world[:, 2] / self.voxel_size).astype(np.int64)
        keys = set(zip(voxel_x.tolist(), voxel_z.tolist()))
        self.traversable_points_set.update(keys)

        return total_valid, filtered_count, dyn_filtered_count

    def _generate_route_map(self) -> Dict:
        """Generate the route knowledge map structure."""
        # Convert voxel keys back to world coordinates (center of each voxel)
        traversable_points = [
            ((vx + 0.5) * self.voxel_size, (vz + 0.5) * self.voxel_size)
            for vx, vz in self.traversable_points_set
        ]

        # Use provided scene bounds or calculate from points
        if self.scene_bounds is not None:
            scene_min_x = self.scene_bounds['min_x']
            scene_max_x = self.scene_bounds['max_x']
            scene_min_z = self.scene_bounds['min_z']
            scene_max_z = self.scene_bounds['max_z']
        else:
            # Fallback: calculate from traversable points and camera positions
            all_x = [p[0] for p in traversable_points] + [p[0] for p in self.camera_positions]
            all_z = [p[1] for p in traversable_points] + [p[1] for p in self.camera_positions]

            if not all_x or not all_z:
                logger.warning("No points to generate route map")
                return {}

            margin = 0.5
            scene_min_x = min(all_x) - margin
            scene_max_x = max(all_x) + margin
            scene_min_z = min(all_z) - margin
            scene_max_z = max(all_z) + margin

        scene_width = scene_max_x - scene_min_x
        scene_height = scene_max_z - scene_min_z

        cell_width = scene_width / self.grid_divisions
        cell_height = scene_height / self.grid_divisions

        # Build traversable grid
        traversable_grid = defaultdict(int)
        for x, z in traversable_points:
            col = int((x - scene_min_x) / cell_width)
            row = int((z - scene_min_z) / cell_height)
            col = max(0, min(col, self.grid_divisions - 1))
            row = max(0, min(row, self.grid_divisions - 1))
            traversable_grid[(row, col)] += 1

        # Filter by minimum points
        traversable_cells = set()
        for (row, col), count in traversable_grid.items():
            if count >= self.min_points_per_cell:
                traversable_cells.add((row, col))

        # Build camera trajectory
        camera_trajectory = []
        for x, z, frame_name in self.camera_positions:
            col = int((x - scene_min_x) / cell_width)
            row = int((z - scene_min_z) / cell_height)
            col = max(0, min(col, self.grid_divisions - 1))
            row = max(0, min(row, self.grid_divisions - 1))
            camera_trajectory.append((row, col, frame_name))

        # Simplify trajectory (remove consecutive duplicates)
        if self.simplify_trajectory and camera_trajectory:
            simplified = [camera_trajectory[0]]
            for i in range(1, len(camera_trajectory)):
                if (camera_trajectory[i][0], camera_trajectory[i][1]) != \
                   (simplified[-1][0], simplified[-1][1]):
                    simplified.append(camera_trajectory[i])
            camera_trajectory = simplified

        # Format trajectory as string
        trajectory_str = " --> ".join([f"({r}, {c})" for r, c, _ in camera_trajectory])

        # Build route knowledge data structure
        fmt_coord = lambda x, y: f"({round(float(x), 3)}, {round(float(y), 3)})"

        # Compute estimated floor area from traversable cells
        estimated_floor_area = round(len(traversable_cells) * cell_width * cell_height, 2)

        route_data = {
            "Route_Knowledge_Metadata": {
                "Coordinate_System": "Grid (NxN cells)",
                "Grid_Divisions": self.grid_divisions,
                "Cell_Size": fmt_coord(cell_width, cell_height),
                "Scene_Bounds": {
                    "Min": fmt_coord(scene_min_x, scene_min_z),
                    "Max": fmt_coord(scene_max_x, scene_max_z)
                },
                "Total_Traversable_Cells": len(traversable_cells),
                "Estimated_Floor_Area_m2": estimated_floor_area,
                "Trajectory_Length": len(camera_trajectory),
                "Note": (
                    "Estimated_Floor_Area_m2 is the traversable floor area only. "
                    "Actual room area ≈ Floor_Area + furniture footprint area. "
                    "Do NOT use Scene_Bounds to calculate room area — Scene_Bounds "
                    "is the axis-aligned bounding box of the entire 3D reconstruction "
                    "and always significantly overestimates room size."
                ),
                "Description": "Route knowledge map showing traversable areas and camera movement path"
            },
            "Traversable_Grid": sorted([f"({r}, {c})" for r, c in traversable_cells]),
            "Camera_Trajectory": trajectory_str
        }

        # Store for visualization
        self._scene_bounds = (scene_min_x, scene_max_x, scene_min_z, scene_max_z)
        self._cell_size = (cell_width, cell_height)
        self._traversable_cells = traversable_cells
        self._camera_trajectory = camera_trajectory

        return route_data

    def save_route_knowledge(
        self,
        route_data: Dict,
        filename: str = "route_knowledge.yaml"
    ):
        """Save route knowledge to YAML file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            yaml.dump(route_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved route knowledge to {output_path}")

    def visualize_route_knowledge(
        self,
        route_data: Dict,
        output_path: str,
        traversable_color: str = "#44AA44",
        trajectory_color: str = "#FF4444",
        camera_orientations: Optional[list] = None,
    ):
        """
        Visualize route knowledge map.

        Args:
            route_data: Route knowledge dictionary
            output_path: Output image path
            traversable_color: Color for traversable cells
            trajectory_color: Color for camera trajectory
        """
        import warnings
        warnings.filterwarnings("ignore", message=".*FancyArrowPatch.*")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import to_rgba

        fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
        ax.set_facecolor('#FFFFFF')

        if not hasattr(self, '_scene_bounds'):
            logger.warning("No scene bounds available for visualization")
            return

        scene_min_x, scene_max_x, scene_min_z, scene_max_z = self._scene_bounds
        cell_width, cell_height = self._cell_size

        # Draw grid
        for row in range(self.grid_divisions):
            for col in range(self.grid_divisions):
                cell_x = scene_min_x + col * cell_width
                cell_z = scene_min_z + row * cell_height

                if (row, col) in self._traversable_cells:
                    # Traversable cell
                    rect = patches.Rectangle(
                        (cell_x, cell_z),
                        cell_width,
                        cell_height,
                        facecolor=traversable_color,
                        alpha=0.5,
                        edgecolor='#333333',
                        linewidth=0.8
                    )
                    ax.add_patch(rect)
                else:
                    # Non-traversable cell
                    rect = patches.Rectangle(
                        (cell_x, cell_z),
                        cell_width,
                        cell_height,
                        facecolor='#F0F0F0',
                        alpha=0.3,
                        edgecolor='#CCCCCC',
                        linewidth=0.5
                    )
                    ax.add_patch(rect)

                # Add grid coordinates
                ax.text(
                    cell_x + cell_width / 2,
                    cell_z + cell_height / 2,
                    f"({row},{col})",
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='#666666',
                    alpha=0.7
                )

        # Draw camera trajectory
        if self._camera_trajectory:
            traj_color = to_rgba(trajectory_color)

            # Draw trajectory line
            traj_x = []
            traj_z = []
            for row, col, _ in self._camera_trajectory:
                cell_center_x = scene_min_x + (col + 0.5) * cell_width
                cell_center_z = scene_min_z + (row + 0.5) * cell_height
                traj_x.append(cell_center_x)
                traj_z.append(cell_center_z)

            # Draw line with arrows
            for i in range(len(traj_x) - 1):
                ax.annotate(
                    '',
                    xy=(traj_x[i + 1], traj_z[i + 1]),
                    xytext=(traj_x[i], traj_z[i]),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=trajectory_color,
                        lw=2.5,
                        mutation_scale=15
                    ),
                    zorder=10
                )

            # Draw start and end markers
            if traj_x:
                # Start point (green circle)
                ax.scatter(
                    traj_x[0], traj_z[0],
                    s=200, c='#00AA00', marker='o',
                    edgecolors='white', linewidths=2,
                    zorder=15, label='Start'
                )
                ax.text(
                    traj_x[0], traj_z[0] + cell_height * 0.6,
                    'START',
                    fontsize=8, ha='center', fontweight='bold',
                    color='#00AA00', zorder=15
                )

                # End point (red square)
                ax.scatter(
                    traj_x[-1], traj_z[-1],
                    s=200, c='#AA0000', marker='s',
                    edgecolors='white', linewidths=2,
                    zorder=15, label='End'
                )
                ax.text(
                    traj_x[-1], traj_z[-1] + cell_height * 0.6,
                    'END',
                    fontsize=8, ha='center', fontweight='bold',
                    color='#AA0000', zorder=15
                )

        # Draw camera orientations (FOV wedge + heading arrow)
        if camera_orientations:
            import matplotlib.colors as mcolors
            from matplotlib.patches import Wedge

            cam_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            fov_radius = max(cell_width, cell_height) * 1.2
            arrow_len = fov_radius * 1.15
            fov_half_angle = 30  # degrees

            def darken(hex_color, factor=0.6):
                rgb = mcolors.to_rgb(hex_color)
                return tuple(c * factor for c in rgb)

            for i, co in enumerate(camera_orientations):
                cx, cz = co["position"]
                heading_deg = co["heading_deg"]
                color = cam_colors[i % len(cam_colors)]
                dark_color = darken(color)

                # FOV wedge: convert heading (0°=+Z, clockwise) to matplotlib (0°=+X, CCW)
                mpl_center_angle = 90 - heading_deg
                wedge = Wedge(
                    (cx, cz), fov_radius,
                    mpl_center_angle - fov_half_angle,
                    mpl_center_angle + fov_half_angle,
                    facecolor=color, alpha=0.25,
                    edgecolor=color, linewidth=1.2, zorder=18,
                )
                ax.add_patch(wedge)

                # Heading arrow (darker, from camera origin)
                heading_rad = np.radians(co["heading_deg"])
                dx = np.sin(heading_rad) * arrow_len
                dz = np.cos(heading_rad) * arrow_len
                ax.annotate(
                    '', xy=(cx + dx, cz + dz), xytext=(cx, cz),
                    arrowprops=dict(
                        arrowstyle='->', color=dark_color, lw=2.5,
                        mutation_scale=18,
                    ),
                    zorder=20,
                )

                # Camera origin dot
                ax.plot(cx, cz, 'o', markersize=6, color=dark_color,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=21)

                # Label: "img N"
                label = f"img {co['image_index']}"
                ax.text(
                    cx, cz - cell_height * 0.45, label,
                    fontsize=8, ha='center', va='top', fontweight='bold',
                    color=dark_color,
                    bbox=dict(
                        facecolor='white', alpha=0.9, edgecolor=color,
                        pad=1.5, boxstyle='round,pad=0.3', linewidth=1
                    ),
                    zorder=22
                )

        # Styling
        ax.set_xlim(scene_min_x - cell_width * 0.5, scene_max_x + cell_width * 0.5)
        ax.set_ylim(scene_min_z - cell_height * 0.5, scene_max_z + cell_height * 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (meters)', fontsize=12, fontweight='bold')

        # Title
        total_trav = len(self._traversable_cells)
        traj_len = len(self._camera_trajectory)
        ax.set_title(
            f"Route Knowledge Map\n"
            f"Traversable Cells: {total_trav} | Trajectory Waypoints: {traj_len}",
            fontsize=14, pad=20, fontweight='bold'
        )

        # Legend
        legend_elements = [
            patches.Patch(facecolor=traversable_color, alpha=0.5, label='Traversable Area'),
            patches.Patch(facecolor='#F0F0F0', alpha=0.3, edgecolor='#CCCCCC', label='Non-traversable'),
            plt.Line2D([0], [0], color=trajectory_color, linewidth=2.5, label='Camera Path'),
        ]
        if camera_orientations:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                           markersize=8, label='Camera FOV + Heading')
            )
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            fontsize=9
        )

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved route knowledge visualization to {output_path}")


def generate_route_knowledge(
    da3_dir: str,
    image_dir: str,
    sam_dir: str,
    output_dir: str,
    scene_name: str,
    config,
    skip_frames: Optional[Set[str]] = None,
    max_depth: float = 10.0,
    scene_bounds: Optional[Dict] = None,
    pixel_conf_threshold: float = 1.1,
    dyn_mask_loader=None,
    camera_orientations: Optional[list] = None,
    memory_cache=None,
) -> Optional[Dict]:
    """
    Main entry point for route knowledge generation.

    Args:
        da3_dir: DA3 output directory
        image_dir: Original images directory
        sam_dir: SAM3 output directory
        output_dir: Output directory
        scene_name: Scene identifier
        config: RouteKnowledgeConfig object
        skip_frames: Set of frame names to skip
        max_depth: Maximum depth distance to include
        scene_bounds: Full scene bounds dict with min_x, max_x, min_z, max_z
        pixel_conf_threshold: Minimum confidence for each pixel (skip pixels below this)
        dyn_mask_loader: DynamicMaskLoader instance for excluding moving objects
        memory_cache: SceneMemoryCache instance (skips all disk reads when set)

    Returns:
        Route knowledge dictionary or None
    """
    generator = RouteKnowledgeGenerator(
        da3_dir=da3_dir,
        image_dir=image_dir,
        sam_dir=sam_dir,
        output_dir=output_dir,
        traversable_categories=config.traversable_categories,
        grid_divisions=config.grid_divisions,
        min_points_per_cell=config.min_points_per_cell,
        simplify_trajectory=config.simplify_trajectory,
        max_depth=max_depth,
        point_skip=config.point_skip,
        voxel_size=config.voxel_size,
        skip_frames=skip_frames,
        scene_bounds=scene_bounds,
        pixel_conf_threshold=pixel_conf_threshold,
        dyn_mask_loader=dyn_mask_loader,
        memory_cache=memory_cache,
    )

    route_data = generator.run()

    if route_data:
        # Add scene name
        route_data["Route_Knowledge_Metadata"]["Scene"] = scene_name

        # Add camera orientations if available
        if camera_orientations:
            fmt_coord = lambda x, y: f"({round(float(x), 3)}, {round(float(y), 3)})"
            route_data["Camera_Orientations"] = [
                {
                    "Frame": co["frame_name"],
                    "Image": f"image {co['image_index']}",
                    "Position": fmt_coord(*co["position"]),
                    "Heading_Deg": co["heading_deg"],
                }
                for co in camera_orientations
            ]

        # Save YAML
        generator.save_route_knowledge(route_data, "route_knowledge.yaml")

        # Save visualization
        if config.save_visualization:
            vis_path = os.path.join(output_dir, "route_knowledge_visualization.png")
            generator.visualize_route_knowledge(
                route_data,
                vis_path,
                traversable_color=config.traversable_color,
                trajectory_color=config.trajectory_color,
                camera_orientations=camera_orientations,
            )

    return route_data
