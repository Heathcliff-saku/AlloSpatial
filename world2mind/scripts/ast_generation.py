"""
AST (Abstract Syntax Tree) generation module for cognitive mapping.
Generates Absolute-Centric Semantic Tree from mapping data.

Supports two modes:
- voxel: Extract instances from voxel grid using connected components
- direct: Extract instances from point cloud using DBSCAN clustering
"""

import os
import logging
import yaml
import numpy as np
import cv2
import colorsys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import ndimage
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_category_eps_config(config_path: Optional[str] = None) -> Dict[str, float]:
    """
    Load category-specific eps configuration from YAML file.

    Args:
        config_path: Path to category_eps.yaml. If None, searches in default locations.

    Returns:
        Dictionary mapping category names to eps values, with 'default_eps' key.
    """
    default_config = {'default_eps': 0.15}

    if config_path is None:
        # Search in default locations
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'category_eps.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config', 'category_eps.yaml'),
        ]
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path is None or not os.path.exists(config_path):
        logger.warning("Category eps config not found, using default eps for all categories")
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        result = {'default_eps': config.get('default_eps', 0.15)}

        # Flatten categories into result dict
        if 'categories' in config:
            for cat_name, cat_config in config['categories'].items():
                if isinstance(cat_config, dict) and 'eps' in cat_config:
                    result[cat_name] = cat_config['eps']
                elif isinstance(cat_config, (int, float)):
                    result[cat_name] = cat_config

        logger.info(f"Loaded category eps config from {config_path} with {len(result)-1} categories")
        return result

    except Exception as e:
        logger.warning(f"Failed to load category eps config: {e}, using defaults")
        return default_config


class Instance:
    """Represents a detected object instance."""

    def __init__(self, node_id: str, category: str, points_2d: np.ndarray,
                 height_range: Tuple[float, float], point_count: int):
        """
        Initialize instance from 2D points (X-Z plane).

        Args:
            node_id: Unique identifier
            category: Category name
            points_2d: 2D points array (N, 2) in X-Z plane
            height_range: (min_y, max_y) height range
            point_count: Total number of 3D points
        """
        self.node_id = node_id
        self.category = category
        self.points_2d = points_2d
        self.point_count = point_count
        self.bottom, self.top = height_range

        # Calculate 2D bounding box
        self.min_2d = np.min(points_2d, axis=0)
        self.max_2d = np.max(points_2d, axis=0)

        # Calculate centroid
        self.centroid_2d = [float(np.mean(points_2d[:, 0])), float(np.mean(points_2d[:, 1]))]

        # Calculate minimum area rectangle using OpenCV
        if len(points_2d) >= 5:
            pts = points_2d.astype(np.float32)
            rect = cv2.minAreaRect(pts)
            (center, (w, h), angle) = rect

            # Ensure major >= minor
            if w >= h:
                self.major, self.minor = float(w), float(h)
                self.angle = float(angle)
            else:
                self.major, self.minor = float(h), float(w)
                self.angle = float(angle + 90)
        else:
            # Fallback to bbox
            dx = self.max_2d[0] - self.min_2d[0]
            dz = self.max_2d[1] - self.min_2d[1]
            self.major = float(max(dx, dz, 0.1))
            self.minor = float(min(dx, dz, 0.1))
            self.angle = 0.0 if dx >= dz else 90.0

        # Calculate area from convex hull
        if len(points_2d) >= 3:
            try:
                hull = cv2.convexHull(points_2d.astype(np.float32))
                self.area_m2 = float(cv2.contourArea(hull))
                if self.area_m2 < 0.001:
                    self.area_m2 = float(self.major * self.minor)
            except:
                self.area_m2 = float(self.major * self.minor)
        else:
            self.area_m2 = float(self.major * self.minor)

        # Hierarchy
        self.children = []
        self.parent = None
        self.relation = None

    def to_dict(self, ast_format: str = "ellipse") -> Dict:
        """Convert to dictionary for YAML output.

        Args:
            ast_format: Output format - "ellipse", "rectangle", or "grid"
        """
        fmt_val = lambda *args: f"({', '.join([str(round(float(a), 3)) for a in args])})"

        if ast_format == "ellipse":
            d = {
                "Node_ID": self.node_id,
                "Spatial_Pose": {
                    "Centroid_2D": fmt_val(self.centroid_2d[0], self.centroid_2d[1]),
                    "Ellipse_Axis": fmt_val(self.major, self.minor, self.angle),
                    "BBox_2D": fmt_val(self.min_2d[0], self.min_2d[1], self.max_2d[0], self.max_2d[1])
                },
                "Attributes": {
                    "Height_Range": fmt_val(self.bottom, self.top),
                    "Area_m2": round(float(self.area_m2), 4),
                    "Point_Count": self.point_count
                }
            }
        elif ast_format == "rectangle":
            # Rectangle format: center + width + height (bounding rectangle)
            width = float(self.max_2d[0] - self.min_2d[0])
            height = float(self.max_2d[1] - self.min_2d[1])
            d = {
                "Node_ID": self.node_id,
                "Spatial_Pose": {
                    "Center": fmt_val(self.centroid_2d[0], self.centroid_2d[1]),
                    "Width": round(width, 3),
                    "Height": round(height, 3),
                },
                "Attributes": {
                    "Height_Range": fmt_val(self.bottom, self.top),
                    "Area_m2": round(float(self.area_m2), 4),
                    "Point_Count": self.point_count
                }
            }
        else:
            # Default to ellipse format
            d = {
                "Node_ID": self.node_id,
                "Spatial_Pose": {
                    "Centroid_2D": fmt_val(self.centroid_2d[0], self.centroid_2d[1]),
                    "Ellipse_Axis": fmt_val(self.major, self.minor, self.angle),
                    "BBox_2D": fmt_val(self.min_2d[0], self.min_2d[1], self.max_2d[0], self.max_2d[1])
                },
                "Attributes": {
                    "Height_Range": fmt_val(self.bottom, self.top),
                    "Area_m2": round(float(self.area_m2), 4),
                    "Point_Count": self.point_count
                }
            }

        if self.children:
            d["Contains_Children"] = [child.to_dict_as_child(ast_format) for child in self.children]
        return d

    def to_dict_as_child(self, ast_format: str = "ellipse") -> Dict:
        """Convert to dictionary as a child node."""
        d = self.to_dict(ast_format)
        d["Relation_To_Parent"] = self.relation or "Adjacent_To"
        return d


# =============================================================================
# Direct Mode: DBSCAN-based instance extraction from point cloud
# =============================================================================

def voxel_downsample_2d(points_2d: np.ndarray, voxel_size: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample 2D points using voxel grid.

    Args:
        points_2d: 2D points (N, 2)
        voxel_size: Voxel size in meters

    Returns:
        Tuple of (downsampled_points, unique_indices)
    """
    if len(points_2d) == 0:
        return points_2d, np.array([], dtype=np.int64)

    # Convert to voxel indices
    voxel_indices = np.floor(points_2d / voxel_size).astype(np.int32)

    # Find unique voxels
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)

    return points_2d[unique_idx], unique_idx


def extract_core_region(points_2d: np.ndarray, percentile: float = 75.0, max_points_for_knn: int = 5000) -> np.ndarray:
    """
    Extract core region of a point cloud by removing low-density "tail" points.

    Uses adaptive density-based filtering:
    1. Compute local density for each point using KNN (with downsampling for large clouds)
    2. Find density threshold based on percentile
    3. Keep only points above threshold

    Args:
        points_2d: 2D points (N, 2)
        percentile: Percentile threshold for density filtering (default 75%)
                   Higher value = more aggressive filtering, smaller core region
        max_points_for_knn: Maximum points for KNN computation (downsample if larger)

    Returns:
        Filtered points representing the core region
    """
    from sklearn.neighbors import NearestNeighbors

    if len(points_2d) < 10:
        return points_2d

    # Convert to float32 for efficiency
    points_2d = np.asarray(points_2d, dtype=np.float32)

    # For large point clouds, use downsampling for density estimation
    if len(points_2d) > max_points_for_knn:
        # Random subsample for density estimation
        subsample_idx = np.random.choice(len(points_2d), max_points_for_knn, replace=False)
        subsample_points = points_2d[subsample_idx]

        # Compute density on subsample
        k = min(max(5, len(subsample_points) // 20), 30)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(subsample_points)

        # Query density for ALL points using the subsampled tree
        distances, _ = nbrs.kneighbors(points_2d)
        avg_distances = np.mean(distances[:, 1:], axis=1)
    else:
        # Original behavior for small point clouds
        k = min(max(5, len(points_2d) // 20), 30)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points_2d)
        distances, _ = nbrs.kneighbors(points_2d)
        avg_distances = np.mean(distances[:, 1:], axis=1)

    # Local density = inverse of average distance to k neighbors
    local_density = 1.0 / (avg_distances + 1e-6)

    # Adaptive threshold based on density distribution
    density_threshold = np.percentile(local_density, 100 - percentile)

    # Keep high-density points
    high_density_mask = local_density >= density_threshold

    if np.sum(high_density_mask) < 10:
        # If too few points remain, use lower threshold
        density_threshold = np.percentile(local_density, 50)
        high_density_mask = local_density >= density_threshold

    if np.sum(high_density_mask) < 5:
        return points_2d  # Return original if filtering too aggressive

    return points_2d[high_density_mask]


def _dbscan_cpu(points_2d: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run sklearn DBSCAN and return integer cluster labels array."""
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples,
                        algorithm='ball_tree', leaf_size=40)
    return clustering.fit_predict(points_2d)


def _dbscan_gpu(points_2d: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run cuML DBSCAN on GPU and return integer cluster labels array (numpy)."""
    import cuml
    import cudf
    df = cudf.DataFrame({'x': points_2d[:, 0].astype(np.float32),
                         'z': points_2d[:, 1].astype(np.float32)})
    clustering = cuml.DBSCAN(eps=float(eps), min_samples=int(min_samples),
                             metric='euclidean', calc_core_sample_indices=False)
    labels = clustering.fit_predict(df)
    return labels.to_numpy().astype(np.int32)


def extract_instances_from_point_cloud(
    points: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    eps: float = 0.15,
    min_samples: int = 10,
    min_points: int = 30,
    use_core_extraction: bool = True,
    core_percentile: float = 70.0,
    category_eps_config: Optional[Dict[str, float]] = None,
    use_downsampling: bool = True,
    downsample_voxel_size: float = 0.05,
    large_class_threshold: int = 10000,
    use_gpu: bool = True,
) -> List[Instance]:
    """
    Extract instances from point cloud using DBSCAN clustering.

    Args:
        points: Point cloud (N, 3)
        labels: Semantic labels (N,)
        class_names: List of class names
        eps: DBSCAN epsilon (neighborhood radius), used as default if category not in config
        min_samples: DBSCAN minimum samples per cluster
        min_points: Minimum points for valid instance
        use_core_extraction: Whether to extract core region to remove low-density tails
        core_percentile: Percentile for core extraction (higher = more aggressive)
        category_eps_config: Dictionary mapping category names to eps values
        use_downsampling: Enable voxel downsampling for large point clouds (memory optimization)
        downsample_voxel_size: Voxel size for downsampling large classes (meters)
        large_class_threshold: Point count threshold to trigger downsampling
        use_gpu: Whether to use cuML GPU-accelerated DBSCAN (falls back to sklearn on failure)

    Returns:
        List of Instance objects
    """
    import gc

    # cuML has significant per-call init overhead; only worth it for large point clouds.
    # Threshold: if a class has >= gpu_dbscan_min_points, use GPU; otherwise sklearn.
    GPU_DBSCAN_MIN_POINTS = 50_000

    _cuml_available = False
    if use_gpu:
        try:
            import cuml  # noqa: F401 — presence check
            _cuml_available = True
            logger.debug("cuML available — will use GPU DBSCAN for classes >= %d pts",
                         GPU_DBSCAN_MIN_POINTS)
        except Exception as e:
            logger.warning(f"cuML unavailable ({e}), falling back to sklearn DBSCAN")

    # Load category eps config if not provided
    if category_eps_config is None:
        category_eps_config = load_category_eps_config()

    default_eps = category_eps_config.get('default_eps', eps)

    num_classes = len(class_names)
    all_instances = []
    instance_counter = defaultdict(int)

    for c_idx in range(num_classes):
        category = class_names[c_idx]

        # Get category-specific eps, fallback to default
        cat_eps = category_eps_config.get(category, default_eps)

        # Get points for this class
        class_mask = labels == c_idx
        if not np.any(class_mask):
            continue

        class_points = points[class_mask]

        if len(class_points) < min_points:
            continue

        # Use 2D points (X-Z) for clustering to get top-down instances
        points_2d = class_points[:, [0, 2]]

        # Optionally downsample large point clouds to reduce memory and computation
        do_downsample = use_downsampling and len(points_2d) > large_class_threshold
        if do_downsample:
            points_2d = points_2d.astype(np.float32)
            points_2d_clustered, downsample_idx = voxel_downsample_2d(points_2d, downsample_voxel_size)
            adjusted_min_samples = max(3, min_samples // 5)
            logger.debug(f"Category '{category}': downsampled {len(points_2d):,} -> {len(points_2d_clustered):,} points")
        else:
            points_2d_clustered = points_2d
            adjusted_min_samples = min_samples

        # DBSCAN clustering — use cuML only when the point cloud is large enough
        # to amortise cuML's per-call init overhead (~150-200 ms).
        pts_for_cluster = points_2d_clustered.astype(np.float32)
        use_gpu_for_this = (_cuml_available and
                            len(pts_for_cluster) >= GPU_DBSCAN_MIN_POINTS)
        try:
            cluster_labels = (_dbscan_gpu(pts_for_cluster, cat_eps, adjusted_min_samples)
                              if use_gpu_for_this
                              else _dbscan_cpu(pts_for_cluster, cat_eps, adjusted_min_samples))
        except Exception as e:
            logger.warning(f"GPU DBSCAN failed for '{category}' ({e}), retrying on CPU")
            cluster_labels = _dbscan_cpu(pts_for_cluster, cat_eps, adjusted_min_samples)

        # Log eps used for this category
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.debug(f"Category '{category}': eps={cat_eps:.3f}, found {n_clusters} clusters")

        # Process each cluster
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Remove noise label

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id

            if do_downsample:
                original_cluster_idx = downsample_idx[cluster_mask]
                cluster_points_3d = class_points[original_cluster_idx]
                cluster_points_2d = points_2d[original_cluster_idx]
            else:
                cluster_points_3d = class_points[cluster_mask]
                cluster_points_2d = points_2d_clustered[cluster_mask]

            if len(cluster_points_2d) < min_points:
                continue

            # Extract core region to remove low-density "tail" caused by depth estimation errors
            if use_core_extraction and len(cluster_points_2d) > 20:
                core_points_2d = extract_core_region(cluster_points_2d, percentile=core_percentile)
                if len(core_points_2d) >= min_points and len(core_points_2d) >= len(cluster_points_2d) * 0.2:
                    cluster_points_2d = core_points_2d

            # Height range (use original 3D points for height calculation)
            height_min = float(np.min(cluster_points_3d[:, 1]))
            height_max = float(np.max(cluster_points_3d[:, 1]))

            instance_counter[category] += 1
            node_id = f"{category}_{instance_counter[category]:02d}"

            inst = Instance(
                node_id=node_id,
                category=category,
                points_2d=cluster_points_2d,
                height_range=(height_min, height_max),
                point_count=len(cluster_points_3d)
            )
            all_instances.append(inst)

        del cluster_labels
        gc.collect()

    logger.info(f"DBSCAN extracted {len(all_instances)} instances from point cloud")
    return all_instances


# =============================================================================
# Voxel Mode: Connected component-based instance extraction
# =============================================================================

def extract_instances_from_voxel_grid(
    grid: np.ndarray,
    meta: Dict,
    min_voxels: int = 30,
    use_dilation: bool = True
) -> List[Instance]:
    """
    Extract instances from voxel grid using connected components.

    Args:
        grid: Voxel grid [num_classes+1, X, Y, Z] or [num_classes, X, Y, Z]
        meta: Grid metadata
        min_voxels: Minimum voxels for valid instance
        use_dilation: Whether to use binary dilation to connect nearby voxels

    Returns:
        List of Instance objects
    """
    origin = meta['origin']
    grid_size = meta['grid_size']
    class_names = meta['class_names']
    num_classes = len(class_names)

    all_instances = []
    instance_counter = defaultdict(int)

    for c_idx in range(num_classes):
        category = class_names[c_idx]

        # Get binary mask for this class
        class_grid = grid[c_idx]
        binary_mask = class_grid > 0

        if not np.any(binary_mask):
            continue

        # Apply binary dilation to connect nearby voxels
        if use_dilation:
            struct = ndimage.generate_binary_structure(3, 2)  # 3D connectivity
            binary_mask = ndimage.binary_dilation(binary_mask, structure=struct, iterations=1)

        # Find connected components
        labeled_array, num_features = ndimage.label(binary_mask)

        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            voxel_count = np.sum(region_mask)

            if voxel_count < min_voxels:
                continue

            # Get voxel indices
            voxel_indices = np.argwhere(region_mask)

            # Convert to world coordinates
            world_points = voxel_indices.astype(float) * grid_size + origin

            # Extract 2D points (X-Z plane)
            points_2d = world_points[:, [0, 2]]

            # Get unique 2D points for better area estimation
            points_2d_unique = np.unique(np.round(points_2d / grid_size) * grid_size, axis=0)

            # Height range
            height_min = float(np.min(world_points[:, 1]))
            height_max = float(np.max(world_points[:, 1]))

            instance_counter[category] += 1
            node_id = f"{category}_{instance_counter[category]:02d}"

            inst = Instance(
                node_id=node_id,
                category=category,
                points_2d=points_2d_unique if len(points_2d_unique) >= 3 else points_2d,
                height_range=(height_min, height_max),
                point_count=int(voxel_count)
            )
            all_instances.append(inst)

    logger.info(f"Voxel grid extracted {len(all_instances)} instances")
    return all_instances


# =============================================================================
# Instance merging and hierarchy inference
# =============================================================================

def merge_nearby_instances(
    instances: List[Instance],
    merge_dist: float = 0.4
) -> List[Instance]:
    """
    Merge nearby instances of the same category.

    Args:
        instances: List of instances
        merge_dist: Distance threshold for merging (meters)

    Returns:
        List of merged instances
    """
    if not instances:
        return []

    # Group by category
    by_category = defaultdict(list)
    for inst in instances:
        by_category[inst.category].append(inst)

    merged_all = []

    for category, cat_instances in by_category.items():
        if len(cat_instances) <= 1:
            merged_all.extend(cat_instances)
            continue

        # Sort by area (largest first)
        cat_instances.sort(key=lambda x: x.area_m2, reverse=True)

        # Iterative merging
        merged = True
        iteration = 0
        max_iterations = 10

        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            new_instances = []
            used = set()

            for i, inst_a in enumerate(cat_instances):
                if i in used:
                    continue

                current_points = inst_a.points_2d.copy()
                current_heights = [(inst_a.bottom, inst_a.top)]
                current_count = inst_a.point_count

                for j, inst_b in enumerate(cat_instances[i + 1:], start=i + 1):
                    if j in used:
                        continue

                    # Calculate distance between centroids
                    centroid_dist = np.sqrt(
                        (inst_a.centroid_2d[0] - inst_b.centroid_2d[0]) ** 2 +
                        (inst_a.centroid_2d[1] - inst_b.centroid_2d[1]) ** 2
                    )

                    # Check if bounding boxes overlap or are close
                    bbox_gap_x = max(inst_b.min_2d[0] - inst_a.max_2d[0], inst_a.min_2d[0] - inst_b.max_2d[0])
                    bbox_gap_z = max(inst_b.min_2d[1] - inst_a.max_2d[1], inst_a.min_2d[1] - inst_b.max_2d[1])
                    bbox_dist = max(bbox_gap_x, bbox_gap_z)

                    # Merge if centroids are close OR bboxes overlap/touch
                    should_merge = centroid_dist < merge_dist or bbox_dist < merge_dist * 0.3

                    if should_merge:
                        # Merge
                        current_points = np.vstack([current_points, inst_b.points_2d])
                        current_heights.append((inst_b.bottom, inst_b.top))
                        current_count += inst_b.point_count
                        used.add(j)
                        merged = True

                # Create merged instance
                height_min = min(h[0] for h in current_heights)
                height_max = max(h[1] for h in current_heights)

                merged_inst = Instance(
                    node_id=inst_a.node_id,
                    category=category,
                    points_2d=current_points,
                    height_range=(height_min, height_max),
                    point_count=current_count
                )
                new_instances.append(merged_inst)

            cat_instances = new_instances

        # Renumber instances
        for idx, inst in enumerate(cat_instances, start=1):
            inst.node_id = f"{category}_{idx:02d}"

        merged_all.extend(cat_instances)

    logger.info(f"After merging: {len(merged_all)} instances")
    return merged_all


def infer_spatial_hierarchy(instances: List[Instance]) -> List[Instance]:
    """
    Infer spatial hierarchy (parent-child relationships).

    Args:
        instances: List of instances

    Returns:
        List of root instances (with children attached)
    """
    if not instances:
        return []

    # Sort by area (largest first) - larger objects are more likely to be parents
    instances.sort(key=lambda x: x.area_m2, reverse=True)

    roots = []

    for i, child in enumerate(instances):
        potential_parents = []

        for j in range(i):
            parent = instances[j]
            if parent.node_id == child.node_id:
                continue

            # Check horizontal containment/overlap
            child_cx, child_cz = child.centroid_2d
            parent_min_x, parent_min_z = parent.min_2d
            parent_max_x, parent_max_z = parent.max_2d

            # Expand parent bbox slightly for tolerance
            margin = 0.15
            in_parent_bbox = (
                parent_min_x - margin <= child_cx <= parent_max_x + margin and
                parent_min_z - margin <= child_cz <= parent_max_z + margin
            )

            if not in_parent_bbox:
                continue

            # Check vertical relationship
            vertical_gap = child.bottom - parent.top

            if -0.15 <= vertical_gap <= 0.35:
                # Child is on top of parent
                potential_parents.append((parent, "On_Top_Of", abs(vertical_gap)))
            elif child.bottom >= parent.bottom - 0.1 and child.top <= parent.top + 0.1:
                # Child is inside parent (vertically contained)
                potential_parents.append((parent, "Inside", 0))

        if potential_parents:
            # Choose the best parent (smallest area, or closest vertical relationship)
            potential_parents.sort(key=lambda x: (x[2], x[0].area_m2))
            best_parent, relation, _ = potential_parents[0]
            best_parent.children.append(child)
            child.parent = best_parent
            child.relation = relation
        else:
            roots.append(child)

    return roots


# =============================================================================
# AST building and visualization
# =============================================================================

def build_ast_structure(
    instances: List[Instance],
    roots: List[Instance],
    scene_name: str,
    class_names: List[str],
    origin: np.ndarray,
    ast_format: str = "ellipse",
    camera_orientations: list = None,
) -> Dict:
    """Build AST dictionary structure.

    Args:
        instances: All instances
        roots: Root instances (with children attached)
        scene_name: Scene name
        class_names: List of class names
        origin: Origin coordinates
        ast_format: Output format - "ellipse", "rectangle", or "grid"
    """
    # Count instances per category
    category_counts = defaultdict(int)
    for inst in instances:
        category_counts[inst.category] += 1

    fmt_coord = lambda x, y: f"({round(float(x), 3)}, {round(float(y), 3)})"

    format_names = {
        "ellipse": "Ellipse (major/minor axis, angle)",
        "rectangle": "Rectangle (center, width, height)",
        "grid": "Grid (NxN cells with semantic labels)"
    }

    ast_data = {
        "Map_Metadata": {
            "Coordinate_System": "2D Absolute Grid (meters)",
            "Origin_Ref": fmt_coord(origin[0], origin[2]) if origin is not None else "(0, 0)",
            "AST_Format": format_names.get(ast_format, ast_format),
        },
        "Spatial_Hierarchy": [root.to_dict(ast_format) for root in roots]
    }

    if camera_orientations:
        ast_data["Map_Metadata"]["Camera_Views"] = [
            {
                "Frame": co["frame_name"],
                "Image": f"image {co['image_index']}",
                "Position": fmt_coord(*co["position"]),
                "Heading_Deg": co["heading_deg"],
            }
            for co in camera_orientations
        ]

    return ast_data


def get_color_for_category(category: str) -> str:
    """Generate a deterministic color for a category."""
    hash_object = hashlib.md5(category.encode())
    hue = int(hash_object.hexdigest(), 16) % 360 / 360.0
    rgb = colorsys.hls_to_rgb(hue, 0.6, 0.7)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _draw_camera_orientations(ax, camera_orientations, scene_bounds=None):
    """Draw camera FOV wedges and heading arrows on a matplotlib axes.

    Args:
        ax: matplotlib Axes
        camera_orientations: list of dicts with position, heading_deg, image_index
        scene_bounds: optional dict with min_x, max_x, min_z, max_z for sizing
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Wedge

    if not camera_orientations:
        return

    cam_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Determine sizes based on scene extent
    if scene_bounds:
        sx = scene_bounds['max_x'] - scene_bounds['min_x']
        sz = scene_bounds['max_z'] - scene_bounds['min_z']
        fov_radius = max(sx, sz) * 0.08
    else:
        fov_radius = 0.3
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

        # FOV wedge: convert heading (0°=+Z, clockwise) to matplotlib angle (0°=+X, CCW)
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
                arrowstyle='->', color=dark_color, lw=2.5,
                mutation_scale=16,
            ),
            zorder=20,
        )

        # Camera origin dot
        ax.plot(cx, cz, 'o', markersize=5, color=dark_color,
                markeredgecolor='white', markeredgewidth=0.6, zorder=21)

        # Label
        label = f"img {co['image_index']}"
        ax.text(
            cx, cz - fov_radius * 0.55, label,
            fontsize=7, ha='center', va='top', fontweight='bold',
            color=dark_color,
            bbox=dict(
                facecolor='white', alpha=0.9, edgecolor=color,
                pad=1.5, boxstyle='round,pad=0.3', linewidth=1
            ),
            zorder=22,
        )

    # Return a legend handle
    return plt.Line2D(
        [0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
        markersize=8, label='Camera FOV + Heading'
    )


def _resolve_label_positions(instances: List[Instance], scene_bounds: Dict = None) -> Dict[str, Tuple[float, float]]:
    """
    Resolve label positions to avoid overlapping.

    Uses a simple greedy algorithm to place labels:
    1. Sort instances by area (larger first, as they are more important)
    2. For each instance, try to place label at preferred position
    3. If overlapping with existing labels, try alternative positions

    Args:
        instances: List of instances
        scene_bounds: Scene bounds for boundary checking

    Returns:
        Dictionary mapping node_id to (x, y) label position
    """
    if not instances:
        return {}

    # Sort by area (larger instances get priority for label placement)
    sorted_instances = sorted(instances, key=lambda x: x.area_m2, reverse=True)

    # Track placed label bounding boxes: (x_min, y_min, x_max, y_max)
    placed_labels = []
    label_positions = {}

    # Estimate label size (approximate)
    label_width = 0.4  # meters
    label_height = 0.15  # meters

    for inst in sorted_instances:
        cx, cz = inst.centroid_2d

        # Try different positions: top, right, left, bottom, top-right, top-left
        candidates = [
            (cx, cz + inst.minor / 2 + 0.12),  # top (preferred)
            (cx + inst.major / 2 + 0.15, cz),  # right
            (cx - inst.major / 2 - 0.15, cz),  # left
            (cx, cz - inst.minor / 2 - 0.12),  # bottom
            (cx + inst.major / 2 * 0.7, cz + inst.minor / 2 * 0.7 + 0.1),  # top-right
            (cx - inst.major / 2 * 0.7, cz + inst.minor / 2 * 0.7 + 0.1),  # top-left
        ]

        best_pos = None
        for pos_x, pos_y in candidates:
            # Check bounds
            if scene_bounds:
                if pos_x < scene_bounds['min_x'] or pos_x > scene_bounds['max_x']:
                    continue
                if pos_y < scene_bounds['min_z'] or pos_y > scene_bounds['max_z']:
                    continue

            # Check overlap with existing labels
            new_bbox = (pos_x - label_width/2, pos_y - label_height/2,
                       pos_x + label_width/2, pos_y + label_height/2)

            overlaps = False
            for existing in placed_labels:
                # Check if bboxes overlap
                if not (new_bbox[2] < existing[0] or new_bbox[0] > existing[2] or
                       new_bbox[3] < existing[1] or new_bbox[1] > existing[3]):
                    overlaps = True
                    break

            if not overlaps:
                best_pos = (pos_x, pos_y)
                break

        # If no good position found, use default (top) but mark as potentially overlapping
        if best_pos is None:
            best_pos = candidates[0]

        label_positions[inst.node_id] = best_pos
        placed_labels.append((best_pos[0] - label_width/2, best_pos[1] - label_height/2,
                             best_pos[0] + label_width/2, best_pos[1] + label_height/2))

    return label_positions


def draw_ast_layout(
    ast_data: Dict,
    output_path: str,
    instances: List[Instance],
    scene_bounds: Dict = None,
    show_all_labels: bool = False,
    camera_orientations: list = None,
):
    """
    Draw AST layout visualization with improved label placement.

    Args:
        ast_data: AST dictionary
        output_path: Output image path
        instances: List of instances for drawing
        scene_bounds: Full scene bounds dict with min_x, max_x, min_z, max_z
        show_all_labels: If True, show all labels; if False, only show labels for root/parent nodes
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*FancyArrowPatch.*")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.set_facecolor('#FAFAFA')

    categories = set()

    # Sort instances: draw larger ones first
    sorted_instances = sorted(instances, key=lambda x: x.area_m2, reverse=True)

    # Identify root/parent nodes (nodes without parents or with children)
    root_nodes = set()
    for inst in instances:
        if inst.parent is None:
            root_nodes.add(inst.node_id)
        if inst.children:
            root_nodes.add(inst.node_id)

    # Draw instances
    for inst in sorted_instances:
        color = get_color_for_category(inst.category)
        categories.add(inst.category)

        # Draw ellipse
        ellipse = patches.Ellipse(
            inst.centroid_2d,
            inst.major,
            inst.minor,
            angle=inst.angle,
            facecolor=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.8,
            linestyle='-'
        )
        ax.add_patch(ellipse)

        # Draw center point
        ax.scatter(inst.centroid_2d[0], inst.centroid_2d[1], color=color,
                   s=30, edgecolors='white', linewidths=1, zorder=10)

    # Collect labels to draw (use adjustText for automatic placement)
    texts = []

    # Determine which instances to label
    if show_all_labels:
        instances_to_label = sorted_instances
    else:
        # Only label root nodes and larger instances
        # Group by category and only label the largest instance per category
        category_largest = {}
        for inst in sorted_instances:
            if inst.category not in category_largest:
                category_largest[inst.category] = inst

        instances_to_label = []
        for inst in sorted_instances:
            # Label if: root node, or largest of its category, or has children
            is_largest = category_largest.get(inst.category) == inst
            if inst.node_id in root_nodes or is_largest or inst.children:
                instances_to_label.append(inst)

    # Draw labels with adjustText for automatic repositioning
    for inst in instances_to_label:
        color = get_color_for_category(inst.category)
        # Shorter label: just category name for cleaner look
        label_text = inst.category if len(instances_to_label) <= 15 else inst.category[:8]

        text = ax.text(
            inst.centroid_2d[0],
            inst.centroid_2d[1] + inst.minor / 2 + 0.1,
            label_text,
            fontsize=8,
            ha='center',
            fontweight='bold',
            color='#333333',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=color,
                     pad=2, boxstyle='round,pad=0.3', linewidth=1),
            zorder=11
        )
        texts.append(text)

    # Use adjustText to prevent overlapping (if available)
    try:
        adjust_text(texts, ax=ax,
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                   expand_points=(1.5, 1.5),
                   force_text=(0.5, 0.5),
                   force_points=(0.3, 0.3))
    except:
        # adjustText not available, use manual positioning
        label_positions = _resolve_label_positions(instances_to_label, scene_bounds)
        for text, inst in zip(texts, instances_to_label):
            if inst.node_id in label_positions:
                pos = label_positions[inst.node_id]
                text.set_position(pos)

    # Draw parent-child relationships
    for inst in instances:
        if inst.parent:
            ax.plot(
                [inst.parent.centroid_2d[0], inst.centroid_2d[0]],
                [inst.parent.centroid_2d[1], inst.centroid_2d[1]],
                color='#888888', linestyle=':', linewidth=1, alpha=0.6, zorder=5
            )

    # Styling
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC')
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (meters)', fontsize=12, fontweight='bold')

    # Set axis limits based on scene bounds or instances
    if scene_bounds is not None:
        margin = 0.5
        ax.set_xlim(scene_bounds['min_x'] - margin, scene_bounds['max_x'] + margin)
        ax.set_ylim(scene_bounds['min_z'] - margin, scene_bounds['max_z'] + margin)
    elif instances:
        all_x = [inst.centroid_2d[0] for inst in instances]
        all_z = [inst.centroid_2d[1] for inst in instances]
        all_major = [inst.major for inst in instances]
        margin = max(all_major) / 2 + 1.0 if all_major else 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    # Title
    scene = ast_data.get('Map_Metadata', {}).get('Scene', 'Unknown')
    total = ast_data.get('Map_Metadata', {}).get('Total_Instances', len(instances))
    labeled = len(instances_to_label)
    ax.set_title(f"Cognitive Map: {scene}\n({total} instances, {labeled} labeled)",
              fontsize=14, pad=20, fontweight='bold')

    # Legend (show all categories)
    legend_elements = [patches.Patch(facecolor=get_color_for_category(cat),
                                     alpha=0.6, label=cat, edgecolor=get_color_for_category(cat))
                       for cat in sorted(categories)]
    cam_handle = _draw_camera_orientations(ax, camera_orientations, scene_bounds)
    if cam_handle:
        legend_elements.append(cam_handle)
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  title="Categories", frameon=True, fontsize=9, title_fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved AST visualization (ellipse) to {output_path}")


def draw_ast_layout_rectangle(
    ast_data: Dict,
    output_path: str,
    instances: List[Instance],
    scene_bounds: Dict = None,
    show_all_labels: bool = False,
    camera_orientations: list = None,
):
    """
    Draw AST layout visualization with rectangles and improved label placement.

    Args:
        ast_data: AST dictionary
        output_path: Output image path
        instances: List of instances for drawing
        scene_bounds: Full scene bounds dict with min_x, max_x, min_z, max_z
        show_all_labels: If True, show all labels; if False, only show labels for root/parent nodes
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*FancyArrowPatch.*")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.set_facecolor('#F5F5F5')

    categories = set()

    # Sort instances: draw larger ones first
    sorted_instances = sorted(instances, key=lambda x: x.area_m2, reverse=True)

    # Identify root/parent nodes
    root_nodes = set()
    for inst in instances:
        if inst.parent is None:
            root_nodes.add(inst.node_id)
        if inst.children:
            root_nodes.add(inst.node_id)

    # Determine which instances to label
    if show_all_labels:
        instances_to_label = set(inst.node_id for inst in sorted_instances)
    else:
        category_largest = {}
        for inst in sorted_instances:
            if inst.category not in category_largest:
                category_largest[inst.category] = inst
        instances_to_label = set()
        for inst in sorted_instances:
            is_largest = category_largest.get(inst.category) == inst
            if inst.node_id in root_nodes or is_largest or inst.children:
                instances_to_label.add(inst.node_id)

    # Draw instances as rectangles
    for inst in sorted_instances:
        color = get_color_for_category(inst.category)
        categories.add(inst.category)

        # Calculate rectangle dimensions
        width = float(inst.max_2d[0] - inst.min_2d[0])
        height = float(inst.max_2d[1] - inst.min_2d[1])

        # Draw rectangle (bounding box)
        rect = patches.Rectangle(
            (inst.min_2d[0], inst.min_2d[1]),
            width,
            height,
            facecolor=color,
            alpha=0.4,
            edgecolor=color,
            linewidth=2.0,
            linestyle='-'
        )
        ax.add_patch(rect)

        # Draw center point with crosshair
        cx, cz = inst.centroid_2d
        ax.scatter(cx, cz, color=color, s=50, edgecolors='white', linewidths=1.5, zorder=10, marker='o')
        # Crosshair lines
        cross_size = min(width, height) * 0.15
        ax.plot([cx - cross_size, cx + cross_size], [cz, cz], color='white', linewidth=1.5, zorder=9)
        ax.plot([cx, cx], [cz - cross_size, cz + cross_size], color='white', linewidth=1.5, zorder=9)

    # Draw labels with adjustText
    texts = []
    labeled_instances = [inst for inst in sorted_instances if inst.node_id in instances_to_label]

    for inst in labeled_instances:
        color = get_color_for_category(inst.category)
        cx, cz = inst.centroid_2d
        label_text = inst.category if len(labeled_instances) <= 15 else inst.category[:8]

        text = ax.text(
            cx, inst.max_2d[1] + 0.15,
            label_text,
            fontsize=8, ha='center', fontweight='bold',
            color='#333333',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, pad=2, boxstyle='round', linewidth=1),
            zorder=11
        )
        texts.append(text)

    # Use adjustText to prevent overlapping
    try:
        adjust_text(texts, ax=ax,
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                   expand_points=(1.5, 1.5),
                   force_text=(0.5, 0.5),
                   force_points=(0.3, 0.3))
    except:
        pass  # adjustText not available

    # Draw parent-child relationships
    for inst in instances:
        if inst.parent:
            ax.annotate('',
                xy=(inst.centroid_2d[0], inst.centroid_2d[1]),
                xytext=(inst.parent.centroid_2d[0], inst.parent.centroid_2d[1]),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5, ls='--'),
                zorder=5
            )

    # Styling
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA')
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (meters)', fontsize=12, fontweight='bold')

    # Set axis limits based on scene bounds or instances
    if scene_bounds is not None:
        margin = 0.5
        ax.set_xlim(scene_bounds['min_x'] - margin, scene_bounds['max_x'] + margin)
        ax.set_ylim(scene_bounds['min_z'] - margin, scene_bounds['max_z'] + margin)
    elif instances:
        all_min_x = [inst.min_2d[0] for inst in instances]
        all_max_x = [inst.max_2d[0] for inst in instances]
        all_min_z = [inst.min_2d[1] for inst in instances]
        all_max_z = [inst.max_2d[1] for inst in instances]
        margin = 1.0
        ax.set_xlim(min(all_min_x) - margin, max(all_max_x) + margin)
        ax.set_ylim(min(all_min_z) - margin, max(all_max_z) + margin)

    # Title
    scene = ast_data.get('Map_Metadata', {}).get('Scene', 'Unknown')
    total = ast_data.get('Map_Metadata', {}).get('Total_Instances', len(instances))
    labeled = len(labeled_instances)
    ax.set_title(f"Cognitive Map (Rectangle): {scene}\n({total} instances, {labeled} labeled)",
              fontsize=14, pad=20, fontweight='bold')

    # Legend
    legend_elements = [patches.Patch(facecolor=get_color_for_category(cat),
                                     alpha=0.6, label=cat, edgecolor=get_color_for_category(cat))
                       for cat in sorted(categories)]
    cam_handle = _draw_camera_orientations(ax, camera_orientations, scene_bounds)
    if cam_handle:
        legend_elements.append(cam_handle)
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  title="Categories", frameon=True, fontsize=9, title_fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved AST visualization (rectangle) to {output_path}")


def build_grid_ast(
    instances: List[Instance],
    scene_name: str,
    class_names: List[str],
    origin: np.ndarray,
    grid_divisions: int = 10,
    scene_bounds: Dict = None,
    camera_orientations: list = None,
) -> Dict:
    """
    Build grid-based AST structure.

    The scene is divided into NxN grid cells, and each cell contains
    a list of semantic categories present in that cell.

    Args:
        instances: List of instances
        scene_name: Scene name
        class_names: List of class names
        origin: Origin coordinates
        grid_divisions: Number of grid divisions (NxN)
        scene_bounds: Full scene bounds dict with min_x, max_x, min_z, max_z

    Returns:
        Grid AST dictionary
    """
    # Use provided scene bounds or calculate from instances
    if scene_bounds is not None:
        scene_min_x = scene_bounds['min_x']
        scene_max_x = scene_bounds['max_x']
        scene_min_z = scene_bounds['min_z']
        scene_max_z = scene_bounds['max_z']
    elif instances:
        # Fallback: calculate from instances (legacy behavior)
        all_min_x = min(inst.min_2d[0] for inst in instances)
        all_max_x = max(inst.max_2d[0] for inst in instances)
        all_min_z = min(inst.min_2d[1] for inst in instances)
        all_max_z = max(inst.max_2d[1] for inst in instances)
        margin = 0.1
        scene_min_x = all_min_x - margin
        scene_max_x = all_max_x + margin
        scene_min_z = all_min_z - margin
        scene_max_z = all_max_z + margin
    else:
        return {
            "Map_Metadata": {
                "Coordinate_System": "Grid (NxN cells)",
                "Grid_Divisions": grid_divisions,
                "Scene": scene_name,
                "AST_Format": "Grid (NxN cells with semantic labels)",
                "Total_Instances": 0,
                "Categories": class_names
            },
            "Grid_Map": {}
        }

    scene_width = scene_max_x - scene_min_x
    scene_height = scene_max_z - scene_min_z

    cell_width = scene_width / grid_divisions
    cell_height = scene_height / grid_divisions

    # Build grid map: (row, col) -> list of categories
    grid_map = defaultdict(set)

    for inst in instances:
        # Find which grid cells this instance occupies
        # Use the bounding box of the instance
        min_col = int((inst.min_2d[0] - scene_min_x) / cell_width)
        max_col = int((inst.max_2d[0] - scene_min_x) / cell_width)
        min_row = int((inst.min_2d[1] - scene_min_z) / cell_height)
        max_row = int((inst.max_2d[1] - scene_min_z) / cell_height)

        # Clamp to valid range
        min_col = max(0, min(min_col, grid_divisions - 1))
        max_col = max(0, min(max_col, grid_divisions - 1))
        min_row = max(0, min(min_row, grid_divisions - 1))
        max_row = max(0, min(max_row, grid_divisions - 1))

        # Mark all cells that the instance occupies
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid_map[(row, col)].add(inst.category)

    # Convert to serializable format
    grid_data = {}
    for (row, col), categories in sorted(grid_map.items()):
        cell_key = f"({row}, {col})"
        grid_data[cell_key] = sorted(list(categories))

    # Count instances per category
    category_counts = defaultdict(int)
    for inst in instances:
        category_counts[inst.category] += 1

    fmt_coord = lambda x, y: f"({round(float(x), 3)}, {round(float(y), 3)})"

    # Compute estimated occupied area from grid cells containing objects
    occupied_cells = len(grid_map)
    estimated_occupied_area = round(occupied_cells * cell_width * cell_height, 2)

    ast_data = {
        "Map_Metadata": {
            "Coordinate_System": "Grid (NxN cells)",
            "Grid_Divisions": grid_divisions,
            "Cell_Size": fmt_coord(cell_width, cell_height),
            "Scene_Bounds": {
                "Min": fmt_coord(scene_min_x, scene_min_z),
                "Max": fmt_coord(scene_max_x, scene_max_z)
            },
            "Occupied_Cells": occupied_cells,
            "Estimated_Occupied_Area_m2": estimated_occupied_area,
            "Origin_Ref": fmt_coord(origin[0], origin[2]) if origin is not None else "(0, 0)",
            "Note": (
                "Scene_Bounds is the axis-aligned bounding box of the entire 3D "
                "reconstruction, NOT the room area. Do NOT multiply Scene_Bounds "
                "width × height to estimate room size. Use Estimated_Occupied_Area_m2 "
                "or combine with route knowledge Estimated_Floor_Area_m2 instead."
            ),
            "AST_Format": "Grid (NxN cells with semantic labels)",
        },
        "Grid_Map": grid_data
    }

    if camera_orientations:
        ast_data["Map_Metadata"]["Camera_Views"] = [
            {
                "Frame": co["frame_name"],
                "Image": f"image {co['image_index']}",
                "Position": fmt_coord(*co["position"]),
                "Heading_Deg": co["heading_deg"],
            }
            for co in camera_orientations
        ]

    return ast_data


def draw_grid_ast(
    ast_data: Dict,
    output_path: str,
    instances: List[Instance],
    grid_divisions: int = 10,
    scene_bounds: Dict = None,
    camera_orientations: list = None,
):
    """
    Draw grid-based AST visualization.

    Args:
        ast_data: AST dictionary
        output_path: Output image path
        instances: List of instances
        grid_divisions: Number of grid divisions
        scene_bounds: Full scene bounds dict with min_x, max_x, min_z, max_z
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

    # Use provided scene bounds or calculate from instances
    if scene_bounds is not None:
        scene_min_x = scene_bounds['min_x']
        scene_max_x = scene_bounds['max_x']
        scene_min_z = scene_bounds['min_z']
        scene_max_z = scene_bounds['max_z']
    elif instances:
        all_min_x = min(inst.min_2d[0] for inst in instances)
        all_max_x = max(inst.max_2d[0] for inst in instances)
        all_min_z = min(inst.min_2d[1] for inst in instances)
        all_max_z = max(inst.max_2d[1] for inst in instances)
        margin = 0.1
        scene_min_x = all_min_x - margin
        scene_max_x = all_max_x + margin
        scene_min_z = all_min_z - margin
        scene_max_z = all_max_z + margin
    else:
        fig.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return

    scene_width = scene_max_x - scene_min_x
    scene_height = scene_max_z - scene_min_z

    cell_width = scene_width / grid_divisions
    cell_height = scene_height / grid_divisions

    # Build grid map
    grid_map = defaultdict(set)
    for inst in instances:
        min_col = int((inst.min_2d[0] - scene_min_x) / cell_width)
        max_col = int((inst.max_2d[0] - scene_min_x) / cell_width)
        min_row = int((inst.min_2d[1] - scene_min_z) / cell_height)
        max_row = int((inst.max_2d[1] - scene_min_z) / cell_height)

        min_col = max(0, min(min_col, grid_divisions - 1))
        max_col = max(0, min(max_col, grid_divisions - 1))
        min_row = max(0, min(min_row, grid_divisions - 1))
        max_row = max(0, min(max_row, grid_divisions - 1))

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid_map[(row, col)].add(inst.category)

    # Collect all categories
    all_categories = set()
    for cats in grid_map.values():
        all_categories.update(cats)

    # Draw grid
    for row in range(grid_divisions):
        for col in range(grid_divisions):
            cell_x = scene_min_x + col * cell_width
            cell_z = scene_min_z + row * cell_height

            categories = grid_map.get((row, col), set())

            if categories:
                # Mix colors for multiple categories
                if len(categories) == 1:
                    cat = list(categories)[0]
                    color = get_color_for_category(cat)
                    alpha = 0.6
                else:
                    # Blend colors for multiple categories
                    colors = [to_rgba(get_color_for_category(cat)) for cat in categories]
                    avg_color = np.mean(colors, axis=0)
                    color = avg_color
                    alpha = 0.7

                rect = patches.Rectangle(
                    (cell_x, cell_z),
                    cell_width,
                    cell_height,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor='#333333',
                    linewidth=0.8
                )
                ax.add_patch(rect)

                # Add category labels in cell (abbreviated if multiple)
                if len(categories) <= 2:
                    label = '\n'.join([c[:6] for c in sorted(categories)])
                else:
                    label = f"{len(categories)} types"

                ax.text(cell_x + cell_width / 2, cell_z + cell_height / 2,
                        label, fontsize=6, ha='center', va='center',
                        fontweight='bold', color='#222222')
            else:
                # Empty cell
                rect = patches.Rectangle(
                    (cell_x, cell_z),
                    cell_width,
                    cell_height,
                    facecolor='#F8F8F8',
                    alpha=0.3,
                    edgecolor='#CCCCCC',
                    linewidth=0.5
                )
                ax.add_patch(rect)

    # Draw grid coordinates on edges
    for i in range(grid_divisions):
        # Column labels (top)
        ax.text(scene_min_x + (i + 0.5) * cell_width, scene_max_z + 0.1,
                str(i), fontsize=8, ha='center', va='bottom', color='#666666')
        # Row labels (left)
        ax.text(scene_min_x - 0.1, scene_min_z + (i + 0.5) * cell_height,
                str(i), fontsize=8, ha='right', va='center', color='#666666')

    # Styling
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (meters)', fontsize=12, fontweight='bold')

    padding = max(cell_width, cell_height) * 0.5
    ax.set_xlim(scene_min_x - padding, scene_max_x + padding)
    ax.set_ylim(scene_min_z - padding, scene_max_z + padding)

    # Title
    scene = ast_data.get('Map_Metadata', {}).get('Scene', 'Unknown')
    total = ast_data.get('Map_Metadata', {}).get('Total_Instances', 0)
    occupied = ast_data.get('Map_Metadata', {}).get('Occupied_Cells', len(grid_map))
    ax.set_title(f"Cognitive Map (Grid {grid_divisions}x{grid_divisions}): {scene}\n({total} instances, {occupied} occupied cells)",
              fontsize=14, pad=20, fontweight='bold')

    # Legend
    legend_elements = [patches.Patch(facecolor=get_color_for_category(cat),
                                     alpha=0.6, label=cat, edgecolor=get_color_for_category(cat))
                       for cat in sorted(all_categories)]
    cam_handle = _draw_camera_orientations(ax, camera_orientations, scene_bounds)
    if cam_handle:
        legend_elements.append(cam_handle)
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  title="Categories", frameon=True, fontsize=9, title_fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved AST visualization (grid) to {output_path}")


# =============================================================================
# Main entry points
# =============================================================================

def generate_cognitive_ast(
    output_dir: str,
    ast_filename: str = "cognitive_ast.yaml",
    min_voxels: int = 30,
    merge_dist: float = 0.4,
    eps: float = 0.15,
    min_samples: int = 10,
    save_visualization: bool = True,
    ast_format: str = "ellipse",
    grid_divisions: int = 10,
    exclude_categories: List[str] = None,
    use_core_extraction: bool = True,
    core_percentile: float = 70.0,
    use_downsampling: bool = True,
    downsample_voxel_size: float = 0.05,
    large_class_threshold: int = 10000,
    camera_orientations: list = None,
    use_gpu: bool = True,
    memory_cache=None,   # SceneMemoryCache | None
) -> bool:
    """
    Generate cognitive AST from mapping output.
    Automatically detects mode (direct or voxel) based on available files.

    Args:
        output_dir: Directory containing mapping output files
        ast_filename: Output AST filename
        min_voxels: Minimum voxels/points for valid instance
        merge_dist: Distance threshold for merging instances
        eps: DBSCAN epsilon for direct mode
        min_samples: DBSCAN min_samples for direct mode
        save_visualization: Whether to save visualization image
        ast_format: AST format - "ellipse", "rectangle", "grid", or "all"
        grid_divisions: Number of grid divisions for grid format (NxN)
        exclude_categories: List of category names to exclude (e.g., traversable categories)
        use_core_extraction: Whether to extract core region to remove low-density tails
        core_percentile: Percentile for core extraction (higher = more aggressive filtering)
        use_downsampling: Enable voxel downsampling for large point clouds (memory optimization)
        downsample_voxel_size: Voxel size for downsampling large classes (meters)
        large_class_threshold: Point count threshold to trigger downsampling

    Returns:
        True if successful
    """
    mapping_dir = Path(output_dir)

    if not mapping_dir.exists():
        logger.error(f"Mapping directory not found: {mapping_dir}")
        return False

    # Validate ast_format
    valid_formats = ["ellipse", "rectangle", "grid", "all"]
    if ast_format not in valid_formats:
        logger.warning(f"Invalid ast_format '{ast_format}', using 'ellipse'")
        ast_format = "ellipse"

    # Normalize exclude categories to lowercase
    if exclude_categories:
        exclude_categories = [c.lower() for c in exclude_categories]
    else:
        exclude_categories = []

    # Check for point cloud data (direct mode)
    point_cloud_path = mapping_dir / "point_cloud.npy"
    point_cloud_meta_path = mapping_dir / "point_cloud_meta.npy"

    # Check for voxel grid data
    voxel_grid_path = mapping_dir / "voxel_grid.npy"
    voxel_grid_meta_path = mapping_dir / "voxel_grid_meta.npy"

    # Get scene name from parent directory
    scene_name = mapping_dir.parent.name

    instances = []
    origin = None
    class_names = []
    scene_bounds = None  # Full scene bounds from point cloud

    # Mode A: in-memory point cloud (from SceneMemoryCache)
    if memory_cache is not None and memory_cache.has_pointcloud():
        logger.info("Using direct mode (point cloud from memory cache)")

        points     = memory_cache.pc_points
        labels     = memory_cache.pc_labels
        class_names = memory_cache.pc_class_names or []
        scene_bounds = memory_cache.pc_scene_bounds

        if scene_bounds:
            logger.info(f"Scene bounds: X=[{scene_bounds['min_x']:.2f}, {scene_bounds['max_x']:.2f}], "
                       f"Z=[{scene_bounds['min_z']:.2f}, {scene_bounds['max_z']:.2f}]")

        logger.info(f"Loaded point cloud from cache: {len(points)} points, {len(class_names)} classes")

        instances = extract_instances_from_point_cloud(
            points, labels, class_names,
            eps=eps, min_samples=min_samples, min_points=min_voxels,
            use_core_extraction=use_core_extraction, core_percentile=core_percentile,
            use_downsampling=use_downsampling,
            downsample_voxel_size=downsample_voxel_size,
            large_class_threshold=large_class_threshold,
            use_gpu=use_gpu,
        )

        if len(points) > 0:
            origin = np.min(points, axis=0)

    # Mode B: disk point cloud
    elif point_cloud_path.exists() and point_cloud_meta_path.exists():
        logger.info("Using direct mode (point cloud data)")

        pc_data = np.load(point_cloud_path, allow_pickle=True).item()
        pc_meta = np.load(point_cloud_meta_path, allow_pickle=True).item()

        points = pc_data['points']
        labels = pc_data['labels']
        class_names = pc_meta['class_names']

        # Get scene bounds if available
        if 'scene_bounds' in pc_meta:
            scene_bounds = pc_meta['scene_bounds']
            logger.info(f"Scene bounds: X=[{scene_bounds['min_x']:.2f}, {scene_bounds['max_x']:.2f}], "
                       f"Z=[{scene_bounds['min_z']:.2f}, {scene_bounds['max_z']:.2f}]")

        logger.info(f"Loaded point cloud: {len(points)} points, {len(class_names)} classes")

        # Extract instances using DBSCAN
        instances = extract_instances_from_point_cloud(
            points, labels, class_names,
            eps=eps, min_samples=min_samples, min_points=min_voxels,
            use_core_extraction=use_core_extraction, core_percentile=core_percentile,
            use_downsampling=use_downsampling,
            downsample_voxel_size=downsample_voxel_size,
            large_class_threshold=large_class_threshold,
            use_gpu=use_gpu,
        )

        # Calculate origin from points
        if len(points) > 0:
            origin = np.min(points, axis=0)

    # Fallback to voxel mode
    elif voxel_grid_path.exists() and voxel_grid_meta_path.exists():
        logger.info("Using voxel mode (voxel grid data)")

        grid = np.load(voxel_grid_path)
        meta = np.load(voxel_grid_meta_path, allow_pickle=True).item()

        class_names = meta['class_names']
        origin = meta['origin']

        # Get scene bounds if available
        if 'scene_bounds' in meta:
            scene_bounds = meta['scene_bounds']

        logger.info(f"Loaded voxel grid: shape {grid.shape}")

        # Extract instances from voxel grid
        instances = extract_instances_from_voxel_grid(grid, meta, min_voxels=min_voxels)

    else:
        logger.error("No valid mapping data found (neither point_cloud.npy nor voxel_grid.npy)")
        return False

    # Filter out excluded categories (e.g., traversable categories like floor, ground, road)
    if exclude_categories and instances:
        original_count = len(instances)
        instances = [
            inst for inst in instances
            if inst.category.lower() not in exclude_categories
        ]
        filtered_count = original_count - len(instances)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} instances from excluded categories: {exclude_categories}")

    if not instances:
        logger.warning("No instances extracted")
        # Still save empty AST
        ast_data = {
            "Map_Metadata": {
                "Scene": scene_name,
                "Total_Instances": 0,
                "Categories": class_names
            },
            "Spatial_Hierarchy": []
        }
        ast_path = mapping_dir / ast_filename
        with open(ast_path, 'w', encoding='utf-8') as f:
            yaml.dump(ast_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        return True

    # Merge nearby instances
    instances = merge_nearby_instances(instances, merge_dist=merge_dist)

    # Infer hierarchy
    logger.info("Inferring spatial hierarchy...")
    roots = infer_spatial_hierarchy(instances)
    logger.info(f"Found {len(roots)} root instances")

    # Determine which formats to generate
    if ast_format == "all":
        formats_to_generate = ["ellipse", "rectangle", "grid"]
    else:
        formats_to_generate = [ast_format]

    logger.info(f"Generating AST in format(s): {formats_to_generate}")

    # Generate AST for each format
    for fmt in formats_to_generate:
        # Determine filename
        if ast_format == "all":
            # Use format-specific filenames
            base_name = ast_filename.replace(".yaml", "")
            current_filename = f"{base_name}_{fmt}.yaml"
        else:
            current_filename = ast_filename

        # Build AST structure based on format
        if fmt == "grid":
            ast_data = build_grid_ast(instances, scene_name, class_names, origin, grid_divisions, scene_bounds, camera_orientations)
        else:
            ast_data = build_ast_structure(instances, roots, scene_name, class_names, origin, fmt, camera_orientations)

        # Save AST
        ast_path = mapping_dir / current_filename
        with open(ast_path, 'w', encoding='utf-8') as f:
            yaml.dump(ast_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        logger.info(f"AST ({fmt}) saved to {ast_path}")

        # Save visualization
        if save_visualization and instances:
            if ast_format == "all":
                vis_filename = f"ast_visualization_{fmt}.png"
            else:
                vis_filename = "ast_visualization.png"

            vis_path = mapping_dir / vis_filename
            try:
                if fmt == "ellipse":
                    draw_ast_layout(ast_data, str(vis_path), instances, scene_bounds,
                                    camera_orientations=camera_orientations)
                elif fmt == "rectangle":
                    draw_ast_layout_rectangle(ast_data, str(vis_path), instances, scene_bounds,
                                              camera_orientations=camera_orientations)
                elif fmt == "grid":
                    draw_grid_ast(ast_data, str(vis_path), instances, grid_divisions, scene_bounds,
                                  camera_orientations=camera_orientations)
            except Exception as e:
                logger.warning(f"Failed to save visualization ({fmt}): {e}")
                import traceback
                traceback.print_exc()

    return True


def run_ast_generation(
    output_dir: str,
    ast_filename: str = "cognitive_ast.yaml",
    min_voxels: int = 30,
    merge_dist: float = 0.4,
    eps: float = 0.15,
    min_samples: int = 10,
    save_visualization: bool = True,
    ast_format: str = "ellipse",
    grid_divisions: int = 10,
    exclude_categories: List[str] = None,
    use_core_extraction: bool = True,
    core_percentile: float = 70.0,
    use_downsampling: bool = True,
    downsample_voxel_size: float = 0.05,
    large_class_threshold: int = 10000,
    camera_orientations: list = None,
    use_gpu: bool = True,
    memory_cache=None,   # SceneMemoryCache | None
) -> bool:
    """
    Run AST generation pipeline.

    Args:
        output_dir: Directory containing mapping output files
        ast_filename: Output AST filename
        min_voxels: Minimum voxels/points for valid instance
        merge_dist: Distance threshold for merging instances
        eps: DBSCAN epsilon for direct mode
        min_samples: DBSCAN min_samples for direct mode
        save_visualization: Whether to save visualization image
        ast_format: AST format - "ellipse", "rectangle", "grid", or "all"
        grid_divisions: Number of grid divisions for grid format (NxN)
        exclude_categories: List of category names to exclude (e.g., traversable categories)
        use_core_extraction: Whether to extract core region to remove low-density tails
        core_percentile: Percentile for core extraction (higher = more aggressive filtering)
        use_downsampling: Enable voxel downsampling for large point clouds (memory optimization)
        downsample_voxel_size: Voxel size for downsampling large classes (meters)
        large_class_threshold: Point count threshold to trigger downsampling
        memory_cache: SceneMemoryCache instance for in-memory pipeline (optional)

    Returns:
        True if successful
    """
    try:
        return generate_cognitive_ast(
            output_dir=output_dir,
            ast_filename=ast_filename,
            min_voxels=min_voxels,
            merge_dist=merge_dist,
            eps=eps,
            min_samples=min_samples,
            save_visualization=save_visualization,
            ast_format=ast_format,
            grid_divisions=grid_divisions,
            exclude_categories=exclude_categories,
            use_core_extraction=use_core_extraction,
            core_percentile=core_percentile,
            use_downsampling=use_downsampling,
            downsample_voxel_size=downsample_voxel_size,
            large_class_threshold=large_class_threshold,
            camera_orientations=camera_orientations,
            use_gpu=use_gpu,
            memory_cache=memory_cache,
        )
    except Exception as e:
        logger.error(f"AST generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
