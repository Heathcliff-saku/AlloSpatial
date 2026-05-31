"""
Configuration module for world2mind pipeline.
Refactored from original raptor project with added ServiceConfig.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class ServiceConfig:
    """Model service configuration."""
    host: str = "0.0.0.0"
    port: int = 8100
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    timeout: int = 1800  # seconds
    workers_per_gpu: int = 1
    max_cpu_concurrent: int = 2  # Max concurrent CPU-phase (mapping/AST/route) requests


@dataclass
class FrameExtractionConfig:
    """Frame extraction configuration."""
    fps: int = 5
    max_frames: int = 150
    output_folder: str = "extract_frame"
    image_quality: int = 2
    image_format: str = "jpg"


@dataclass
class DynamicMaskConfig:
    """Dynamic mask configuration for excluding moving objects."""
    enable: bool = False
    annot_base: str = ""
    dyn_pixel_threshold: float = 0.01
    indexes_filename: str = "indexes.txt"
    dyn_masks_filename: str = "dyn_masks.npz"


@dataclass
class DA3Config:
    """Depth Anything 3 configuration."""
    enable: bool = True
    model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    output_folder: str = "da3_output"
    ref_view_strategy: str = "saddle_balanced"
    use_ray_pose: bool = True
    save_rgb: bool = False
    conf_threshold: float = 1.1
    scene_conf_threshold: float = 2.0


@dataclass
class SAM3Config:
    """SAM3 semantic segmentation configuration."""
    enable: bool = True
    model_path: str = "sam3.pt"
    conf: float = 0.65
    output_folder: str = "sam_output"
    save_vis: bool = False
    save_vis_video: bool = False
    video_fps: int = 10
    half_precision: bool = True
    seg_batch_size: int = 8  # Frames per encoder batch (>1 enables batched encoding).
                             # Set to 1 to fall back to the original frame-by-frame path.


@dataclass
class MappingConfig:
    """Point cloud mapping configuration."""
    enable: bool = True
    output_folder: str = "cognitive_map_output"
    mode: str = "direct"
    grid_size: float = 0.05
    ceiling_height: Optional[float] = None
    min_height: Optional[float] = None
    min_depth: float = 0.1
    pixel_conf_threshold: float = 1.1
    point_skip: int = 10
    denoise_k: int = 10
    min_points_per_voxel: int = 3
    flip_y: bool = False
    save_grid_npy: bool = False
    eps: float = 0.15
    min_samples: int = 10
    use_downsampling: bool = True
    downsample_voxel_size: float = 0.05
    large_class_threshold: int = 10000
    merge_dist: float = 0.4
    use_gpu: bool = True  # GPU-accelerated back-projection (mapping) + cuML DBSCAN (AST)
    min_voxels: int = 30
    ast_filename: str = "cognitive_ast.yaml"
    ast_format: str = "ellipse"
    grid_divisions: int = 10
    use_core_extraction: bool = True
    core_percentile: float = 70.0


@dataclass
class RouteKnowledgeConfig:
    """Route knowledge cognitive map configuration."""
    enable: bool = True
    output_folder: str = "cognitive_map_output"
    traversable_categories: List[str] = field(default_factory=list)
    grid_divisions: int = 10
    min_points_per_cell: int = 5
    point_skip: int = 10
    voxel_size: float = 0.1
    simplify_trajectory: bool = True
    save_visualization: bool = True
    trajectory_color: str = "#FF4444"
    traversable_color: str = "#44AA44"


# Pre-defined traversable categories for different scene types
INDOOR_TRAVERSABLE_CATEGORIES = [
    "floor",
    "carpet",
    "tile flooring"
]

OUTDOOR_TRAVERSABLE_CATEGORIES = [
    "road",
    "sidewalk",
    "path",
    "pavement",
]


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    output_base: str = "./workspace"
    hf_endpoint: str = "https://hf-mirror.com"
    scene_type: str = "outdoor"
    max_depth_indoor: float = 10.0
    max_depth_outdoor: float = 30.0
    ceiling_height_offset: float = 0.5
    camera_orientation_max_frames: int = 8
    keep_extracted_frames: bool = True
    keep_da3_output: bool = True
    keep_sam3_output: bool = True
    use_memory_cache: bool = True  # Keep DA3/SAM3/mapping intermediate data in RAM (no disk I/O)
    categories: List[str] = field(default_factory=list)

    # Sub-configurations
    service: ServiceConfig = field(default_factory=ServiceConfig)
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    dynamic_mask: DynamicMaskConfig = field(default_factory=DynamicMaskConfig)
    da3: DA3Config = field(default_factory=DA3Config)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    route_knowledge: RouteKnowledgeConfig = field(default_factory=RouteKnowledgeConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Top-level fields
        for key in [
            'output_base', 'hf_endpoint', 'scene_type',
            'max_depth_indoor', 'max_depth_outdoor', 'ceiling_height_offset',
            'camera_orientation_max_frames',
            'keep_extracted_frames', 'keep_da3_output', 'keep_sam3_output',
            'use_memory_cache',
            'categories',
        ]:
            if key in data:
                setattr(config, key, data[key])

        # Sub-config sections
        sub_configs = {
            'service': config.service,
            'frame_extraction': config.frame_extraction,
            'dynamic_mask': config.dynamic_mask,
            'da3': config.da3,
            'sam3': config.sam3,
            'mapping': config.mapping,
            'route_knowledge': config.route_knowledge,
        }
        for section_name, section_obj in sub_configs.items():
            if section_name in data and isinstance(data[section_name], dict):
                for key, value in data[section_name].items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

        config._auto_configure_categories()
        return config

    def _auto_configure_categories(self):
        """Auto-configure traversable categories based on scene_type."""
        if not self.route_knowledge.traversable_categories:
            if self.scene_type == "indoor":
                self.route_knowledge.traversable_categories = INDOOR_TRAVERSABLE_CATEGORIES.copy()
            else:
                self.route_knowledge.traversable_categories = OUTDOOR_TRAVERSABLE_CATEGORIES.copy()

        if self.route_knowledge.enable:
            for cat in self.route_knowledge.traversable_categories:
                if cat not in self.categories:
                    self.categories.append(cat)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'output_base': self.output_base,
            'hf_endpoint': self.hf_endpoint,
            'scene_type': self.scene_type,
            'max_depth_indoor': self.max_depth_indoor,
            'max_depth_outdoor': self.max_depth_outdoor,
            'ceiling_height_offset': self.ceiling_height_offset,
            'camera_orientation_max_frames': self.camera_orientation_max_frames,
            'keep_extracted_frames': self.keep_extracted_frames,
            'keep_da3_output': self.keep_da3_output,
            'keep_sam3_output': self.keep_sam3_output,
            'use_memory_cache': self.use_memory_cache,
            'categories': self.categories,
            'service': {
                'host': self.service.host,
                'port': self.service.port,
                'gpu_ids': self.service.gpu_ids,
                'timeout': self.service.timeout,
                'workers_per_gpu': self.service.workers_per_gpu,
                'max_cpu_concurrent': self.service.max_cpu_concurrent,
            },
            'frame_extraction': {
                'fps': self.frame_extraction.fps,
                'max_frames': self.frame_extraction.max_frames,
                'output_folder': self.frame_extraction.output_folder,
                'image_quality': self.frame_extraction.image_quality,
                'image_format': self.frame_extraction.image_format,
            },
            'dynamic_mask': {
                'enable': self.dynamic_mask.enable,
                'annot_base': self.dynamic_mask.annot_base,
                'dyn_pixel_threshold': self.dynamic_mask.dyn_pixel_threshold,
                'indexes_filename': self.dynamic_mask.indexes_filename,
                'dyn_masks_filename': self.dynamic_mask.dyn_masks_filename,
            },
            'da3': {
                'enable': self.da3.enable,
                'model': self.da3.model,
                'output_folder': self.da3.output_folder,
                'ref_view_strategy': self.da3.ref_view_strategy,
                'use_ray_pose': self.da3.use_ray_pose,
                'save_rgb': self.da3.save_rgb,
                'conf_threshold': self.da3.conf_threshold,
                'scene_conf_threshold': self.da3.scene_conf_threshold,
            },
            'sam3': {
                'enable': self.sam3.enable,
                'model_path': self.sam3.model_path,
                'conf': self.sam3.conf,
                'output_folder': self.sam3.output_folder,
                'save_vis': self.sam3.save_vis,
                'save_vis_video': self.sam3.save_vis_video,
                'video_fps': self.sam3.video_fps,
                'half_precision': self.sam3.half_precision,
            },
            'mapping': {
                'enable': self.mapping.enable,
                'output_folder': self.mapping.output_folder,
                'mode': self.mapping.mode,
                'grid_size': self.mapping.grid_size,
                'ceiling_height': self.mapping.ceiling_height,
                'min_height': self.mapping.min_height,
                'min_depth': self.mapping.min_depth,
                'pixel_conf_threshold': self.mapping.pixel_conf_threshold,
                'denoise_k': self.mapping.denoise_k,
                'min_points_per_voxel': self.mapping.min_points_per_voxel,
                'point_skip': self.mapping.point_skip,
                'flip_y': self.mapping.flip_y,
                'save_grid_npy': self.mapping.save_grid_npy,
                'eps': self.mapping.eps,
                'min_samples': self.mapping.min_samples,
                'merge_dist': self.mapping.merge_dist,
                'min_voxels': self.mapping.min_voxels,
                'ast_filename': self.mapping.ast_filename,
                'ast_format': self.mapping.ast_format,
                'grid_divisions': self.mapping.grid_divisions,
                'use_core_extraction': self.mapping.use_core_extraction,
                'core_percentile': self.mapping.core_percentile,
                'use_downsampling': self.mapping.use_downsampling,
                'downsample_voxel_size': self.mapping.downsample_voxel_size,
                'large_class_threshold': self.mapping.large_class_threshold,
                'use_gpu': self.mapping.use_gpu,
            },
            'route_knowledge': {
                'enable': self.route_knowledge.enable,
                'output_folder': self.route_knowledge.output_folder,
                'traversable_categories': self.route_knowledge.traversable_categories,
                'grid_divisions': self.route_knowledge.grid_divisions,
                'min_points_per_cell': self.route_knowledge.min_points_per_cell,
                'point_skip': self.route_knowledge.point_skip,
                'voxel_size': self.route_knowledge.voxel_size,
                'simplify_trajectory': self.route_knowledge.simplify_trajectory,
                'save_visualization': self.route_knowledge.save_visualization,
                'trajectory_color': self.route_knowledge.trajectory_color,
                'traversable_color': self.route_knowledge.traversable_color,
            },
        }

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    config = PipelineConfig()
    config._auto_configure_categories()
    return config
