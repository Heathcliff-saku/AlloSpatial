"""
Spatial Intelligence Tools for LLM Function Calling.

Uses HTTP model service for the full cognitive map pipeline
(depth → segmentation → mapping → AST → route).
"""

import os
import sys
import json
import yaml
import base64
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import (
    PipelineConfig, INDOOR_TRAVERSABLE_CATEGORIES, OUTDOOR_TRAVERSABLE_CATEGORIES
)
from scripts.frame_extraction import extract_frames
from services.client import ModelServiceClient


@dataclass
class CognitiveMapResult:
    """Result from cognitive map generation."""
    success: bool
    knowledge_type: str
    scene_id: str
    landmark_yaml: Optional[str] = None
    landmark_yaml_path: Optional[str] = None
    landmark_visualization_path: Optional[str] = None
    landmark_visualization_base64: Optional[str] = None
    route_yaml: Optional[str] = None
    route_yaml_path: Optional[str] = None
    route_visualization_path: Optional[str] = None
    route_visualization_base64: Optional[str] = None
    pointcloud_rgb_vis_path: Optional[str] = None
    pointcloud_semantic_vis_path: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def _available_visualizations(self) -> Dict[str, str]:
        """Collect all available visualization file paths."""
        vis = {}
        for key, path in [
            ("landmark_vis", self.landmark_visualization_path),
            ("route_vis", self.route_visualization_path),
            ("pointcloud_rgb_topdown", self.pointcloud_rgb_vis_path),
            ("pointcloud_semantic_topdown", self.pointcloud_semantic_vis_path),
        ]:
            if path and os.path.exists(path):
                vis[key] = path
        return vis

    def to_json(self) -> str:
        d = {}
        d["success"] = self.success
        d["knowledge_type"] = self.knowledge_type
        d["scene_id"] = self.scene_id
        if self.error_message:
            d["error_message"] = self.error_message
        if self.landmark_yaml:
            d["landmark_yaml"] = self.landmark_yaml
        if self.route_yaml:
            d["route_yaml"] = self.route_yaml
        # Only output available visualization type names (no paths)
        vis_types = [k for k, path in [
            ("landmark_vis", self.landmark_visualization_path),
            ("route_vis", self.route_visualization_path),
            ("pointcloud_rgb_topdown", self.pointcloud_rgb_vis_path),
            ("pointcloud_semantic_topdown", self.pointcloud_semantic_vis_path),
        ] if path and os.path.exists(path)]
        if vis_types:
            d["available_visualizations"] = vis_types
        return json.dumps(d, indent=2, ensure_ascii=False)


class SpatialIntelligenceTools:
    """
    Spatial Intelligence Tools for generating cognitive maps.

    Calls pipeline functions directly in-process and uses HTTP model service
    Calls the model service HTTP endpoint for the full cognitive map pipeline.
    Supports video, image list, and single image inputs.

    Input paths are bound at construction time. The LLM does not need to know
    file paths — it only provides semantic parameters (categories, scene_type, etc.).
    Workspace is lazily initialized on first tool call.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_base: Optional[str] = None,
        service_url: str = "http://localhost:9100",
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        service_client: Optional[object] = None,
    ):
        # Load config
        if config_path is None:
            config_path = str(_PROJECT_ROOT / "config" / "default_config.yaml")
        self.config_path = str(config_path)
        self.config = PipelineConfig.from_yaml(self.config_path)

        if output_base:
            self.config.output_base = output_base
        self.output_base = Path(self.config.output_base).resolve()
        self.output_base.mkdir(parents=True, exist_ok=True)

        # HF mirror
        if self.config.hf_endpoint:
            os.environ['HF_ENDPOINT'] = self.config.hf_endpoint

        # Model service client (external client takes precedence, e.g. multi-port)
        if service_client is not None:
            self.service_client = service_client
        else:
            self.service_client = ModelServiceClient(
                service_url=service_url,
                timeout=self.config.service.timeout,
            )

        # Bound input paths (set at construction, used by all tool calls)
        self.video_path = video_path
        self.image_paths = image_paths

        # Cache for visualization paths from the last world2mind call
        # Maps visualization_type -> file path
        self._last_visualizations: Dict[str, str] = {}

    # ============================================================
    # Input Preprocessing
    # ============================================================

    def _prepare_workspace(self, scene_id: str) -> Path:
        """Create workspace directory for a task."""
        workspace = self.output_base / scene_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _prepare_input(
        self,
        workspace: Path,
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Path:
        """
        Prepare input images in workspace.

        For video: extract frames into workspace/extract_frame/
        For image list: symlink images into workspace/extract_frame/
        For single image: symlink into workspace/extract_frame/

        Returns:
            Path to the frames directory
        """
        frames_dir = workspace / self.config.frame_extraction.output_folder
        fmt = self.config.frame_extraction.image_format

        # Already have frames?
        if frames_dir.exists() and len(list(frames_dir.glob(f"*.{fmt}"))) > 0:
            return frames_dir

        frames_dir.mkdir(parents=True, exist_ok=True)

        if video_path:
            # Video → extract frames
            extract_frames(
                video_path,
                frames_dir,
                fps=self.config.frame_extraction.fps,
                max_frames=self.config.frame_extraction.max_frames,
                image_quality=self.config.frame_extraction.image_quality,
                image_format=fmt,
            )
        elif image_paths:
            # Image list → symlink/copy into workspace
            for i, img_path in enumerate(sorted(image_paths)):
                src = Path(img_path).resolve()
                if not src.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue
                ext = src.suffix
                dst = frames_dir / f"frame_{i:06d}{ext}"
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except OSError:
                        shutil.copy2(str(src), str(dst))

        num_frames = len(list(frames_dir.iterdir()))
        logger.info(f"Prepared {num_frames} frames in {frames_dir}")
        return frames_dir

    # ============================================================
    # Pipeline Execution
    # ============================================================

    def _run_pipeline(
        self,
        frames_dir: Path,
        workspace: Path,
        categories: List[str],
        scene_type: str = "outdoor",
        run_landmark: bool = True,
        run_route: bool = True,
        output_format: str = "grid",
        traversable_categories: Optional[List[str]] = None,
        scene_id: str = "",
    ) -> Dict[str, Any]:
        """Run the full pipeline via the model service HTTP endpoint."""
        return self.service_client.run_cognitive_map(
            image_dir=str(frames_dir),
            workspace=str(workspace),
            categories=categories,
            scene_type=scene_type,
            run_landmark=run_landmark,
            run_route=run_route,
            output_format=output_format,
            traversable_categories=traversable_categories or [],
            config_path=self.config_path,
            scene_id=scene_id,
        )

    # ============================================================
    # Public API
    # ============================================================

    def generate_cognitive_map(
        self,
        categories: List[str] = None,
        scene_type: str = "outdoor",
        knowledge_type: str = "both",
        output_format: str = "grid",
        traversable_categories: Optional[List[str]] = None,
        include_visualization: bool = True,
        request_id: Optional[str] = None,
    ) -> CognitiveMapResult:
        """
        Generate cognitive map from the bound video/images.

        Input paths are taken from self.video_path / self.image_paths
        (set at construction time). The LLM only provides semantic parameters.

        Args:
            categories: Object categories to detect
            scene_type: "indoor" or "outdoor"
            knowledge_type: "landmark", "route", or "both"
            output_format: "grid", "rectangle", or "ellipse"
            traversable_categories: Ground/surface categories for route knowledge
            include_visualization: Include base64 visualizations in result
            request_id: Optional unique ID (auto-generated if None)
        """
        video_path = self.video_path
        image_paths = self.image_paths

        if not video_path and not image_paths:
            return CognitiveMapResult(
                success=False, knowledge_type=knowledge_type, scene_id="",
                error_message="Must provide either video_path or image_paths"
            )

        # Generate scene ID
        if video_path:
            name_prefix = Path(video_path).stem[:20]
        else:
            name_prefix = f"images_{len(image_paths)}"
        uid = request_id or str(uuid.uuid4())[:12]
        scene_id = f"{name_prefix}_{uid}"

        run_landmark = knowledge_type in ["landmark", "both"]
        run_route = knowledge_type in ["route", "both"]

        # Default traversable categories
        if run_route and not traversable_categories:
            if scene_type == "indoor":
                traversable_categories = list(INDOOR_TRAVERSABLE_CATEGORIES)
            else:
                traversable_categories = list(OUTDOOR_TRAVERSABLE_CATEGORIES)

        try:
            # Prepare workspace and input
            workspace = self._prepare_workspace(scene_id)
            frames_dir = self._prepare_input(
                workspace, video_path=video_path, image_paths=image_paths
            )

            # Run pipeline
            result = self._run_pipeline(
                frames_dir=frames_dir,
                workspace=workspace,
                categories=categories,
                scene_type=scene_type,
                run_landmark=run_landmark,
                run_route=run_route,
                output_format=output_format,
                traversable_categories=traversable_categories,
                scene_id=scene_id,
            )
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return CognitiveMapResult(
                success=False, knowledge_type=knowledge_type, scene_id=scene_id,
                error_message=str(e)
            )

        if not result.get("success"):
            return CognitiveMapResult(
                success=False, knowledge_type=knowledge_type, scene_id=scene_id,
                error_message=result.get("error", "Unknown error")
            )

        # Read outputs
        landmark_yaml = None
        landmark_vis_b64 = None
        route_yaml = None
        route_vis_b64 = None

        if run_landmark:
            landmark_yaml = _read_file_text(result.get("landmark_yaml_path"))
            if include_visualization:
                landmark_vis_b64 = _read_file_base64(result.get("landmark_visualization_path"))

        if run_route:
            route_yaml = _read_file_text(result.get("route_yaml_path"))
            if include_visualization:
                route_vis_b64 = _read_file_base64(result.get("route_visualization_path"))

        return CognitiveMapResult(
            success=True,
            knowledge_type=knowledge_type,
            scene_id=scene_id,
            landmark_yaml=landmark_yaml,
            landmark_yaml_path=result.get("landmark_yaml_path") if run_landmark else None,
            landmark_visualization_path=result.get("landmark_visualization_path") if run_landmark else None,
            landmark_visualization_base64=landmark_vis_b64,
            route_yaml=route_yaml,
            route_yaml_path=result.get("route_yaml_path") if run_route else None,
            route_visualization_path=result.get("route_visualization_path") if run_route else None,
            route_visualization_base64=route_vis_b64,
            pointcloud_rgb_vis_path=result.get("pointcloud_rgb_vis_path"),
            pointcloud_semantic_vis_path=result.get("pointcloud_semantic_vis_path"),
        )


# ============================================================
# Tool Call Handler
# ============================================================

def handle_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    tools: SpatialIntelligenceTools,
    request_id: Optional[str] = None,
) -> Any:
    """
    Handle a tool call from the LLM.

    Errors are caught and returned as JSON error messages so the model
    can see what went wrong and retry or adjust its approach.

    Returns:
        For world2mind: JSON string with tool result
        For view_image: dict with {"type": "image", "base64": ...} for image injection
        On error: JSON string with {"error": "..."} describing the problem
    """
    try:
        if tool_name == "world2mind":
            # Validate required parameters
            if "categories" not in arguments or not arguments["categories"]:
                return json.dumps({
                    "error": "Missing required parameter 'categories'. "
                    "Please provide a list of object categories to detect, "
                    "e.g. [\"table\", \"chair\", \"door\"]."
                })
            result = tools.generate_cognitive_map(
                categories=arguments["categories"],
                scene_type=arguments.get("scene_type", "outdoor"),
                knowledge_type=arguments.get("knowledge_type", "both"),
                output_format=arguments.get("output_format", "rectangle"),
                traversable_categories=arguments.get("traversable_categories"),
                request_id=request_id,
            )
            # Cache visualization paths for subsequent view_image calls
            tools._last_visualizations = result._available_visualizations()
            return result.to_json()
        elif tool_name == "view_image":
            vis_type = arguments.get("visualization_type", "")
            if not vis_type:
                available = list(tools._last_visualizations.keys()) if tools._last_visualizations else []
                return json.dumps({
                    "error": "Missing required parameter 'visualization_type'. "
                    f"Available types: {available}"
                })
            return handle_view_image(vis_type, tools._last_visualizations)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}. Available tools: world2mind, view_image"})
    except Exception as e:
        logger.error(f"Tool '{tool_name}' execution failed: {e}")
        return json.dumps({"error": f"Tool execution failed: {e}"})


def handle_view_image(visualization_type: str, vis_cache: Dict[str, str]) -> Dict[str, Any]:
    """
    Look up a visualization by type and return image data for conversation injection.

    Args:
        visualization_type: One of landmark_vis, route_vis, pointcloud_rgb_topdown, pointcloud_semantic_topdown
        vis_cache: Mapping from type name to file path (from last world2mind call)

    Returns:
        Dict with type="image", base64 data, and visualization_type.
        The caller (demo script) should convert this into the appropriate image message format.
    """
    image_path = vis_cache.get(visualization_type)
    if not image_path or not os.path.exists(image_path):
        return {"type": "text", "content": json.dumps({
            "error": f"Visualization '{visualization_type}' not available. "
                     f"Available: {list(vis_cache.keys())}"
        })}

    b64 = _read_file_base64(image_path)
    if b64 is None:
        return {"type": "text", "content": json.dumps({"error": f"Failed to read visualization: {visualization_type}"})}

    return {
        "type": "image",
        "base64": b64,
        "visualization_type": visualization_type,
    }

# ============================================================
# Helpers
# ============================================================

def _read_file_text(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def _read_file_base64(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None
