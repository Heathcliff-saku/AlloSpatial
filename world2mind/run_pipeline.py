"""
CLI entry point for running the world2mind pipeline.

Supports video, image list, and single image inputs.
Uses the HTTP model service for DA3/SAM3 inference.

Usage:
    # Video input
    python run_pipeline.py --video_path /path/to/video.mp4 --categories "car,building,tree"

    # Image list input
    python run_pipeline.py --image_paths "/path/a.jpg,/path/b.jpg" --categories "car,building"

    # Single image
    python run_pipeline.py --video_path /path/to/image.jpg --categories "chair,table"
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.spatial_tools import SpatialIntelligenceTools

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="World2Mind Pipeline - Generate cognitive maps from video/images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video
  python run_pipeline.py --video_path /path/to/video.mp4 --categories "car,building,tree"

  # Process image list
  python run_pipeline.py --image_paths "/path/a.jpg,/path/b.jpg" --categories "car,building"

  # Indoor scene with route knowledge
  python run_pipeline.py --video_path /path/to/video.mp4 --categories "chair,table" \\
      --scene_type indoor --knowledge_type both

  # Landmark only with ellipse format
  python run_pipeline.py --video_path /path/to/video.mp4 --categories "car,building" \\
      --knowledge_type landmark --output_format ellipse
        """
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Path to video or single image file")
    input_group.add_argument("--image_paths", type=str, help="Comma-separated list of image paths")

    # Required
    parser.add_argument("--categories", type=str, required=True,
                        help="Comma-separated object categories to detect")

    # Pipeline options
    parser.add_argument("--scene_type", choices=["indoor", "outdoor"], default="outdoor")
    parser.add_argument("--knowledge_type", choices=["landmark", "route", "both"], default="both")
    parser.add_argument("--output_format", choices=["grid", "rectangle", "ellipse", "all"], default="grid")
    parser.add_argument("--traversable_categories", type=str, default=None,
                        help="Comma-separated traversable categories (auto-detected if not set)")

    # Service
    parser.add_argument("--service_url", type=str, default="http://localhost:9100",
                        help="Model service URL")

    # Config
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--output_base", type=str, default=None, help="Override output base directory")

    args = parser.parse_args()

    # Parse categories
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        print("Error: No categories provided")
        sys.exit(1)

    # Parse traversable categories
    traversable_cats = None
    if args.traversable_categories:
        traversable_cats = [c.strip() for c in args.traversable_categories.split(",") if c.strip()]

    # Parse image paths
    video_path = args.video_path
    image_paths = None
    if args.image_paths:
        image_paths = [p.strip() for p in args.image_paths.split(",") if p.strip()]
        # Validate
        for p in image_paths:
            if not os.path.exists(p):
                print(f"Error: Image not found: {p}")
                sys.exit(1)

    # Validate video path
    if video_path and not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    # Initialize tools
    tools = SpatialIntelligenceTools(
        config_path=args.config,
        output_base=args.output_base,
        service_url=args.service_url,
        video_path=video_path,
        image_paths=image_paths,
    )

    # Run
    logger.info("Starting pipeline...")
    result = tools.generate_cognitive_map(
        categories=categories,
        scene_type=args.scene_type,
        knowledge_type=args.knowledge_type,
        output_format=args.output_format,
        traversable_categories=traversable_cats,
        include_visualization=False,
    )

    if result.success:
        logger.info(f"Success! Scene ID: {result.scene_id}")
        if result.landmark_yaml_path:
            logger.info(f"  Landmark: {result.landmark_yaml_path}")
        if result.route_yaml_path:
            logger.info(f"  Route: {result.route_yaml_path}")
        if result.landmark_yaml:
            print("\n--- Landmark Knowledge (first 1000 chars) ---")
            print(result.landmark_yaml[:1000])
        if result.route_yaml:
            print("\n--- Route Knowledge ---")
            print(result.route_yaml)
    else:
        logger.error(f"Failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
