"""
OpenAI API Demo for World2Mind Spatial Intelligence Tools.

Supports both OpenAI API and vLLM local server via OpenAI-compatible API.
Uses the refactored SpatialIntelligenceTools (HTTP model service).

Usage:
    # Direct tool execution - video
    python demo/demo_openai.py --video /path/to/video.mp4 --direct --categories car building tree

    # Direct tool execution - image list
    python demo/demo_openai.py --image-paths /path/a.jpg /path/b.jpg --direct --categories car building

    # With OpenAI API - video
    python demo/demo_openai.py --video /path/to/video.mp4 --query "What objects are in this scene?"

    # With OpenAI API - image list
    python demo/demo_openai.py --image-paths /path/a.jpg /path/b.jpg --query "Describe the layout"

    # With vLLM local server
    python demo/demo_openai.py --video /path/to/video.mp4 --query "Describe the scene" \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from typing import List, Optional

import numpy as np
from PIL import Image
from openai import OpenAI

# Ensure project root is in path
# Locate the World2Mind tool package (the `world2mind/` dir of this repo).
_W2M_ROOT = os.environ.get("WORLD2MIND_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world2mind")
sys.path.insert(0, _W2M_ROOT)

from tools.tool_definitions import get_tool_definitions
from tools.spatial_tools import SpatialIntelligenceTools, handle_tool_call
from tools.prompts import OPENAI_SYSTEM_PROMPT, OPENAI_USER_PROMPT


def create_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    """Create OpenAI client, supporting both OpenAI API and vLLM local server."""
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def encode_image(image) -> str:
    """Encode image to base64."""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()
    output_buffer = BytesIO()
    img.save(output_buffer, format="PNG")
    return base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def encode_video(video_path: str, max_frame_num: int = 32, fps: float = 1.0) -> List[str]:
    """Encode video frames to base64."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = vr.get_avg_fps()

    if fps > 0:
        frame_interval = max(1, int(video_fps / fps))
        frame_idx = list(range(0, total_frame_num, frame_interval))
        if len(frame_idx) > max_frame_num:
            indices = np.linspace(0, len(frame_idx) - 1, max_frame_num, dtype=int)
            frame_idx = [frame_idx[i] for i in indices]
    else:
        frame_idx = np.linspace(0, total_frame_num - 1, max_frame_num, dtype=int).tolist()

    if total_frame_num - 1 not in frame_idx:
        frame_idx.append(total_frame_num - 1)

    frames = vr.get_batch(frame_idx).asnumpy()

    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="PNG")
        base64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return base64_frames


def encode_visual_input(
    video_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    max_frame_num: int = 32,
    fps: float = 1.0,
    verbose: bool = True,
) -> List[str]:
    """
    Encode visual input (video or image list) to base64 frames.

    Returns:
        List of base64-encoded PNG strings
    """
    if video_path:
        if video_path.endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv')):
            if verbose:
                print(f"Encoding video frames (max={max_frame_num}, fps={fps})...")
            imgs = encode_video(video_path, max_frame_num=max_frame_num, fps=fps)
            if verbose:
                print(f"Encoded {len(imgs)} frames")
            return imgs
        else:
            if verbose:
                print("Encoding single image...")
            return [encode_image(video_path)]
    elif image_paths:
        if verbose:
            print(f"Encoding {len(image_paths)} images...")
        return [encode_image(p) for p in image_paths]
    return []


def _make_image_content(b64_data: str, api_type: str = "openai") -> dict:
    """Create an image content block in the correct format for the API type."""
    if api_type == "claude":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_data}
        }
    else:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_data}"}
        }


def run_conversation(
    client: OpenAI,
    model: str,
    user_query: str,
    tools_instance: SpatialIntelligenceTools,
    video_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    max_turns: int = 5,
    max_frame_num: int = 32,
    fps: float = 1.0,
    api_type: str = "openai",
    verbose: bool = True,
) -> str:
    """Run a multi-turn conversation with tool calling."""
    # Encode visual input
    imgs = encode_visual_input(
        video_path=video_path, image_paths=image_paths,
        max_frame_num=max_frame_num, fps=fps, verbose=verbose,
    )

    # Build multimodal user content: images first, then reasoning instructions + query
    user_content = []
    for img in imgs:
        user_content.append(_make_image_content(img, api_type))
    user_content.append({"type": "text", "text": OPENAI_USER_PROMPT.format(query=user_query)})

    messages = [
        {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    tool_definitions = get_tool_definitions()

    for turn in range(max_turns):
        if verbose:
            print(f"\n{'='*50}\nTurn {turn + 1}\n{'='*50}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
            max_tokens=12800,
        )

        msg = response.choices[0].message

        if verbose and msg.content:
            print(f"\nAssistant:\n{msg.content}")

        if not msg.tool_calls:
            return msg.content or "No response"

        # Preserve the full assistant message (including vendor-specific fields
        # like Gemini's thought_signature) instead of manually reconstructing it.
        assistant_msg = msg.model_dump(exclude_none=True, exclude_unset=True)
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            arguments = json.loads(tc.function.arguments)

            if verbose:
                print(f"\nTool Call: {tool_name}")
                print(f"Arguments: {json.dumps(arguments, indent=2)}")

            tool_result = handle_tool_call(tool_name, arguments, tools_instance)

            # view_image returns a dict with image data for conversation injection
            if isinstance(tool_result, dict) and tool_result.get("type") == "image":
                vis_type = tool_result.get("visualization_type", "unknown")
                if verbose:
                    print(f"\nView Image: {vis_type}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": [
                        _make_image_content(tool_result["base64"], api_type),
                        {"type": "text", "text": f"Visualization: {vis_type}"}
                    ],
                })
            else:
                if isinstance(tool_result, dict):
                    tool_result = json.dumps(tool_result)

                if verbose:
                    preview = tool_result
                    print(f"\nTool Result (preview):\n{preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

    return "Max turns reached without final response"


def demo_without_api(
    video_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    categories: List[str] = None,
    scene_type: str = "outdoor",
    knowledge_type: str = "both",
    output_format: str = "grid",
    service_url: str = "http://localhost:8100",
    config_path: Optional[str] = None,
):
    """Run tools directly without LLM API (for testing)."""
    print("=" * 60)
    print("Direct Tool Execution Demo (No LLM API)")
    print("=" * 60)

    tools = SpatialIntelligenceTools(
        config_path=config_path, service_url=service_url,
        video_path=video_path, image_paths=image_paths,
    )

    input_desc = video_path or f"{len(image_paths)} images"
    print(f"\nInput: {input_desc}")
    print(f"Categories: {categories}")
    print(f"Scene type: {scene_type}")
    print(f"Knowledge type: {knowledge_type}")
    print(f"Output format: {output_format}")

    print("\nGenerating Cognitive Map...")
    result = tools.generate_cognitive_map(
        categories=categories,
        scene_type=scene_type,
        knowledge_type=knowledge_type,
        output_format=output_format,
        include_visualization=False,
    )

    if result.success:
        print(f"\nSuccess! Scene ID: {result.scene_id}")
        if result.landmark_yaml_path:
            print(f"  Landmark: {result.landmark_yaml_path}")
            if result.landmark_yaml:
                print(f"\nLandmark Knowledge (first 1000 chars):")
                print(result.landmark_yaml[:1000])
        if result.route_yaml_path:
            print(f"  Route: {result.route_yaml_path}")
            if result.route_yaml:
                print(f"\nRoute Knowledge:")
                print(result.route_yaml)
    else:
        print(f"\nFailed: {result.error_message}")


def demo_with_api(
    query: str,
    model: str = "gpt-4o",
    video_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_frame_num: int = 32,
    fps: float = 1.0,
    api_type: str = "openai",
    service_url: str = "http://localhost:8100",
    config_path: Optional[str] = None,
):
    """Run with OpenAI API and function calling."""
    print("=" * 60)
    print("OpenAI Function Calling Demo")
    print("=" * 60)
    print(f"\nModel: {model}")
    print(f"API type: {api_type}")
    input_desc = video_path or f"{len(image_paths)} images"
    print(f"Input: {input_desc}")
    print(f"Query: {query}")

    client = create_client(base_url=base_url, api_key=api_key)
    tools = SpatialIntelligenceTools(
        config_path=config_path, service_url=service_url,
        video_path=video_path, image_paths=image_paths,
    )

    print("\nStarting conversation...")
    response = run_conversation(
        client=client, model=model, user_query=query,
        tools_instance=tools, video_path=video_path, image_paths=image_paths,
        max_frame_num=max_frame_num, fps=fps, api_type=api_type, verbose=True,
    )

    print("\n" + "=" * 60)
    print("Final Answer:")
    print("=" * 60)
    print(response)


def main():
    parser = argparse.ArgumentParser(description="World2Mind OpenAI Demo")

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to video or single image file")
    input_group.add_argument("--image-paths", nargs="+", help="List of image file paths")

    parser.add_argument("--query", help="User query (required for API mode)")
    parser.add_argument("--direct", action="store_true", help="Run tools directly without LLM API")
    parser.add_argument("--categories", nargs="+", default=["car", "building", "tree", "person"])
    parser.add_argument("--scene-type", choices=["indoor", "outdoor"], default="outdoor")
    parser.add_argument("--knowledge-type", choices=["landmark", "route", "both"], default="both")
    parser.add_argument("--output-format", choices=["grid", "rectangle", "ellipse"], default="grid")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model name")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key (defaults to env OPENAI_API_KEY)")
    parser.add_argument("--api-type", choices=["openai", "claude"], default=None,
                        help="API image format: 'openai' for GPT/vLLM, 'claude' for Anthropic Claude. Auto-detected from model name if not set.")
    parser.add_argument("--max-frame-num", type=int, default=32)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--service-url", default="http://localhost:8100", help="Model service URL")
    parser.add_argument("--config", default=None, help="Path to config YAML")

    args = parser.parse_args()

    # Auto-detect api_type from model name if not explicitly set
    if args.api_type is None:
        args.api_type = "claude" if "claude" in args.model.lower() else "openai"

    # Validate input
    if args.video and not os.path.exists(args.video):
        print(f"Error: File not found: {args.video}")
        sys.exit(1)
    if args.image_paths:
        for p in args.image_paths:
            if not os.path.exists(p):
                print(f"Error: File not found: {p}")
                sys.exit(1)

    if args.direct:
        demo_without_api(
            video_path=args.video, image_paths=args.image_paths,
            categories=args.categories, scene_type=args.scene_type,
            knowledge_type=args.knowledge_type, output_format=args.output_format,
            service_url=args.service_url, config_path=args.config,
        )
    else:
        if not args.query:
            print("Error: --query is required for API mode. Use --direct for direct tool execution.")
            sys.exit(1)
        demo_with_api(
            query=args.query, model=args.model,
            video_path=args.video, image_paths=args.image_paths,
            base_url=args.base_url, api_key=args.api_key,
            max_frame_num=args.max_frame_num, fps=args.fps,
            api_type=args.api_type,
            service_url=args.service_url, config_path=args.config,
        )


if __name__ == "__main__":
    main()
