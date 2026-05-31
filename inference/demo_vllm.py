"""
Demo for World2Mind Spatial Intelligence Tools with locally-served Qwen3-VL model.

The trained model uses TEXT-BASED tool calling (not structured OpenAI function
calling). Tool calls appear as:
    <tool_call>
    {"name": "world2mind", "arguments": {...}}
    </tool_call>

Tool results are sent back as plain user messages (matching the training format).

Connects to:
  1. vLLM / swift deploy server (OpenAI-compatible API) for LLM inference
  2. DA3+SAM3 model service (HTTP) for spatial intelligence pipeline

Usage:
    # Start model server first (in another terminal):
    bash demo/start_server.sh

    # Single query - video
    python demo/demo_vllm.py --video /path/to/video.mp4 --query "What objects are here?"

    # Single query - image list
    python demo/demo_vllm.py --image-paths /path/a.jpg /path/b.jpg --query "Describe the layout"

    # Interactive mode
    python demo/demo_vllm.py --interactive

    # Custom server
    python demo/demo_vllm.py --video /path/to/video.mp4 --query "..." \
        --server-url http://localhost:8003/v1 --model qwen3-vl-raptor
"""

import os
import sys
import re
import json
import base64
import argparse
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from openai import OpenAI

# Locate the World2Mind tool package (the `world2mind/` dir of this repo).
_W2M_ROOT = os.environ.get("WORLD2MIND_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world2mind")
sys.path.insert(0, _W2M_ROOT)

from tools.spatial_tools import SpatialIntelligenceTools, handle_tool_call
from tools.prompts import VLLM_SYSTEM_PROMPT, VLLM_USER_PROMPT

logger = logging.getLogger(__name__)


# ============================================================
# Tool call parsing (text-based, not structured)
# ============================================================

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from model text output.

    The trained model outputs tool calls as:
        <tool_call>
        {"name": "world2mind", "arguments": {...}}
        </tool_call>

    Returns list of dicts with 'name' and 'arguments' keys.
    """
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if 'name' in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call JSON: {match[:200]}")
    return tool_calls


def split_text_and_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Split model output into reasoning text and tool calls.

    Returns (reasoning_text, tool_calls).
    """
    tool_calls = parse_tool_calls(text)
    # Remove tool call blocks from text to get pure reasoning
    reasoning = re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', text, flags=re.DOTALL).strip()
    return reasoning, tool_calls


# ============================================================
# Visual encoding
# ============================================================

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


# ============================================================
# SpatialAgent: multi-turn tool calling with text parsing
# ============================================================

class SpatialAgent:
    """
    Agent that uses a locally-served Qwen3-VL model for reasoning
    and the DA3+SAM3 HTTP service for spatial intelligence tools.

    Key difference from the original demo_vllm.py:
    - Does NOT use structured OpenAI tool calling (tools=, tool_choice=)
    - Parses <tool_call> from model text output
    - Sends tool results as user messages (matching training format)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8003/v1",
        model: str = "qwen3-vl-raptor",
        service_url: str = "http://localhost:8100",
        config_path: Optional[str] = None,
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        max_frame_num: int = 32,
        fps: float = 1.0,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(base_url=server_url, api_key="EMPTY")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.video_path = video_path
        self.image_paths = image_paths
        self.max_frame_num = max_frame_num
        self.fps = fps

        self.tools = SpatialIntelligenceTools(
            config_path=config_path,
            service_url=service_url,
            video_path=video_path,
            image_paths=image_paths,
        )

    def _build_visual_content(self) -> List[dict]:
        """Build visual content blocks for the first user message.

        vLLM native API only supports image_url content blocks, not video.
        Videos are encoded as multiple base64 image frames.
        """
        content = []
        if self.video_path:
            if self.video_path.endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv')):
                b64_frames = encode_video(self.video_path, self.max_frame_num, self.fps)
                print(f"Encoded {len(b64_frames)} video frames")
                for b64 in b64_frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
            else:
                b64 = encode_image(self.video_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        elif self.image_paths:
            for p in self.image_paths:
                b64 = encode_image(p)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        return content

    def chat(
        self,
        user_query: str,
        max_turns: int = 5,
        verbose: bool = True,
    ) -> str:
        """Process a query with multi-turn tool calling."""
        # Build system message (contains tool descriptions embedded in text)
        system_content = VLLM_SYSTEM_PROMPT

        # Build user message: visual content + reasoning protocol + question
        user_text = VLLM_USER_PROMPT.format(query=user_query)
        user_content = self._build_visual_content()
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        for turn in range(max_turns):
            if verbose:
                print(f"\n[Turn {turn + 1}/{max_turns}]")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
            except Exception as e:
                return f"API Error: {e}"

            msg = response.choices[0].message
            text = msg.content or ""

            # Parse tool calls from text
            reasoning, tool_calls = split_text_and_tool_calls(text)

            if verbose and reasoning:
                print(f"\nAssistant:\n{reasoning}")

            if not tool_calls:
                # No tool calls - return final response
                return text

            # Add assistant message to history
            messages.append({"role": "assistant", "content": text})

            # Execute each tool call and add results as user messages
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                arguments = tc.get("arguments", {})

                if verbose:
                    print(f"\n  Tool Call: {tool_name}")
                    print(f"    Arguments: {json.dumps(arguments, ensure_ascii=False)}")

                try:
                    result = handle_tool_call(tool_name, arguments, self.tools)
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' raised exception: {e}")
                    result = json.dumps({"error": f"Tool execution failed: {e}"})

                if isinstance(result, dict) and result.get("type") == "image":
                    # view_image result: send as user message with image
                    vis_type = result.get("visualization_type", "unknown")
                    if verbose:
                        print(f"    Result: [IMAGE] {vis_type}")

                    b64_data = result["base64"]
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}},
                            {"type": "text", "text": f"Visualization: {vis_type}"},
                        ],
                    })
                else:
                    # world2mind result: send as user message with JSON text
                    if isinstance(result, dict):
                        result = json.dumps(result, ensure_ascii=False)

                    if verbose:
                        preview = result[:200] + "..." if len(result) > 200 else result
                        print(f"    Result: {preview}")

                    messages.append({
                        "role": "user",
                        "content": result,
                    })

        return "Max turns reached"


# ============================================================
# Interactive mode
# ============================================================

def interactive_mode(
    server_url: str,
    model: str,
    service_url: str,
    config_path: Optional[str],
    default_video: Optional[str] = None,
    max_frame_num: int = 32,
    fps: float = 1.0,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode (Qwen3-VL + World2Mind)")
    print("=" * 60)
    print("Commands:")
    print("  /video <path>       - Set video path")
    print("  /images <p1> <p2>   - Set image list")
    print("  /quit               - Exit")
    print("=" * 60)

    video_path = default_video
    image_paths = None
    agent = None

    if video_path:
        agent = SpatialAgent(
            server_url=server_url, model=model,
            service_url=service_url, config_path=config_path,
            video_path=video_path,
            max_frame_num=max_frame_num, fps=fps,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.startswith("/video "):
            video_path = query[7:].strip()
            image_paths = None
            agent = SpatialAgent(
                server_url=server_url, model=model,
                service_url=service_url, config_path=config_path,
                video_path=video_path,
                max_frame_num=max_frame_num, fps=fps,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            print(f"Video set to: {video_path}")
            continue
        if query.startswith("/images "):
            image_paths = query[8:].strip().split()
            video_path = None
            agent = SpatialAgent(
                server_url=server_url, model=model,
                service_url=service_url, config_path=config_path,
                image_paths=image_paths,
                max_frame_num=max_frame_num, fps=fps,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            print(f"Images set to: {image_paths}")
            continue
        if query == "/quit":
            print("Goodbye!")
            break
        if agent is None:
            print("Please set input first: /video <path> or /images <p1> <p2>")
            continue

        response = agent.chat(query, verbose=True)
        print(f"\n{'=' * 60}")
        print("Final Answer:")
        print("=" * 60)
        print(response)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="World2Mind Demo (Qwen3-VL local model)")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--video", help="Path to video/image")
    input_group.add_argument("--image-paths", nargs="+", help="List of image file paths")

    parser.add_argument("--query", help="Question about the video/images")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--server-url", default="http://localhost:8003/v1",
                        help="vLLM/swift deploy server URL (default: http://localhost:8003/v1)")
    parser.add_argument("--model", default="qwen3-vl-raptor",
                        help="Model name on the server (default: qwen3-vl-raptor)")
    parser.add_argument("--service-url", default="http://localhost:9100",
                        help="DA3+SAM3 model service URL")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--max-frame-num", type=int, default=24, help="Max video frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(
            server_url=args.server_url, model=args.model,
            service_url=args.service_url, config_path=args.config,
            default_video=args.video,
            max_frame_num=args.max_frame_num, fps=args.fps,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
    elif (args.video or args.image_paths) and args.query:
        agent = SpatialAgent(
            server_url=args.server_url, model=args.model,
            service_url=args.service_url, config_path=args.config,
            video_path=args.video, image_paths=args.image_paths,
            max_frame_num=args.max_frame_num, fps=args.fps,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        response = agent.chat(args.query)
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        print(response)
    else:
        print("Usage:")
        print("  Video:       python demo/demo_vllm.py --video <path> --query <question>")
        print("  Images:      python demo/demo_vllm.py --image-paths <p1> <p2> --query <question>")
        print("  Interactive:  python demo/demo_vllm.py --interactive")


if __name__ == "__main__":
    main()
