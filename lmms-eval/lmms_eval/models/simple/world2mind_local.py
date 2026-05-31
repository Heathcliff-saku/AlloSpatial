"""
World2Mind Local model for lmms-eval.

A variant of World2Mind that works with locally-served models (e.g., Qwen3-VL
fine-tuned with ms-swift) that use TEXT-BASED tool calling instead of structured
OpenAI function calling.

The trained model outputs tool calls as plain text:
    <tool_call>
    {"name": "world2mind", "arguments": {...}}
    </tool_call>

Tool results are sent back as user messages (matching training format).

Usage with lmms-eval:
    python -m lmms_eval \
        --model world2mind_local \
        --model_args model_version=qwen3-vl-raptor,base_url=http://localhost:8003/v1,...
        --tasks vsibench \
        --batch_size 1
"""

import json
import os
import re
import sys
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from lmms_eval.api.registry import register_model

# Import the base World2Mind class
from .world2mind import World2Mind

logger = logging.getLogger(__name__)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from model text output.

    The trained model outputs:
        <tool_call>
        {"name": "world2mind", "arguments": {...}}
        </tool_call>
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


@register_model("world2mind_local")
class World2MindLocal(World2Mind):
    """
    World2Mind variant for locally-served models with text-based tool calling.

    Inherits visual encoding, workspace management, and evaluation logic
    from World2Mind. Overrides the conversation method to use text-based
    tool call parsing instead of structured OpenAI function calling.
    """

    def __init__(self, **kwargs):
        # Default base_url for local vLLM/swift deploy server
        if 'base_url' not in kwargs or not kwargs['base_url']:
            kwargs['base_url'] = "http://localhost:8003/v1"
        if 'api_key' not in kwargs or not kwargs['api_key']:
            kwargs['api_key'] = "EMPTY"
        super().__init__(**kwargs)

        # Override system prompt to match training data exactly
        from tools.prompts import TRAINING_SYSTEM_PROMPT
        self._system_prompt = TRAINING_SYSTEM_PROMPT

        self._mp_client = self._maybe_build_multiport_client()

    def _maybe_build_multiport_client(self):
        """Build a MultiPortModelClient when WORLD2MIND_MULTIPORT=1.

        Mirrors the GRPO RaptorToolScheduler convention so a single set of
        env vars drives both training rollout and benchmark evaluation.
        Returns None in single-port mode, in which case SpatialIntelligenceTools
        falls back to creating its own ModelServiceClient against service_url.
        """
        mp_flag = os.environ.get("WORLD2MIND_MULTIPORT", "0")
        logger.info(f"WORLD2MIND_MULTIPORT={mp_flag!r}")
        if mp_flag != "1":
            logger.info(
                f"Multi-port DISABLED — using single-port service_url={self.service_url}"
            )
            return None
        try:
            from tools.train_multiproc_client import MultiPortModelClient
            base_port = int(os.environ["WORLD2MIND_BASE_PORT"])
            gpu_ids = [int(x) for x in os.environ["WORLD2MIND_GPU_IDS"].split(",") if x.strip()]
            instances_per_gpu = int(os.environ.get("WORLD2MIND_INSTANCES_PER_GPU", "1"))
        except (KeyError, ValueError) as e:
            logger.warning(
                f"Multi-port requested but env vars invalid ({e}); "
                f"falling back to single-port service_url={self.service_url}"
            )
            return None
        client = MultiPortModelClient(
            base_port=base_port,
            gpu_ids=gpu_ids,
            instances_per_gpu=instances_per_gpu,
        )
        logger.info(
            f"Multi-port ENABLED base_port={base_port} gpu_ids={gpu_ids} "
            f"instances_per_gpu={instances_per_gpu} "
            f"total_ports={len(gpu_ids) * instances_per_gpu}"
        )
        return client

    def _init_client(self):
        """Initialize client - same as parent but with EMPTY api_key default."""
        from openai import OpenAI
        if not self.api_key:
            self.api_key = "EMPTY"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _make_visual_content(self, b64_frames: List[str], video_path: Optional[str] = None) -> List[dict]:
        """Build visual content blocks for vLLM / OpenAI-compatible API.

        vLLM native API uses standard OpenAI image_url format.
        Videos are pre-encoded as multiple base64 image frames by the caller.
        """
        content = []
        for b64 in b64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        return content

    def _run_conversation_openai(
        self,
        messages: list,
        tools_instance: Any,
        tool_definitions: list,
        verbose: bool = False,
        video_path: Optional[str] = None,
    ) -> str:
        """Run multi-turn conversation with text-based tool call parsing.

        Key differences from parent:
        - Does NOT pass tools= or tool_choice= to the API
        - Parses <tool_call> from text output
        - Sends tool results as user messages (not role=tool)
        """
        if verbose:
            self._verbose_print("\n── [System Prompt] " + "─" * 49)
            self._verbose_print(self._format_content_for_print(messages[0].get("content", "")))
            self._verbose_print("\n── [User Question] " + "─" * 49)
            self._verbose_print(self._format_content_for_print(messages[1].get("content", "")))

        for turn in range(self.max_turns):
            for attempt in range(self.max_retries):
                try:
                    payload = {
                        "model": self.model_version,
                        "messages": messages,
                        "max_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                    }
                    response = self.client.chat.completions.create(**payload)
                    if not hasattr(response, 'choices') or not response.choices:
                        raise ValueError(f"Invalid API response: {str(response)[:200]}")
                    self._accumulate_usage(response)
                    break
                except Exception as e:
                    logger.warning(f"API attempt {attempt+1}/{self.max_retries} failed: {e}")
                    if attempt == self.max_retries - 1:
                        return f"[API_ERROR] {e}"
                    backoff = self.retry_backoff_s * (2 ** attempt)
                    logger.info(f"Retrying in {backoff:.0f}s...")
                    time.sleep(backoff)

            msg = response.choices[0].message
            text = msg.content or ""

            # Parse tool calls from text
            tool_calls = parse_tool_calls(text)

            if not tool_calls:
                # No tool calls - return final response
                if verbose:
                    self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Final Response " + "─" * 30)
                    self._verbose_print(text)
                return text

            # Verbose: print reasoning + tool calls
            if verbose:
                self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Assistant " + "─" * 37)
                self._verbose_print(text)

            # Add assistant message to history (full text including tool_call tags)
            messages.append({"role": "assistant", "content": text})

            # Execute each tool call and add results as user messages
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                arguments = tc.get("arguments", {})

                if verbose:
                    self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Executing: {tool_name} " + "─" * 20)
                    self._verbose_print(f"Arguments: {json.dumps(arguments, ensure_ascii=False)}")

                try:
                    tool_result = self._handle_tool_call(tool_name, arguments, tools_instance)
                except Exception as e:
                    logger.warning(f"Tool '{tool_name}' raised exception: {e}")
                    tool_result = json.dumps({"error": f"Tool execution failed: {e}"})

                if isinstance(tool_result, dict) and tool_result.get("type") == "image":
                    # view_image result: send as user message with image
                    vis_type = tool_result.get("visualization_type", "")
                    b64_data = tool_result["base64"]
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}},
                            {"type": "text", "text": f"Visualization: {vis_type}"},
                        ],
                    })
                    if verbose:
                        self._verbose_print(f"Result: [IMAGE] {vis_type}")
                else:
                    # world2mind result: send as user message with JSON text
                    if isinstance(tool_result, dict):
                        tool_result = json.dumps(tool_result, ensure_ascii=False)
                    if not tool_result:
                        tool_result = "{}"
                    messages.append({
                        "role": "user",
                        "content": tool_result,
                    })
                    if verbose:
                        self._verbose_print(f"Result: {tool_result[:200]}...")

        last_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if verbose:
            self._verbose_print(f"\n── [Max Turns Reached] " + "─" * 46)
            self._verbose_print(self._format_content_for_print(last_content))
        return last_content if isinstance(last_content, str) else str(last_content)

    def _process_single_request(
        self,
        context: str,
        visuals: list,
        gen_kwargs: dict,
        task_name: str = "",
        doc_id: int = 0,
        verbose: bool = False,
    ) -> str:
        """Process a single benchmark question with text-based tool calling.

        Overrides parent to:
        - Use swift deploy's visual content format (video path or base64 images)
        - Not pass structured tool definitions to the API
        """
        # Encode visuals
        b64_frames, video_path = self.encode_visuals(visuals)

        # Save PIL images to disk for the reconstruction pipeline
        image_paths = None
        if video_path is None and visuals:
            image_paths = self._save_images_to_disk(visuals, task_name, doc_id)

        # Create SpatialIntelligenceTools instance.
        # service_client is a shared MultiPortModelClient when WORLD2MIND_MULTIPORT=1
        # (load-balances across all w2m ports with failover); None falls back to
        # a per-instance single-port ModelServiceClient against service_url.
        tools_instance = self._SpatialIntelligenceTools(
            config_path=self.config_path,
            output_base=self.workspace_dir,
            service_url=self.service_url,
            video_path=video_path,
            image_paths=image_paths,
            service_client=self._mp_client,
        )

        # Build user prompt
        user_text = self._user_prompt_template.format(query=context)

        # Build visual content for swift deploy
        visual_content = self._make_visual_content(b64_frames, video_path)
        user_content = visual_content + [{"type": "text", "text": user_text}]
        del b64_frames

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        # tool_definitions not used for API calls, but needed for handle_tool_call
        response = self._run_conversation_openai(
            messages=messages,
            tools_instance=tools_instance,
            tool_definitions=[],  # not used in text-based mode
            verbose=verbose,
            video_path=video_path,
        )

        return response
