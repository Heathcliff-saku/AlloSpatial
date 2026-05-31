"""
Blind Benchmark model for lmms-eval.

Integrates the World2Mind spatial intelligence pipeline with LLM tool-calling,
but WITHOUT providing the original video/image frames to the LLM.

This is an ablation model that tests whether the cognitive map alone
(without direct visual observation) is sufficient for spatial reasoning.

Each question triggers a multi-turn conversation:
  1. System prompt (blind spatial intelligence role + tool descriptions)
  2. User message (reasoning protocol + question, NO video frames)
  3. LLM calls world2mind tool (inferring categories from question text)
     → pipeline processes the actual video internally → YAML results returned
  4. LLM may call view_image to see map visualizations
  5. LLM produces final answer wrapped in <Answer></Answer>

Supports both OpenAI-compatible and Anthropic Claude APIs.
"""

import base64
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")


@register_model("blind_benchmark")
class BlindBenchmark(lmms):
    """
    Blind Benchmark: LLM + Spatial Intelligence Tool Calling WITHOUT visual input.

    Uses a remote LLM with world2mind tool calling backed by a local DA3+SAM3
    model service, but does NOT send the original video/image frames to the LLM.
    The LLM must infer categories from the question text and rely solely on
    cognitive map data (YAML + visualization images) for reasoning.
    """

    def __init__(
        self,
        model_version: str = "gpt-5.2",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        api_type: str = "auto",  # "openai", "claude", or "auto"
        service_url: str = "http://localhost:8100",
        # Path to the World2Mind tool package (the `world2mind/` dir of this repo).
        # Defaults to the WORLD2MIND_ROOT env var; pass raptor_root=... to override.
        raptor_root: str = None,
        num_workers: int = 1,  # kept for backward compat
        num_concurrent: int = 0,  # 0 = auto (use num_workers)
        max_turns: int = 5,
        max_frames_num: int = 32,
        fps: float = 1.0,
        max_retries: int = 5,
        retry_backoff_s: float = 5.0,
        timeout: float = 600.0,
        workspace_dir: str = "./workspace",
        config_path: str = None,
        max_new_tokens: int = 16384,
        temperature: float = 0.0,
        batch_size: int = 1,
        resume_log: str = "",
        doc_id: str = "",
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_version = model_version
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.service_url = service_url
        raptor_root = raptor_root or os.environ.get("WORLD2MIND_ROOT", "")
        if not raptor_root:
            raise ValueError(
                "World2Mind tool package not found. Set the WORLD2MIND_ROOT env var "
                "(or pass raptor_root=...) to the `world2mind/` directory of this repo."
            )
        self.raptor_root = raptor_root
        self.num_workers = max(1, int(num_workers))
        self.num_concurrent = int(num_concurrent) if int(num_concurrent) > 0 else self.num_workers
        self.max_turns = int(max_turns)
        self.max_frames_num = int(max_frames_num)
        self.fps = float(fps)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.timeout = float(timeout)
        self.workspace_dir = workspace_dir
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.batch_size_per_gpu = int(batch_size)

        # Auto-detect api_type
        if api_type == "auto":
            self.api_type = "claude" if "claude" in model_version.lower() else "openai"
        else:
            self.api_type = api_type

        # Config path for SpatialIntelligenceTools
        self.config_path = config_path
        if self.config_path is None:
            self.config_path = os.path.join(raptor_root, "config", "default_config.yaml")

        # Add raptor project to path
        if raptor_root not in sys.path:
            sys.path.insert(0, raptor_root)

        # Import raptor components (deferred to allow path setup)
        from tools.tool_definitions import get_tool_definitions
        from tools.spatial_tools import SpatialIntelligenceTools, handle_tool_call
        from tools.blind_prompts import BLIND_SYSTEM_PROMPT, BLIND_USER_PROMPT

        self._get_tool_definitions = get_tool_definitions
        self._SpatialIntelligenceTools = SpatialIntelligenceTools
        self._handle_tool_call = handle_tool_call
        self._system_prompt = BLIND_SYSTEM_PROMPT
        self._user_prompt_template = BLIND_USER_PROMPT

        # Create LLM client
        self._init_client()

        # Workspace setup
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Thread-safe log writing
        self._log_lock = threading.Lock()

        # Token usage tracking (thread-safe)
        self._usage_lock = threading.Lock()
        self._total_api_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0

        # Smart log naming: {task}_{model}_{timestamp}.jsonl
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Resume support: load completed doc_ids from a prior log
        self._resume_log_path = resume_log if resume_log else ""
        self._completed_cache: Dict[int, str] = {}
        if self._resume_log_path and os.path.isfile(self._resume_log_path):
            with open(self._resume_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    entry_doc_id = entry["doc_id"]
                    self._completed_cache[entry_doc_id] = entry["full_response"]
            eval_logger.info(
                f"Resume: loaded {len(self._completed_cache)} completed doc_ids "
                f"from {self._resume_log_path}"
            )

        # Rank/world size (single process)
        self._rank = 0
        self._world_size = 1

        # Single doc_id debug mode
        self._filter_doc_id = int(doc_id) if doc_id else None
        if self._filter_doc_id is not None:
            eval_logger.info(f"Single doc mode: only doc_id={self._filter_doc_id} will be evaluated (verbose output enabled)")

        eval_logger.info(
            f"BlindBenchmark initialized: model={model_version}, api_type={self.api_type}, "
            f"service={service_url}, workers={num_workers}, concurrent={self.num_concurrent}"
        )
        if self.num_concurrent > self.num_workers * 2:
            eval_logger.warning(
                f"num_concurrent ({self.num_concurrent}) is much higher than num_workers ({self.num_workers}). "
                f"This can cause excessive memory usage. Consider setting num_concurrent <= {self.num_workers + 2}."
            )

    def _init_client(self):
        """Initialize the LLM API client (unified OpenAI-compatible for all models)."""
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    def tok_encode(self, string: str):
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens):
        return ""

    @property
    def eot_token_id(self):
        return 0

    # ================================================================
    # Image content formatting (for tool result images only)
    # ================================================================

    def _make_image_content(self, b64_data: str) -> dict:
        """Create image content block in the correct format for the target API.

        Note: In blind mode, this is ONLY used for tool result images
        (e.g., view_image visualizations), NOT for original video frames.
        """
        if self.api_type == "claude":
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64_data},
            }
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_data}"},
        }

    # ================================================================
    # Token usage tracking
    # ================================================================

    def _accumulate_usage(self, response):
        """Extract token usage from API response and accumulate (thread-safe)."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        with self._usage_lock:
            self._total_api_calls += 1
            self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self._total_tokens += getattr(usage, "total_tokens", 0) or 0

    def get_token_usage(self) -> dict:
        """Return cumulative API call count and token usage."""
        with self._usage_lock:
            return {
                "total_api_calls": self._total_api_calls,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_tokens,
            }

    # ================================================================
    # Verbose printing helpers (single doc debug mode)
    # ================================================================

    @staticmethod
    def _format_content_for_print(content) -> str:
        """Format message content for terminal output, replacing images with [IMAGE]."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("image_url", "image"):
                        parts.append("[IMAGE]")
                    elif block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)

    def _verbose_print(self, text: str):
        """Print text to stdout (bypasses logging, goes directly to terminal)."""
        print(text, flush=True)

    # ================================================================
    # Multi-turn tool-calling conversation
    # ================================================================

    def _run_conversation_openai(
        self,
        messages: list,
        tools_instance: Any,
        tool_definitions: list,
        verbose: bool = False,
    ) -> str:
        """Run multi-turn tool-calling conversation via OpenAI-compatible API."""
        if verbose:
            # Print initial user question
            self._verbose_print("\n── [System Prompt] " + "─" * 49)
            self._verbose_print(self._format_content_for_print(messages[0].get("content", "")))
            self._verbose_print("\n── [User Question (BLIND - no images)] " + "─" * 29)
            self._verbose_print(self._format_content_for_print(messages[1].get("content", "")))

        for turn in range(self.max_turns):
            for attempt in range(self.max_retries):
                try:
                    payload = {
                        "model": self.model_version,
                        "messages": messages,
                        "tools": tool_definitions,
                        "tool_choice": "auto",
                        "max_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                    }
                    # Reasoning models don't support temperature/max_tokens
                    if any(k in self.model_version for k in ("o1", "o3", "o4", "gpt-5")):
                        payload.pop("temperature")
                        payload.pop("max_tokens")
                        payload["max_completion_tokens"] = self.max_new_tokens
                        payload["response_format"] = {"type": "text"}

                    response = self.client.chat.completions.create(**payload)
                    # Validate response type
                    if not hasattr(response, 'choices') or not response.choices:
                        raise ValueError(f"Invalid API response (type={type(response).__name__}): {str(response)[:200]}")
                    self._accumulate_usage(response)
                    break
                except Exception as e:
                    eval_logger.warning(f"OpenAI API attempt {attempt+1}/{self.max_retries} failed: {e}")
                    if attempt == self.max_retries - 1:
                        return f"[API_ERROR] {e}"
                    backoff = self.retry_backoff_s * (2 ** attempt)
                    eval_logger.info(f"Retrying in {backoff:.0f}s...")
                    time.sleep(backoff)

            msg = response.choices[0].message

            if not msg.tool_calls:
                final_text = msg.content or ""
                if verbose:
                    self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Final Response " + "─" * 30)
                    self._verbose_print(final_text)
                return final_text

            # Verbose: print assistant text + tool calls
            if verbose:
                self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Assistant " + "─" * 37)
                if msg.content:
                    self._verbose_print(msg.content)
                for tc in msg.tool_calls:
                    self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Tool Call: {tc.function.name} " + "─" * 20)
                    self._verbose_print(f"Arguments: {tc.function.arguments}")

            # Append assistant message preserving all fields
            assistant_msg = msg.model_dump(exclude_none=True, exclude_unset=True)
            if not assistant_msg.get("content"):
                assistant_msg.pop("content", None)
            messages.append(assistant_msg)

            # Process each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                arguments = json.loads(tc.function.arguments)

                tool_result = self._handle_tool_call(tool_name, arguments, tools_instance)

                if isinstance(tool_result, dict) and tool_result.get("type") == "image":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": [
                            self._make_image_content(tool_result["base64"]),
                            {"type": "text", "text": f"Visualization: {tool_result.get('visualization_type', '')}"},
                        ],
                    })
                    if verbose:
                        self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Tool Result: {tool_name} " + "─" * 18)
                        self._verbose_print(f"[IMAGE] Visualization: {tool_result.get('visualization_type', '')}")
                else:
                    if isinstance(tool_result, dict):
                        tool_result = json.dumps(tool_result)
                    # Ensure tool content is never empty
                    if not tool_result:
                        tool_result = "{}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                    if verbose:
                        self._verbose_print(f"\n── [Turn {turn+1}/{self.max_turns}] Tool Result: {tool_name} " + "─" * 18)
                        self._verbose_print(tool_result)

        last_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if verbose:
            self._verbose_print(f"\n── [Max Turns Reached] " + "─" * 46)
            self._verbose_print(self._format_content_for_print(last_content))
        return last_content

    # ================================================================
    # Image saving for non-video inputs (needed by pipeline, NOT by LLM)
    # ================================================================

    def _save_images_to_disk(
        self, visuals: list, task_name: str, doc_id: int
    ) -> Optional[List[str]]:
        """Save PIL Images to workspace directory for the pipeline to consume.

        The reconstruction pipeline needs files on disk even though the LLM
        doesn't see these images in blind mode.
        """
        save_dir = os.path.join(
            self.workspace_dir, "_temp_images", f"{task_name}_{doc_id}"
        )
        os.makedirs(save_dir, exist_ok=True)
        image_paths: List[str] = []
        for i, v in enumerate(visuals):
            if isinstance(v, Image.Image):
                path = os.path.join(save_dir, f"image_{i:04d}.png")
                if not os.path.exists(path):
                    v.convert("RGB").save(path, format="PNG")
                image_paths.append(path)
            elif isinstance(v, str) and not v.endswith(
                (".mp4", ".avi", ".mov", ".flv", ".wmv", ".mkv")
            ):
                image_paths.append(v)
        return image_paths if image_paths else None

    # ================================================================
    # Extract video/image paths from visuals (without encoding for LLM)
    # ================================================================

    def _extract_input_paths(
        self, visuals: list, task_name: str, doc_id: int
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract video_path and image_paths from visual inputs.

        Unlike world2mind.py, we do NOT encode frames for the LLM.
        We only need the paths for the SpatialIntelligenceTools pipeline.
        """
        video_path = None
        for v in visuals:
            if isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".flv", ".wmv", ".mkv")):
                video_path = v
                break

        image_paths = None
        if video_path is None and visuals:
            image_paths = self._save_images_to_disk(visuals, task_name, doc_id)

        return video_path, image_paths

    # ================================================================
    # Single request processing
    # ================================================================

    def _process_single_request(
        self,
        context: str,
        visuals: list,
        gen_kwargs: dict,
        task_name: str = "",
        doc_id: int = 0,
        verbose: bool = False,
    ) -> str:
        """Process a single benchmark question with tool-calling conversation.

        Key difference from World2Mind: NO video/image frames are sent to the LLM.
        The user message contains only text (reasoning protocol + question).
        The pipeline still processes the video internally when tools are called.
        """
        # Extract input paths for the pipeline (NOT for LLM message)
        video_path, image_paths = self._extract_input_paths(visuals, task_name, doc_id)

        # Create SpatialIntelligenceTools instance for this request
        tools_instance = self._SpatialIntelligenceTools(
            config_path=self.config_path,
            output_base=self.workspace_dir,
            service_url=self.service_url,
            video_path=video_path,
            image_paths=image_paths,
        )

        # Build user prompt: reasoning protocol + question (NO images)
        user_text = self._user_prompt_template.format(query=context)

        # BLIND MODE: user message contains only text, no image frames
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_text},
        ]

        tool_definitions = self._get_tool_definitions()

        response = self._run_conversation_openai(
            messages=messages,
            tools_instance=tools_instance,
            tool_definitions=tool_definitions,
            verbose=verbose,
        )

        return response

    # ================================================================
    # Workspace logging
    # ================================================================

    def _get_log_file_path(self, task_name: str) -> str:
        """Determine the log file path for this run."""
        if self._resume_log_path:
            return self._resume_log_path
        log_dir = os.path.join(self.workspace_dir, task_name)
        os.makedirs(log_dir, exist_ok=True)
        safe_model = self.model_version.replace("/", "-")
        filename = f"{task_name}_blind_{safe_model}_{self._run_timestamp}.jsonl"
        return os.path.join(log_dir, filename)

    def _log_result(
        self,
        task_name: str,
        doc_id: int,
        question: str,
        full_response: str,
        video_path: Optional[str],
        ground_truth: str = "",
        scene_name: str = "",
    ):
        """Write a single result entry to the workspace log (thread-safe)."""
        log_file = self._get_log_file_path(task_name)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        entry = {
            "doc_id": doc_id,
            "scene_name": scene_name,
            "video_path": video_path or "",
            "question": question,
            "full_response": full_response,
            "ground_truth": ground_truth,
        }

        with self._log_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ================================================================
    # lmms interface
    # ================================================================

    def generate_until(self, requests) -> List[str]:
        if not requests:
            return []

        # Reorder for efficiency
        from lmms_eval import utils

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        ordered_requests = []
        for single_request in re_ords.get_batched(n=1, batch_fn=None):
            ordered_requests.extend(single_request)

        pbar = tqdm(
            total=len(ordered_requests),
            disable=(self.rank != 0),
            desc="Blind Benchmark Evaluating",
        )

        responses: List[Union[str, None]] = [None] * len(ordered_requests)

        def process_request(idx: int):
            (context, gen_kwargs, doc_to_visual_fn, doc_id, task_name, split_name) = ordered_requests[idx]

            # Single doc_id filter: skip non-matching requests
            if self._filter_doc_id is not None and doc_id != self._filter_doc_id:
                return idx, ""

            verbose = self._filter_doc_id is not None and doc_id == self._filter_doc_id

            # Resume: return cached response if available
            if doc_id in self._completed_cache:
                eval_logger.info(f"[{task_name}] doc_id={doc_id} skipped (cached from resume log)")
                return idx, self._completed_cache[doc_id]

            # Get visual input (for pipeline, NOT for LLM)
            doc = self.task_dict[task_name][split_name][doc_id]
            visuals_raw = doc_to_visual_fn(doc)
            visuals = visuals_raw if visuals_raw is not None else []
            if not isinstance(visuals, list):
                visuals = [visuals]

            # Get ground truth and scene info for logging
            ground_truth = str(doc.get("ground_truth", doc.get("gt_answer", "")))
            scene_name = str(doc.get("scene_name", doc.get("id", "")))
            video_path = None
            for v in visuals:
                if isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov")):
                    video_path = v
                    break

            if verbose:
                self._verbose_print("\n" + "=" * 68)
                self._verbose_print(f" [Single Doc Mode - BLIND] doc_id={doc_id}, scene={scene_name}")
                self._verbose_print(f" Ground Truth: {ground_truth}")
                self._verbose_print(f" Video: {video_path or 'N/A'} (NOT sent to LLM)")
                self._verbose_print("=" * 68)

            started = time.time()
            try:
                response = self._process_single_request(
                    context=context,
                    visuals=visuals,
                    gen_kwargs=gen_kwargs,
                    task_name=task_name,
                    doc_id=doc_id,
                    verbose=verbose,
                )
            except Exception as e:
                eval_logger.error(f"Request {idx} failed: {e}")
                response = f"[ERROR] {e}"

            elapsed = time.time() - started
            eval_logger.info(f"[{task_name}] doc_id={doc_id} done in {elapsed:.1f}s")

            if verbose:
                self._verbose_print("\n" + "=" * 68)
                self._verbose_print(f" Completed: doc_id={doc_id}, elapsed={elapsed:.1f}s")
                self._verbose_print("=" * 68 + "\n")

            # Log to workspace
            self._log_result(
                task_name=task_name,
                doc_id=doc_id,
                question=context,
                full_response=response,
                video_path=video_path,
                ground_truth=ground_truth,
                scene_name=scene_name,
            )

            return idx, response

        with ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
            futures = {executor.submit(process_request, i): i for i in range(len(ordered_requests))}
            for future in as_completed(futures):
                idx, response = future.result()
                responses[idx] = response
                pbar.update(1)

        pbar.close()
        completed = [r if r is not None else "" for r in responses]
        return re_ords.get_original(completed)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Use generate_until which already supports multi-round tool calling")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("BlindBenchmark does not support loglikelihood")
