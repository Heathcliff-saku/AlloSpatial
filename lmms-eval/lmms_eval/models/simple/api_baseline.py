"""
API Baseline model for lmms-eval.

Pure LLM API inference without any tool calling or spatial intelligence pipeline.
Used as a baseline to measure the performance gain from World2Mind.

Each question: video frames + question → LLM → answer (single turn, no tools).

Supports both OpenAI-compatible and Anthropic Claude APIs.
"""

import base64
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
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


@register_model("api_baseline")
class APIBaseline(lmms):
    """
    API Baseline: Direct LLM API calls without tool calling.

    Sends video frames + question to a remote LLM and returns the response.
    Supports OpenAI-compatible and Anthropic Claude APIs.
    """

    def __init__(
        self,
        model_version: str = "gpt-5.2",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        api_type: str = "auto",  # "openai", "claude", or "auto"
        num_concurrent: int = 4,
        max_frames_num: int = 32,
        fps: float = 1.0,
        max_retries: int = 5,
        retry_backoff_s: float = 5.0,
        timeout: float = 600.0,
        workspace_dir: str = "./workspace",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        batch_size: int = 1,
        doc_id: str = "",
        system_prompt: str = "You are a helpful assistant. Answer the question based on the provided visual input.",
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_version = model_version
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.num_concurrent = max(1, int(num_concurrent))
        self.max_frames_num = int(max_frames_num)
        self.fps = float(fps)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.timeout = float(timeout)
        self.workspace_dir = workspace_dir
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.batch_size_per_gpu = int(batch_size)
        self.system_prompt = system_prompt

        # Auto-detect api_type
        if api_type == "auto":
            self.api_type = "claude" if "claude" in model_version.lower() else "openai"
        else:
            self.api_type = api_type

        # Create LLM client
        self._init_client()

        # Workspace setup
        os.makedirs(self.workspace_dir, exist_ok=True)
        self._log_lock = threading.Lock()

        # Token usage tracking (thread-safe)
        self._usage_lock = threading.Lock()
        self._total_api_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0

        self._rank = 0
        self._world_size = 1

        # Single doc_id debug mode
        self._filter_doc_id = int(doc_id) if doc_id else None
        if self._filter_doc_id is not None:
            eval_logger.info(f"Single doc mode: only doc_id={self._filter_doc_id} will be evaluated (verbose output enabled)")

        eval_logger.info(
            f"APIBaseline initialized: model={model_version}, api_type={self.api_type}, "
            f"concurrent={num_concurrent}, system_prompt={'yes' if self.system_prompt else 'none'}"
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
    # Visual encoding
    # ================================================================

    def encode_image(self, image: Union[Image.Image, str]) -> str:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy().convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def encode_video(self, video_path: str) -> List[str]:
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        video_fps = vr.get_avg_fps()

        if self.fps > 0:
            interval = max(1, int(video_fps / self.fps))
            frame_idx = list(range(0, total, interval))
            if len(frame_idx) > self.max_frames_num:
                indices = np.linspace(0, len(frame_idx) - 1, self.max_frames_num, dtype=int)
                frame_idx = [frame_idx[i] for i in indices]
        else:
            frame_idx = np.linspace(0, total - 1, self.max_frames_num, dtype=int).tolist()

        if total - 1 not in frame_idx:
            frame_idx.append(total - 1)

        frames = vr.get_batch(frame_idx).asnumpy()
        b64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return b64_frames

    def encode_visuals(self, visuals: list) -> Tuple[List[str], Optional[str]]:
        b64_frames = []
        video_path = None
        for v in visuals:
            if isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".flv", ".wmv", ".mkv")):
                video_path = v
                b64_frames.extend(self.encode_video(v))
            elif isinstance(v, str):
                b64_frames.append(self.encode_image(v))
            elif isinstance(v, Image.Image):
                b64_frames.append(self.encode_image(v))
            elif hasattr(v, "convert"):
                # Duck-typing fallback for PIL-compatible objects (e.g. HF lazy wrappers)
                eval_logger.debug(f"encode_visuals: duck-typed PIL-compatible object {type(v).__name__}")
                b64_frames.append(self.encode_image(v))
            else:
                eval_logger.warning(f"encode_visuals: skipping unrecognized visual type {type(v).__name__}")
        if visuals and not b64_frames:
            eval_logger.warning(f"encode_visuals: 0 frames encoded from {len(visuals)} visual inputs! Types: {[type(v).__name__ for v in visuals]}")
        return b64_frames, video_path

    # ================================================================
    # Image content formatting
    # ================================================================

    def _make_image_content(self, b64_data: str) -> dict:
        if self.api_type == "claude":
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64_data},
            }
        else:
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
    # Single-turn API call
    # ================================================================

    def _call_openai(self, messages: list) -> str:
        """Single-turn OpenAI-compatible API call."""
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_version,
                    "messages": messages,
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
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError(f"Invalid API response (type={type(response).__name__}): {str(response)[:200]}")
                self._accumulate_usage(response)
                return response.choices[0].message.content or ""
            except Exception as e:
                eval_logger.warning(f"OpenAI API attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    return f"[API_ERROR] {e}"
                backoff = self.retry_backoff_s * (2 ** attempt)
                eval_logger.info(f"Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
        return ""

    def _process_single_request(self, context: str, visuals: list, verbose: bool = False) -> str:
        """Process a single benchmark question via direct API call."""
        b64_frames, _ = self.encode_visuals(visuals)

        # Build user content: frames + question text
        user_content = []
        for b64 in b64_frames:
            user_content.append(self._make_image_content(b64))
        user_content.append({"type": "text", "text": context})

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})

        if verbose:
            if self.system_prompt:
                self._verbose_print("\n── [System Prompt] " + "─" * 49)
                self._verbose_print(self.system_prompt)
            else:
                self._verbose_print("\n── [No System Prompt] " + "─" * 46)
            self._verbose_print("\n── [User Question] " + "─" * 49)
            self._verbose_print(self._format_content_for_print(user_content))

        response = self._call_openai(messages)

        if verbose:
            self._verbose_print("\n── [Response] " + "─" * 54)
            self._verbose_print(response)

        return response

    # ================================================================
    # Workspace logging
    # ================================================================

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
        log_dir = os.path.join(self.workspace_dir, task_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "baseline_eval_log.jsonl")

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
            desc="API Baseline Evaluating",
        )

        responses: List[Union[str, None]] = [None] * len(ordered_requests)

        def process_request(idx: int):
            (context, gen_kwargs, doc_to_visual_fn, doc_id, task_name, split_name) = ordered_requests[idx]

            # Single doc_id filter: skip non-matching requests
            if self._filter_doc_id is not None and doc_id != self._filter_doc_id:
                return idx, ""

            verbose = self._filter_doc_id is not None and doc_id == self._filter_doc_id

            doc = self.task_dict[task_name][split_name][doc_id]
            visuals_raw = doc_to_visual_fn(doc)
            visuals = visuals_raw if visuals_raw is not None else []
            if not isinstance(visuals, list):
                visuals = [visuals]

            ground_truth = str(doc.get("ground_truth", doc.get("gt_answer", "")))
            scene_name = str(doc.get("scene_name", doc.get("id", "")))
            video_path = None
            for v in visuals:
                if isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov")):
                    video_path = v
                    break

            if verbose:
                self._verbose_print("\n" + "=" * 68)
                self._verbose_print(f" [Single Doc Mode] doc_id={doc_id}, scene={scene_name}")
                self._verbose_print(f" Ground Truth: {ground_truth}")
                self._verbose_print(f" Video: {video_path or 'N/A'}")
                self._verbose_print("=" * 68)

            started = time.time()
            try:
                response = self._process_single_request(context=context, visuals=visuals, verbose=verbose)
            except Exception as e:
                eval_logger.error(f"Request {idx} failed: {e}")
                response = f"[ERROR] {e}"

            elapsed = time.time() - started
            eval_logger.info(f"[{task_name}] doc_id={doc_id} done in {elapsed:.1f}s")

            if verbose:
                self._verbose_print("\n" + "=" * 68)
                self._verbose_print(f" Completed: doc_id={doc_id}, elapsed={elapsed:.1f}s")
                self._verbose_print("=" * 68 + "\n")

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
        raise NotImplementedError("APIBaseline is single-turn only")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("APIBaseline does not support loglikelihood")
