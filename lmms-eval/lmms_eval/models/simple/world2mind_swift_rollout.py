"""
World2Mind swift-rollout adapter for lmms-eval.

Evaluates a model being trained by ms-swift GRPO by talking directly to the
rollout server's `/infer/` endpoint. The server-side `raptor_tool_scheduler`
plugin handles multi-turn tool calling internally, so a single /infer/ call
returns the final merged assistant output (same text the reward functions see).

Used by `InlineEvalCallback` during GRPO training to measure pass@1 on
MindCube / VSI-bench against the current policy (weights synced every step
by swift).

Usage (programmatic via simple_evaluate):
    simple_evaluate(
        model="world2mind_swift_rollout",
        model_args=(
            "base_url=http://localhost:9200,"
            "service_url=http://localhost:9100,"
            "model=/path/to/training_model,"
            "num_concurrent=8"
        ),
        tasks=["mindcube_tiny", "vsibench_tiny"],
        limit=100,
    )
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from lmms_eval.api.registry import register_model

from .world2mind_local import World2MindLocal

logger = logging.getLogger(__name__)


@register_model("world2mind_swift_rollout")
class World2MindSwiftRollout(World2MindLocal):
    """
    Adapter that sends a single /infer/ request per sample to a swift rollout
    server. The server's RaptorToolScheduler runs the multi-turn tool-calling
    loop; we only see the final merged assistant text.

    Key differences from World2MindLocal:
      - base_url points at the swift-rollout root (not /v1).
      - No OpenAI client; we POST to {base_url}/infer/ directly.
      - No local tools_instance, no client-side tool parsing loop — scheduler
        owns all of that.
    """

    def __init__(self, **kwargs):
        # Default: swift-rollout root (not /v1/...).
        base_url = kwargs.pop("base_url", None) or "http://localhost:9200"
        base_url = base_url.rstrip("/")
        # Strip an accidental /v1 suffix; /infer/ is at the root.
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]

        # Bypass World2MindLocal's OpenAI client init by passing a dummy
        # base_url and api_key; we'll override self.client afterward.
        kwargs["base_url"] = base_url + "/v1"
        kwargs["api_key"] = "EMPTY"
        super().__init__(**kwargs)

        # Replace the OpenAI client; we don't use it here.
        self.client = None
        self.rollout_base_url = base_url
        self._session = requests.Session()

    def _init_client(self):
        """No-op: /infer/ is called with a requests.Session, no OpenAI client."""
        self.client = None

    # ------------------------------------------------------------------
    # Single-request execution against /infer/
    # ------------------------------------------------------------------

    def _process_single_request(
        self,
        context: str,
        visuals: list,
        gen_kwargs: dict,
        task_name: str = "",
        doc_id: int = 0,
        verbose: bool = False,
    ) -> str:
        """Send one /infer/ request; scheduler runs the full tool-calling loop
        server-side and returns merged assistant text.
        """
        # Resolve media: prefer on-disk paths (videos or saved images) so the
        # scheduler's media_key and the raptor pipeline both see real files.
        video_path: Optional[str] = None
        image_paths: Optional[List[str]] = None

        for v in visuals:
            if isinstance(v, str) and v.endswith(
                (".mp4", ".avi", ".mov", ".flv", ".wmv", ".mkv")
            ):
                video_path = v
                break

        if video_path is None and visuals:
            image_paths = self._save_images_to_disk(visuals, task_name, doc_id)

        # Build user prompt text (reasoning protocol + question).
        user_text = self._user_prompt_template.format(query=context)

        # Messages: swift/scheduler accepts standard chat format. For multimodal
        # we keep text-only in the message body; actual media goes through the
        # top-level `videos` / `images` fields (the scheduler reads those).
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_text},
        ]

        payload_req: Dict[str, Any] = {
            "messages": messages,
            "images": image_paths or [],
            "videos": [video_path] if video_path else [],
            "audios": [],
            "tools": None,
            "objects": {},
            "data_dict": {},
            "uuid": f"{task_name}-{doc_id}-{int(time.time()*1000)}",
        }

        request_config = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "n": 1,
            "stream": False,
            "seed": int(gen_kwargs.get("seed", 42)) if isinstance(gen_kwargs, dict) else 42,
        }

        body = {
            "infer_requests": [payload_req],
            "request_config": request_config,
            "metrics": None,
            "use_tqdm": False,
            "adapter_request": None,
        }

        url = f"{self.rollout_base_url}/infer/"
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=body, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"/infer/ returned {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                break
            except Exception as e:
                logger.warning(
                    f"/infer/ attempt {attempt+1}/{self.max_retries} failed: {e}"
                )
                if attempt == self.max_retries - 1:
                    return f"[ROLLOUT_ERROR] {e}"
                time.sleep(self.retry_backoff_s * (2 ** attempt))

        # Response is a list (one per infer_request); take the first.
        if not isinstance(data, list) or not data:
            return f"[ROLLOUT_ERROR] empty response: {str(data)[:200]}"
        item = data[0]

        # Try RolloutOutput shape first, then fall back to ChatCompletionResponse.
        response_obj = item.get("response") if isinstance(item, dict) else None
        if response_obj is None:
            response_obj = item  # raw ChatCompletionResponse

        try:
            choices = response_obj.get("choices") or []
            if not choices:
                return f"[ROLLOUT_ERROR] no choices: {str(item)[:200]}"
            msg = choices[0].get("message") or {}
            content = msg.get("content", "") or ""
        except Exception as e:
            return f"[ROLLOUT_ERROR] parse failed: {e}; raw={str(item)[:200]}"

        if verbose:
            self._verbose_print(f"\n── [SwiftRollout doc_id={doc_id}] " + "─" * 40)
            self._verbose_print(self._format_content_for_print(content))

        return content if isinstance(content, str) else str(content)
