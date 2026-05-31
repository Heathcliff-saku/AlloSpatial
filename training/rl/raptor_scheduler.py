"""
Raptor-R1 Real-time Tool Calling Scheduler for GRPO Training.

Implements a MultiTurnScheduler that calls the world2mind HTTP service
during GRPO rollout, exactly as done during inference/evaluation.

During each rollout:
1. Model generates text with <tool_call> blocks
2. Scheduler parses tool calls (world2mind / view_image)
3. Calls world2mind HTTP service in real-time (async, non-blocking)
4. Injects tool results into conversation
5. Sets loss_mask=0 for tool result tokens (no gradient)
6. Model continues reasoning until final answer or max_turns

Key optimizations:
- Overrides run() to use asyncio.run_in_executor() for tool HTTP calls,
  so multiple rollouts proceed in parallel.
- AsyncToolResultCache deduplicates concurrent identical tool calls
  (e.g., 8 generations for the same prompt with identical arguments).
- Workspace cleanup after each batch of rollouts.

Requires:
- world2mind service running (start_service.py)
- ms-swift with multi_turn_scheduler support

Register via --external_plugins raptor_scheduler.py

Usage:
    swift rlhf --rlhf_type grpo \
        --external_plugins raptor_scheduler.py \
        --multi_turn_scheduler raptor_tool_scheduler \
        --max_turns 6
"""

import asyncio
import json
import logging
import os
import re
import hashlib
import shutil
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.infer_engine.protocol import RolloutOutput
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import remove_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool call parsing (matches SFT training format: <tool_call>...</tool_call>)
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse <tool_call> blocks from model text output."""
    matches = _TOOL_CALL_PATTERN.findall(text)
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if 'name' in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call JSON: {match[:200]}")
    return tool_calls


# ---------------------------------------------------------------------------
# Thread-safe LRU Cache for tool results
# ---------------------------------------------------------------------------

class ToolResultCache:
    """Thread-safe LRU cache for tool results."""

    def __init__(self, maxsize: int = 512):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def _make_key(self, media_key: str, tool_name: str, arguments: dict) -> str:
        args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        raw = f"{media_key}|{tool_name}|{args_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, media_key: str, tool_name: str, arguments: dict) -> Optional[Any]:
        key = self._make_key(media_key, tool_name, arguments)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, media_key: str, tool_name: str, arguments: dict, result: Any):
        key = self._make_key(media_key, tool_name, arguments)
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Async deduplication wrapper for concurrent identical tool calls
# ---------------------------------------------------------------------------

class AsyncToolResultCache:
    """
    Wraps ToolResultCache to deduplicate concurrent requests for the same key.

    When num_generations=8, all 8 rollouts for the same prompt may call
    world2mind with identical arguments simultaneously. Without dedup,
    all 8 execute the full pipeline. With this wrapper, only the first
    executes; the rest wait for the result via asyncio.Event.

    Also handles restoring tools._last_visualizations from cache so that
    subsequent view_image calls work correctly even for cache-hit rollouts.
    """

    def __init__(self, inner: ToolResultCache):
        self._inner = inner
        self._in_flight: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def get_or_execute(
        self,
        media_key: str,
        tool_name: str,
        arguments: dict,
        executor_fn,
        tools=None,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Get cached result or execute, with concurrent deduplication.

        Returns a 3-tuple: (result_text, vis_type_or_none, workspace_path_or_none).
        For cache hits the workspace was already tracked by the first executor,
        so workspace_path is None to avoid double-tracking.
        """
        key = self._inner._make_key(media_key, tool_name, arguments)

        # Fast path: cache hit (no lock needed)
        cached = self._inner.get(media_key, tool_name, arguments)
        if cached is not None:
            self._restore_visualizations(cached, tools)
            return (cached['result_text'], cached['vis_type'], None)

        # Concurrent deduplication
        async with self._lock:
            cached = self._inner.get(media_key, tool_name, arguments)
            if cached is not None:
                self._restore_visualizations(cached, tools)
                return (cached['result_text'], cached['vis_type'], None)

            if key in self._in_flight:
                event = self._in_flight[key]
            else:
                event = asyncio.Event()
                self._in_flight[key] = event
                event = None

        if event is not None:
            await event.wait()
            cached = self._inner.get(media_key, tool_name, arguments)
            if cached is not None:
                self._restore_visualizations(cached, tools)
                return (cached['result_text'], cached['vis_type'], None)

        try:
            result = await executor_fn()
            # executor_fn returns 3-tuple from _execute_tool_call_sync
            return result
        finally:
            async with self._lock:
                if key in self._in_flight:
                    self._in_flight[key].set()
                    del self._in_flight[key]

    @staticmethod
    def _restore_visualizations(cached: dict, tools) -> None:
        """Restore _last_visualizations on the tools instance from cache."""
        if cached.get('visualizations') and tools is not None:
            tools._last_visualizations = dict(cached['visualizations'])


# ---------------------------------------------------------------------------
# RaptorToolScheduler
# ---------------------------------------------------------------------------

class RaptorToolScheduler(MultiTurnScheduler):
    """
    Multi-turn tool-calling scheduler for Raptor-R1 GRPO training.

    Calls the world2mind HTTP service in real-time during rollout,
    exactly matching the inference-time logic in demo_vllm.py.

    Overrides run() to use asyncio.run_in_executor() for tool calls,
    ensuring multiple rollouts are truly parallel (not serialized by
    synchronous HTTP calls blocking the event loop).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_url = os.environ.get("WORLD2MIND_SERVICE_URL", "http://localhost:9100")
        self._spatial_tools_imported = False
        # Optional multi-port client (training-only; see tools/train_multiproc_service.py)
        self._mp_client = self._maybe_build_multiport_client()
        self._tool_cache = ToolResultCache(maxsize=512)
        self._async_cache = AsyncToolResultCache(self._tool_cache)
        # Workspace cleanup
        self._workspace_dirs: List[str] = []
        self._workspace_lock = threading.Lock()
        self._cleanup_workspace = os.environ.get("RAPTOR_CLEANUP_WORKSPACE", "1") == "1"
        # Per-rollout cleanup (delete workspace as soon as a single rollout
        # finishes, rather than waiting for the whole batch). Required for
        # colocate mode (which has no async_infer hook), recommended whenever
        # disk pressure matters.
        self._cleanup_per_rollout = os.environ.get(
            "RAPTOR_CLEANUP_PER_ROLLOUT", "0") == "1"
        # Per-rollout state map: keyed by id(infer_request). Used by the
        # colocate path (step / check_finished) to track cumulative tool counts,
        # tools instance, workspace dirs, and force-finish flag across turns.
        self._rollout_states: Dict[int, dict] = {}
        self._rollout_states_lock = threading.Lock()
        # Per-tool caps per rollout. Exceeding either cap triggers early stop
        # BEFORE executing the next tool batch.
        # - world2mind: expensive (~30s GPU/call) AND directly incentive-hackable
        #   (bonus stacks with each call). Hard cap = 2.
        # - view_image: cheap (file lookup) but adds ~1024 image tokens per call,
        #   accelerating context overflow. Binary reward so no direct hacking
        #   vulnerability, but a soft cap = 4 as defensive programming.
        self._max_w2m_per_rollout = int(os.environ.get("RAPTOR_MAX_W2M", "2"))
        self._max_view_image_per_rollout = int(
            os.environ.get("RAPTOR_MAX_VIEW_IMAGE", "4"))
        logger.info(f"RaptorToolScheduler initialized, service_url={self.service_url}, "
                     f"cleanup_workspace={self._cleanup_workspace}, "
                     f"cleanup_per_rollout={self._cleanup_per_rollout}, "
                     f"max_w2m_per_rollout={self._max_w2m_per_rollout}, "
                     f"max_view_image_per_rollout={self._max_view_image_per_rollout}")

    def _maybe_build_multiport_client(self):
        """Construct a MultiPortModelClient when WORLD2MIND_MULTIPORT=1.

        Requires WORLD2MIND_BASE_PORT and WORLD2MIND_GPU_IDS. Single-port mode
        (default) returns None — spatial_tools will fall back to ModelServiceClient.
        """
        mp_flag = os.environ.get("WORLD2MIND_MULTIPORT", "0")
        print(f"[RaptorToolScheduler] WORLD2MIND_MULTIPORT={mp_flag!r}", flush=True)
        if mp_flag != "1":
            print("[RaptorToolScheduler] Multi-port DISABLED — using single-port "
                  f"service_url={self.service_url}", flush=True)
            return None
        self._ensure_imports()
        from tools.train_multiproc_client import MultiPortModelClient
        base_port = int(os.environ["WORLD2MIND_BASE_PORT"])
        gpu_ids = [int(x) for x in os.environ["WORLD2MIND_GPU_IDS"].split(",") if x.strip()]
        instances_per_gpu = int(os.environ.get("WORLD2MIND_INSTANCES_PER_GPU", "1"))
        client = MultiPortModelClient(
            base_port=base_port, gpu_ids=gpu_ids, instances_per_gpu=instances_per_gpu)
        print(f"[RaptorToolScheduler] Multi-port ENABLED base_port={base_port} "
              f"gpu_ids={gpu_ids} instances_per_gpu={instances_per_gpu} "
              f"total_ports={len(gpu_ids) * instances_per_gpu}", flush=True)
        return client

    def _ensure_imports(self):
        """Lazy import spatial tools (once per process)."""
        if self._spatial_tools_imported:
            return
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        self._spatial_tools_imported = True

    def _get_media_key(self, infer_request) -> str:
        """Get a unique key for the media input (for caching)."""
        videos = getattr(infer_request, 'videos', []) or []
        images = getattr(infer_request, 'images', []) or []
        if videos:
            return videos[0]
        elif images:
            return "|".join(images[:4])
        data_dict = getattr(infer_request, 'data_dict', {}) or {}
        if 'videos' in data_dict and data_dict['videos']:
            return data_dict['videos'][0] if isinstance(data_dict['videos'], list) else str(data_dict['videos'])
        if 'images' in data_dict and data_dict['images']:
            imgs = data_dict['images']
            return "|".join(imgs[:4]) if isinstance(imgs, list) else str(imgs)
        return "unknown"

    def _get_media_paths(self, infer_request) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract video/image paths from infer_request."""
        videos = getattr(infer_request, 'videos', []) or []
        images = getattr(infer_request, 'images', []) or []
        data_dict = getattr(infer_request, 'data_dict', {}) or {}
        if not videos and 'videos' in data_dict:
            videos = data_dict['videos'] if isinstance(data_dict['videos'], list) else [data_dict['videos']]
        if not images and 'images' in data_dict:
            images = data_dict['images'] if isinstance(data_dict['images'], list) else [data_dict['images']]
        video_path = videos[0] if videos else None
        image_paths = images if images else None
        return video_path, image_paths

    def _create_tools(self, video_path: Optional[str], image_paths: Optional[List[str]]):
        """Create a NEW SpatialIntelligenceTools instance (no sharing, no state conflict)."""
        self._ensure_imports()
        from tools.spatial_tools import SpatialIntelligenceTools
        return SpatialIntelligenceTools(
            service_url=self.service_url,
            video_path=video_path,
            image_paths=image_paths,
            service_client=self._mp_client,  # None in single-port mode
        )

    def _execute_tool_call_sync(
        self,
        tool_call: Dict[str, Any],
        tools,
        media_key: str,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Execute a single tool call (synchronous, runs in thread pool).

        Cache entries store a dict with keys:
            result_text, vis_type, visualizations (vis path mapping for world2mind)

        Returns:
            (result_text, vis_type_or_none, workspace_path_or_none)
            workspace_path is None on cache hit (already tracked by first executor).
        """
        from tools.spatial_tools import handle_tool_call

        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        # Check cache first
        cached = self._tool_cache.get(media_key, tool_name, arguments)
        if cached is not None:
            logger.debug(f"Cache hit: {tool_name} for {media_key}")
            # Restore _last_visualizations so subsequent view_image calls work
            if cached.get('visualizations') and tools is not None:
                tools._last_visualizations = dict(cached['visualizations'])
            return (cached['result_text'], cached['vis_type'], None)

        try:
            result = handle_tool_call(tool_name, arguments, tools)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            result = json.dumps({"error": f"Tool execution failed: {e}"})

        # Process result
        if isinstance(result, dict) and result.get("type") == "image":
            vis_type = result.get("visualization_type", "unknown")
            result_text = json.dumps({
                "visualization_type": vis_type,
                "description": f"Visualization '{vis_type}' generated successfully. "
                               f"The bird's-eye view shows the spatial layout of detected objects.",
            })
        else:
            vis_type = None
            if isinstance(result, dict):
                result_text = json.dumps(result, ensure_ascii=False)
            else:
                result_text = str(result)

        # Build cache entry with visualization paths for world2mind calls
        vis_map = None
        if tool_name == "world2mind" and tools is not None:
            vis_map = dict(getattr(tools, '_last_visualizations', {}) or {})

        cache_entry = {
            'result_text': result_text,
            'vis_type': vis_type,
            'visualizations': vis_map,
        }
        self._tool_cache.put(media_key, tool_name, arguments, cache_entry)

        # Track workspace directory for cleanup. Always returned to caller so
        # both run() (server) and step() (colocate) can route it to the
        # appropriate per-rollout / batch-level list.
        workspace_path: Optional[str] = None
        if tool_name == "world2mind":
            workspace_path = self._resolve_workspace_path(result, tools)
            if workspace_path is not None and not self._cleanup_per_rollout:
                with self._workspace_lock:
                    self._workspace_dirs.append(workspace_path)

        return (result_text, vis_type, workspace_path)

    def _resolve_workspace_path(self, result, tools) -> Optional[str]:
        """Return the workspace directory path from a world2mind result, if any."""
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            scene_id = parsed.get("scene_id", "") if isinstance(parsed, dict) else ""
            if scene_id and tools is not None and hasattr(tools, 'output_base'):
                ws = os.path.join(str(tools.output_base), scene_id)
                if os.path.isdir(ws):
                    return ws
        except Exception:
            pass
        return None

    @staticmethod
    def _cleanup_paths(paths: List[str]) -> int:
        """Delete a list of workspace directories. Returns number removed."""
        removed = 0
        for d in set(paths):
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup workspace {d}: {e}")
        return removed

    def _merge_messages_for_logging(self, messages):
        """Merge multi-turn assistant + tool messages into a single assistant
        message so reward functions and wandb logging see the full chain.

        Uses structured markers (Turn / Tool Result) so logs are readable
        even when truncated. Returns (new_messages, merged_content_or_None).
        If there are no completion parts to merge, messages is returned unchanged
        and merged_content is None.
        """
        completion_parts = []
        prefix_messages = []
        found_first_assistant = False
        turn_counter = 0
        all_merged_tools = []
        for msg in messages:
            if msg['role'] in ('assistant', 'tool') and msg.get('content'):
                if msg['role'] == 'assistant':
                    turn_counter += 1
                    completion_parts.append(
                        f"\n--- [Turn {turn_counter}: Assistant] ---\n{msg['content']}")
                else:
                    completion_parts.append(
                        f"\n--- [Tool Result] ---\n{msg['content']}")
                found_first_assistant = True
            elif not found_first_assistant:
                prefix_messages.append(msg)
            # skip system/user messages after first assistant

        if not completion_parts:
            return messages, None

        for msg in messages:
            if msg['role'] == 'assistant' and msg.get('content'):
                for tc in parse_tool_calls(msg['content']):
                    all_merged_tools.append(tc.get('name', 'unknown'))
        summary = f"[Rollout: {turn_counter} turns, tools: {all_merged_tools}]"
        merged = summary + "\n".join(completion_parts)
        new_messages = prefix_messages + [{'role': 'assistant', 'content': merged}]
        return new_messages, merged

    # -----------------------------------------------------------------------
    # Override run() for async tool execution
    # -----------------------------------------------------------------------

    async def run(self, infer_request, request_config, **kwargs):
        """
        Multi-turn rollout with async tool calling.

        Overrides MultiTurnScheduler.run() to wrap synchronous tool HTTP calls
        in asyncio.run_in_executor(), so they don't block the event loop and
        multiple rollout coroutines can run tool calls in parallel.
        """
        current_request = infer_request
        current_turn = 1
        rollout_infos = {}
        total_response_ids = []
        total_response_loss_mask = []
        total_rollout_logprobs = []

        # Track cumulative tool calls across all turns. Once either cap is
        # reached, we stop the rollout before executing the next tool batch
        # (instead of wasting GPU / accumulating context on repeated bad calls).
        cumulative_w2m_count = 0
        cumulative_view_image_count = 0

        # Create a fresh tools instance per rollout (avoids state conflicts)
        media_key = self._get_media_key(infer_request)
        video_path, image_paths = self._get_media_paths(infer_request)
        # Per-rollout workspace tracking for per-rollout cleanup mode.
        rollout_workspace_dirs: List[str] = []

        # Only create tools when we actually need them (lazy)
        tools = None

        # State from the previous turn, used to build a graceful early-stop
        # RolloutOutput if the model's context overflows on a subsequent turn.
        last_response = None

        while True:
            messages = current_request.messages
            if current_turn == 1:
                remove_response(messages)

            # Get model response (async, non-blocking).
            # On turn > 1, the accumulated prompt (prev turns + tool results) can
            # exceed vllm's max_model_len and raise ValueError. Catch that and
            # finalize the rollout using the previous turn's state instead of
            # crashing the whole async_infer batch.
            try:
                response = await self.infer_engine.infer_async(
                    current_request, request_config, **kwargs)
            except Exception as e:
                msg = str(e)
                is_overflow = isinstance(e, ValueError) and (
                    "longer than the maximum model length" in msg
                    or ("decoder prompt" in msg and "longer than" in msg)
                )
                # Data-level failure inside vLLM template / media pipeline,
                # e.g. KeyError('video_fps') when torchcodec falls back to
                # torchvision, or broken video metadata. Don't crash the batch:
                # return an empty rollout so training skips this sample.
                is_data_error = (
                    isinstance(e, (KeyError, RuntimeError, OSError))
                    or ("video_fps" in msg)
                    or ("ffmpeg" in msg.lower())
                    or ("Resource temporarily unavailable" in msg)
                )
                if current_turn > 1 and is_overflow and last_response is not None:
                    logger.warning(
                        f"Context overflow at turn {current_turn}, stopping rollout "
                        f"early (reusing turn {current_turn - 1} state): {e}")
                    # Drop the trailing tool message(s) appended after the last
                    # assistant turn — they caused the overflow and the model
                    # never produced a response consuming them.
                    while messages and messages[-1].get('role') == 'tool':
                        messages.pop()
                    # Merge and return based on already-collected state.
                    merged_messages, merged_content = (
                        self._merge_messages_for_logging(messages))
                    if merged_content is not None and last_response.choices:
                        last_response.choices[0].message.content = merged_content
                    if self._cleanup_per_rollout and rollout_workspace_dirs:
                        n = self._cleanup_paths(rollout_workspace_dirs)
                        logger.debug(f"Per-rollout cleanup (overflow): {n} dirs")
                    return RolloutOutput(
                        response=last_response,
                        messages=merged_messages,
                        response_token_ids=total_response_ids,
                        response_loss_mask=total_response_loss_mask,
                        rollout_infos={
                            **rollout_infos,
                            'num_turns': current_turn - 1,
                            'early_stop_reason': 'context_overflow',
                        },
                        rollout_logprobs=total_rollout_logprobs,
                    )
                if is_data_error or current_turn == 1:
                    logger.warning(
                        f"infer_async failed at turn {current_turn} "
                        f"(type={type(e).__name__}): {e}. "
                        f"Returning empty rollout for this sample.")
                    if self._cleanup_per_rollout and rollout_workspace_dirs:
                        try:
                            self._cleanup_paths(rollout_workspace_dirs)
                        except Exception:
                            pass
                    # Build a minimal synthetic response so downstream code
                    # doesn't have to special-case None. Reuse last_response if
                    # present, otherwise build a fresh empty ChatCompletion.
                    fallback_response = last_response
                    if fallback_response is None:
                        try:
                            from swift.infer_engine.protocol import (
                                ChatCompletionResponse, ChatCompletionResponseChoice,
                                ChatMessage, UsageInfo)
                            fallback_response = ChatCompletionResponse(
                                model="",
                                choices=[ChatCompletionResponseChoice(
                                    index=0,
                                    message=ChatMessage(role="assistant", content=""),
                                    finish_reason="stop",
                                    token_ids=[],
                                )],
                                usage=UsageInfo(
                                    prompt_tokens=0, completion_tokens=0, total_tokens=0),
                            )
                        except Exception:
                            fallback_response = None
                    return RolloutOutput(
                        response=fallback_response,
                        messages=messages,
                        response_token_ids=total_response_ids or [[]],
                        response_loss_mask=total_response_loss_mask or [[]],
                        rollout_infos={
                            **rollout_infos,
                            'num_turns': current_turn - 1,
                            'early_stop_reason': f'data_error:{type(e).__name__}',
                        },
                        rollout_logprobs=total_rollout_logprobs,
                    )
                raise
            response_choice = response.choices[0]
            last_response = response

            if current_turn > 1 and not messages[-1]['content']:
                remove_response(messages)

            # Update conversation history
            completion = response_choice.message.content
            is_continuation = False
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
                is_continuation = True
            else:
                messages.append({'role': 'assistant', 'content': completion})

            # Parse tool calls from completion
            tool_calls = parse_tool_calls(completion)

            # Count per-tool calls in this turn
            w2m_in_this_turn = sum(
                1 for tc in tool_calls if tc.get('name') == 'world2mind')
            vi_in_this_turn = sum(
                1 for tc in tool_calls if tc.get('name') == 'view_image')

            # Unified per-tool cap: stop rollout if executing this turn's tool
            # calls would exceed either per-rollout limit. Assistant message is
            # already appended so reward functions see the offending tool_call
            # text (and apply overuse penalty where applicable).
            w2m_overflow = (
                cumulative_w2m_count + w2m_in_this_turn
                > self._max_w2m_per_rollout)
            vi_overflow = (
                cumulative_view_image_count + vi_in_this_turn
                > self._max_view_image_per_rollout)
            if w2m_overflow:
                rollout_infos['early_stop_reason'] = 'w2m_limit'
                logger.debug(
                    f"W2M limit reached (cum={cumulative_w2m_count}, "
                    f"this_turn={w2m_in_this_turn}, max={self._max_w2m_per_rollout}), "
                    f"stopping rollout at turn {current_turn}")
            elif vi_overflow:
                rollout_infos['early_stop_reason'] = 'view_image_limit'
                logger.debug(
                    f"view_image limit reached (cum={cumulative_view_image_count}, "
                    f"this_turn={vi_in_this_turn}, max={self._max_view_image_per_rollout}), "
                    f"stopping rollout at turn {current_turn}")

            # Check stopping: no tool calls, max turns, or tool budget overflow
            should_stop = (not tool_calls) or w2m_overflow or vi_overflow
            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            if should_stop:
                # Final turn: collect token data
                current_logprobs = self._extract_logprobs_from_choice(response_choice)
                final_token_ids = response_choice.token_ids

                if is_continuation and total_response_ids:
                    total_response_ids[-1].extend(final_token_ids)
                    if total_response_loss_mask:
                        total_response_loss_mask[-1].extend([1] * len(final_token_ids))
                    if total_rollout_logprobs and current_logprobs:
                        total_rollout_logprobs[-1].extend(current_logprobs)
                else:
                    # New turn (not continuation) or first turn
                    if final_token_ids:
                        total_response_ids.append(list(final_token_ids))
                        total_response_loss_mask.append([1] * len(final_token_ids))
                    if current_logprobs:
                        total_rollout_logprobs.append(current_logprobs)

                # Validate logprobs completeness
                final_rollout_logprobs = total_rollout_logprobs
                if total_rollout_logprobs:
                    total_logprob_count = sum(len(lps) for lps in total_rollout_logprobs)
                    if total_response_loss_mask:
                        total_loss_mask_1_count = sum(sum(mask) for mask in total_response_loss_mask)
                        if total_loss_mask_1_count != total_logprob_count:
                            final_rollout_logprobs = []
                    else:
                        if total_response_ids:
                            total_id_count = sum(len(ids) for ids in total_response_ids)
                            if total_id_count != total_logprob_count:
                                final_rollout_logprobs = []
                        else:
                            final_rollout_logprobs = []

                # Merge all completion parts (assistant + tool results) into a
                # single assistant message so reward functions (which read
                # messages[-1]['content']) see the full multi-turn chain.
                if current_turn > 1:
                    messages, merged_content = self._merge_messages_for_logging(messages)
                    if merged_content is not None:
                        response.choices[0].message.content = merged_content

                if self._cleanup_per_rollout and rollout_workspace_dirs:
                    n = self._cleanup_paths(rollout_workspace_dirs)
                    logger.debug(f"Per-rollout cleanup: {n} dirs")

                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos={**rollout_infos, 'num_turns': current_turn},
                    rollout_logprobs=final_rollout_logprobs,
                )

            # --- Tool execution (async, non-blocking) ---

            # Lazy-create tools instance on first actual tool call
            if tools is None:
                loop = asyncio.get_running_loop()
                tools = await loop.run_in_executor(
                    None, self._create_tools, video_path, image_paths)

            token_ids = list(response_choice.token_ids)
            loss_mask = [1] * len(token_ids)
            all_tool_names = []
            tool_result_text_parts = []

            # Update cumulative counts (this turn will be executed since neither
            # overflow was true — otherwise we would have stopped above)
            cumulative_w2m_count += w2m_in_this_turn
            cumulative_view_image_count += vi_in_this_turn

            # Execute tool calls via async cache (deduplicates concurrent identical calls)
            loop = asyncio.get_running_loop()
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                all_tool_names.append(tool_name)

                # Use async cache for deduplication across concurrent generations
                async def _exec(_tc=tc, _tools=tools, _mk=media_key):
                    return await loop.run_in_executor(
                        None, self._execute_tool_call_sync, _tc, _tools, _mk)

                result_text, vis_type, ws_path = await self._async_cache.get_or_execute(
                    media_key, tool_name, tc.get("arguments", {}), _exec, tools=tools)
                tool_result_text_parts.append(result_text)
                if ws_path is not None and self._cleanup_per_rollout:
                    rollout_workspace_dirs.append(ws_path)

                # Append tool result to conversation
                current_request.messages.append({
                    "role": "tool",
                    "content": result_text,
                })

            # Tokenize tool result text for loss masking
            tokenizer = self.tokenizer
            combined_tool_text = "\n".join(tool_result_text_parts)
            if tokenizer is not None:
                result_tokens = tokenizer.encode(combined_tool_text, add_special_tokens=False)
                token_ids.extend(result_tokens)
                loss_mask.extend([0] * len(result_tokens))

            # Track response tokens and masks
            if is_continuation and total_response_ids:
                total_response_ids[-1].extend(token_ids)
            else:
                total_response_ids.append(token_ids)

            if is_continuation and total_response_loss_mask:
                total_response_loss_mask[-1].extend(loss_mask)
            else:
                total_response_loss_mask.append(loss_mask)

            # Track logprobs
            current_logprobs = self._extract_logprobs_from_choice(response_choice)
            if current_logprobs:
                if is_continuation and total_rollout_logprobs:
                    total_rollout_logprobs[-1].extend(current_logprobs)
                else:
                    total_rollout_logprobs.append(current_logprobs)

            # Update rollout infos
            rollout_infos.update({
                'tool_names': all_tool_names,
                'tool_call_count': len(tool_calls),
                'num_turns': current_turn,
                'w2m_count': cumulative_w2m_count,
                'view_image_count': cumulative_view_image_count,
            })

            # Prepare next turn
            if current_request.messages[-1]['role'] == 'assistant':
                current_request.messages.append({'role': 'assistant', 'content': None})

            current_turn += 1

    # -----------------------------------------------------------------------
    # Workspace cleanup
    # -----------------------------------------------------------------------

    def _cleanup_workspaces(self) -> None:
        """Remove workspace directories created during rollouts.

        Called after async_infer completes (all rollouts in a batch are done),
        so view_image files are no longer needed.
        """
        if not self._cleanup_workspace:
            return
        with self._workspace_lock:
            dirs = list(set(self._workspace_dirs))
            self._workspace_dirs.clear()
        removed = 0
        for d in dirs:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup workspace {d}: {e}")
        if removed:
            logger.info(f"Cleaned {removed} workspace directories")

    async def async_infer(self, infer_requests, request_config, *, use_tqdm=None, **kwargs):
        """Override to add workspace cleanup after all rollouts complete."""
        results = await super().async_infer(
            infer_requests, request_config, use_tqdm=use_tqdm, **kwargs)
        self._cleanup_workspaces()
        return results

    # -----------------------------------------------------------------------
    # check_finished() / step(): colocate-mode multi-turn path.
    #
    # ms-swift's _colocate_multi_turn_infer (rollout_mixin.py) calls
    # check_finished() and step() per-sample per-turn instead of using run().
    # We implement the same semantics as run() here:
    #   - per-rollout state (cumulative tool counts, tools instance, workspace)
    #   - tool budget enforcement (max_w2m / max_view_image)
    #   - async tool execution with AsyncToolResultCache deduplication
    #   - message merge for wandb / reward (when finishing)
    #   - per-rollout workspace cleanup (when finishing)
    # -----------------------------------------------------------------------

    def _get_state(self, infer_request) -> dict:
        """Get-or-init per-rollout state, keyed by id(infer_request).

        infer_request lifetime spans the entire multi-turn rollout (held in the
        colocate loop's `requests` list), so id() is stable for the rollout.
        """
        key = id(infer_request)
        with self._rollout_states_lock:
            state = self._rollout_states.get(key)
            if state is None:
                video_path, image_paths = self._get_media_paths(infer_request)
                state = {
                    'media_key': self._get_media_key(infer_request),
                    'video_path': video_path,
                    'image_paths': image_paths,
                    'tools': None,
                    'w2m_count': 0,
                    'view_image_count': 0,
                    'workspace_dirs': [],
                    'finalized': False,
                    'force_finish': False,
                    'early_stop_reason': None,
                    'tool_names': [],
                }
                self._rollout_states[key] = state
            return state

    def _drop_state(self, infer_request) -> None:
        key = id(infer_request)
        with self._rollout_states_lock:
            self._rollout_states.pop(key, None)

    def _finalize_rollout(self, infer_request, state, reason: str) -> None:
        """Merge messages for wandb/reward + per-rollout workspace cleanup.

        Idempotent: safe to call from check_finished and step independently.
        """
        if state.get('finalized'):
            return
        state['finalized'] = True
        # Merge multi-turn assistant + tool messages so the trainer-side
        # log_completions (`inputs[i]['messages'][-1]['content']`) sees the
        # full chain. Reward functions also read this last-content slot.
        merged_messages, merged_content = self._merge_messages_for_logging(
            infer_request.messages)
        if merged_content is not None:
            # In-place replace so the colocate loop's reference still works.
            infer_request.messages.clear()
            infer_request.messages.extend(merged_messages)
        # Per-rollout workspace cleanup
        if self._cleanup_per_rollout and state['workspace_dirs']:
            n = self._cleanup_paths(state['workspace_dirs'])
            logger.debug(
                f"[rollout={state['media_key'][:24]}] cleaned {n} workspace dirs")
        logger.info(
            f"[rollout={state['media_key'][:24]}] finalized "
            f"reason={reason} w2m={state['w2m_count']} "
            f"view_image={state['view_image_count']} "
            f"tools={state['tool_names']}")
        # Drop state so id(infer_request) doesn't leak across rollouts
        # if Python recycles the object id.
        self._drop_state(infer_request)

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        """Decide whether this rollout should stop after the current turn.

        Mirrors the should_stop logic in run() (no tool calls / max_turns /
        per-tool budget overflow). When returning True, finalizes the
        rollout (merge messages + cleanup workspace).
        """
        state = self._get_state(infer_request)
        if state['force_finish']:
            self._finalize_rollout(
                infer_request, state, reason=state.get('early_stop_reason') or 'forced')
            return True

        completion = response_choice.message.content or ""
        tool_calls = parse_tool_calls(completion)

        if not tool_calls:
            self._finalize_rollout(infer_request, state, reason='no_tool_calls')
            return True

        # Pre-check budgets: same semantics as run() lines 525-545. We stop
        # BEFORE executing this turn's tool calls if either cap would be
        # exceeded — the assistant message is already appended (by the colocate
        # loop) so reward functions still see the offending tool_call text.
        w2m_in_turn = sum(
            1 for tc in tool_calls if tc.get('name') == 'world2mind')
        vi_in_turn = sum(
            1 for tc in tool_calls if tc.get('name') == 'view_image')
        if state['w2m_count'] + w2m_in_turn > self._max_w2m_per_rollout:
            state['force_finish'] = True
            state['early_stop_reason'] = 'w2m_limit'
            self._finalize_rollout(infer_request, state, reason='w2m_limit')
            return True
        if state['view_image_count'] + vi_in_turn > self._max_view_image_per_rollout:
            state['force_finish'] = True
            state['early_stop_reason'] = 'view_image_limit'
            self._finalize_rollout(infer_request, state, reason='view_image_limit')
            return True

        if self.max_turns and current_turn >= self.max_turns:
            self._finalize_rollout(infer_request, state, reason='max_turns')
            return True

        return False

    def step(self, infer_request, response_choice, current_turn) -> Dict:
        """Execute tool calls for the colocate-mode multi-turn loop.

        Called only when check_finished returned False, i.e. tool_calls are
        present and budgets allow execution. Uses asyncio.run() to leverage
        AsyncToolResultCache for in-call dedup (and cross-sample sync dedup
        via the inner ToolResultCache).
        """
        state = self._get_state(infer_request)
        completion = response_choice.message.content or ""
        token_ids = list(response_choice.token_ids)
        loss_mask = [1] * len(token_ids)

        tool_calls = parse_tool_calls(completion)
        if not tool_calls:
            # check_finished should have caught this, but stay defensive.
            return {
                'infer_request': infer_request,
                'response_token_ids': token_ids,
                'response_loss_mask': loss_mask,
            }

        # Lazy-create tools instance per rollout (held on state across turns).
        if state['tools'] is None:
            state['tools'] = self._create_tools(
                state['video_path'], state['image_paths'])
        tools = state['tools']
        media_key = state['media_key']

        # Execute tool calls via async cache. asyncio.run is safe here because
        # _colocate_multi_turn_infer is a synchronous loop (no outer event loop).
        tool_result_text_parts: List[str] = []
        all_tool_names: List[str] = []

        async def _exec_all():
            results = []
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                all_tool_names.append(tool_name)

                async def _exec(_tc=tc, _tools=tools, _mk=media_key):
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, self._execute_tool_call_sync, _tc, _tools, _mk)

                result_text, vis_type, ws_path = await self._async_cache.get_or_execute(
                    media_key, tool_name, tc.get("arguments", {}), _exec, tools=tools)
                results.append((result_text, ws_path))
            return results

        exec_results = asyncio.run(_exec_all())
        for result_text, ws_path in exec_results:
            tool_result_text_parts.append(result_text)
            infer_request.messages.append({
                "role": "tool",
                "content": result_text,
            })
            if ws_path is not None:
                state['workspace_dirs'].append(ws_path)

        # Update cumulative tool counts on state (used by next check_finished).
        state['w2m_count'] += sum(
            1 for n in all_tool_names if n == 'world2mind')
        state['view_image_count'] += sum(
            1 for n in all_tool_names if n == 'view_image')
        state['tool_names'].extend(all_tool_names)

        # Tokenize tool result text for loss masking
        tokenizer = self.tokenizer
        combined_tool_text = "\n".join(tool_result_text_parts)
        if tokenizer is not None:
            result_tokens = tokenizer.encode(combined_tool_text, add_special_tokens=False)
            token_ids.extend(result_tokens)
            loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_names': list(state['tool_names']),
                'tool_call_count': len(tool_calls),
                'num_turns': current_turn,
                'w2m_count': state['w2m_count'],
                'view_image_count': state['view_image_count'],
            },
        }


# ---------------------------------------------------------------------------
# Register scheduler
# ---------------------------------------------------------------------------

multi_turns['raptor_tool_scheduler'] = RaptorToolScheduler
