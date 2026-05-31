"""
Raptor-R1 inline evaluation — swift-native.

Design
------
1. ``--val_dataset grpo_val_tiny.jsonl`` holds MindCube-tiny + VSI-Bench-tiny
   samples tagged with ``task``, ``question_type``, ``val_id``.
2. Every ``--eval_steps``, swift runs ``evaluation_loop`` which feeds each val
   sample through the same multi-turn rollout (vLLM + raptor_tool_scheduler).
3. A zero-weight reward function ``raptor_val_collector`` runs on every batch,
   filters to val rows (rows with a ``task`` key), and stashes per-sample
   scores in a module-level bucket.
4. ``EvalMetricsCallback.on_evaluate`` aggregates the bucket into
   MindCube (overall + rotation/around/among) and VSI-Bench (per question-type
   + overall) metrics, logs them to wandb, and clears the bucket.

Rationale vs. the prior lmms-eval-subprocess approach
-----------------------------------------------------
* No subprocess — no env-var scrubbing, no cross-env python, no shell-out.
* No additional vLLM contention — eval reuses the same rollout pipeline.
* No baseline-eval hang — rank-0 never blocks; scoring happens in the
  reward-func path which swift already gathers across ranks.

Activation:  ``RAPTOR_INLINE_EVAL=1`` (default 1; set to 0 to disable).
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:  # pragma: no cover
    TrainerCallback = object  # type: ignore

try:
    from swift.rewards import ORM, orms
except ImportError:
    ORM, orms = object, None  # type: ignore


# ---------------------------------------------------------------------------
# VSI-Bench question-type sets (kept in sync with lmms-eval/vsibench/utils.py)
# ---------------------------------------------------------------------------

VSI_MCA = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
VSI_NA = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


# ---------------------------------------------------------------------------
# Shared, process-local bucket (single training process writes to it; callback
# reads from it in on_evaluate). Protected by a lock because swift may call
# reward functions from worker threads.
# ---------------------------------------------------------------------------

_BUCKET_LOCK = threading.Lock()
_EVAL_BUCKET: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Answer extractors
# ---------------------------------------------------------------------------

_ANSWER_TAG_STRICT = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.DOTALL)
_ANSWER_TAG_LOOSE = re.compile(
    r"<Answer>\s*(.*?)\s*</Answer>", re.IGNORECASE | re.DOTALL
)


def _find_answer_tag(text: str):
    """Hybrid <Answer>...</Answer> search.

    Strict (case-sensitive) first to keep parity with lmms-eval; falls back
    to IGNORECASE only when the strict pattern misses, so model drift to
    lowercase <answer> still scores instead of silently dropping to 0.
    """
    if not text:
        return None
    m = _ANSWER_TAG_STRICT.search(text)
    if m is not None:
        return m
    return _ANSWER_TAG_LOOSE.search(text)


def _extract_mindcube_letter(text: str) -> Optional[str]:
    """Reimpl of lmms-eval/mindcube/utils.py::extract_answer (A-E).

    Mirrors the reference step-by-step: <Answer> tag → dedicated `X.`
    last-match pass → 8-pattern priority list → line-by-line stage-1 →
    bare \\b[A-E]\\b bottom-up fallback.
    """
    if not text:
        return None

    m = _find_answer_tag(text)
    if m is not None:
        content = m.group(1).strip()
        letter = re.search(r"\b([A-E])\b", content)
        if letter:
            return letter.group(1)

    simple = list(re.finditer(r"([A-E])\.", text))
    if simple:
        return simple[-1].group(1)

    for pat in (
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',
        r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)",
        r"[\*\"]([A-E])[\*\"]",
        r"\bAnswer:?\s*([A-E])\b",
        r"[Mm]y answer is ([A-E])",
        r"[Mm]y answer is ([A-E])\.",
        r"answer is ([A-E])",
    ):
        matches = list(re.finditer(pat, text))
        if matches:
            return matches[-1].group(1)

    lines = text.split("\n")
    line_matches: List[tuple] = []
    for i, line in enumerate(lines):
        m2 = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if m2:
            line_matches.append((i, m2.group(1)))
    if line_matches:
        return line_matches[-1][1]

    for line in reversed(lines):
        m2 = re.search(r"\b([A-E])\b", line)
        if m2:
            return m2.group(1)
    return None


def _vsi_fuzzy(pred: str) -> str:
    """Reimpl of lmms-eval/vsibench/utils.py::fuzzy_matching with hybrid
    <Answer> tag (strict, then IGNORECASE fallback)."""
    if not pred:
        return ""
    m = _find_answer_tag(pred)
    raw = m.group(1).strip() if m is not None else pred
    return raw.split(" ")[0].rstrip(".").strip()


def _to_float(s) -> Optional[float]:
    """Verbatim port of lmms-eval/vsibench/utils.py::to_float.

    No comma-stripping: '1,250' → None, matching lmms-eval exactly.
    """
    try:
        return float(s)
    except BaseException:
        return None


def _mra(pred: float, target: float, start: float = 0.5,
         end: float = 0.95, interval: float = 0.05) -> float:
    """Mean Relative Accuracy (VSI NA metric).

    Verbatim port of lmms-eval/vsibench/utils.py::mean_relative_accuracy +
    abs_dist_norm. Uses target (signed) in the denominator to mirror the
    reference exactly. Explicit zero-guard short-circuits the
    div-by-zero RuntimeWarning lmms-eval would emit (both still return 0).
    """
    if target == 0:
        return 0.0
    num_pts = int((end - start) / interval + 2)
    if num_pts < 2:
        num_pts = 2
    step = (end - start) / (num_pts - 1)
    thresholds = [start + i * step for i in range(num_pts)]
    rel = abs(pred - target) / target
    return sum(1.0 for t in thresholds if rel <= (1.0 - t)) / len(thresholds)


# ---------------------------------------------------------------------------
# MindCube category from id (matches lmms-eval/mindcube/utils.py:154-164)
# ---------------------------------------------------------------------------

def _mindcube_type(val_id: str) -> str:
    first = val_id.split("_", 1)[0]
    if first == "among":
        return "among"
    if first == "rotation":
        return "rotation"
    if first in ("around", "aroundnew"):
        return "around"
    return "other"


# ---------------------------------------------------------------------------
# Reward function: val metric collector (weight 0)
# ---------------------------------------------------------------------------

class ValMetricsCollectorReward(ORM):  # type: ignore[misc]
    """
    Zero-weight reward that collects per-sample eval metrics from val rows.

    Val rows carry a ``task`` field ("mindcube_tiny" or "vsibench_tiny"); train
    rows don't. Non-val batches short-circuit to zeros.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        n = len(completions)
        tasks = kwargs.get("task") or [None] * n
        # If the first entry is None (or missing), everyone is train: fast return
        if not any(t for t in tasks):
            return [0.0] * n

        solutions = kwargs.get("solution") or [""] * n
        qtypes = kwargs.get("question_type") or [""] * n
        sources = kwargs.get("source") or [""] * n
        val_ids = kwargs.get("val_id") or [""] * n

        scores: List[float] = [0.0] * n
        with _BUCKET_LOCK:
            for i, comp in enumerate(completions):
                task = tasks[i] if i < len(tasks) else None
                if not task:
                    continue
                solution = solutions[i] if i < len(solutions) else ""
                qt = qtypes[i] if i < len(qtypes) else ""
                source = sources[i] if i < len(sources) else ""
                vid = val_ids[i] if i < len(val_ids) else ""

                try:
                    if source == "mindcube":
                        pred = _extract_mindcube_letter(comp)
                        s = 1.0 if pred == solution else 0.0
                        _EVAL_BUCKET[task].append({
                            "score": s,
                            "mc_type": _mindcube_type(vid),
                        })
                        scores[i] = s
                    elif source == "vsibench":
                        if qt in VSI_MCA:
                            pred = _vsi_fuzzy(comp)
                            s = 1.0 if pred.lower() == solution.lower() else 0.0
                            _EVAL_BUCKET[task].append({
                                "score": s,
                                "question_type": qt,
                                "metric": "accuracy",
                            })
                        elif qt in VSI_NA:
                            pred_f = _to_float(_vsi_fuzzy(comp))
                            sol_f = _to_float(solution)
                            if pred_f is None or sol_f is None:
                                s = 0.0
                            else:
                                s = _mra(pred_f, sol_f)
                            _EVAL_BUCKET[task].append({
                                "score": s,
                                "question_type": qt,
                                "metric": "MRA",
                            })
                        else:
                            s = 0.0
                        scores[i] = s
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[ValCollector] scoring failed ({source}/{qt}): {e}")
        return scores


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_mindcube(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    out: Dict[str, float] = {}
    scores = [r["score"] for r in rows]
    out["overall_accuracy"] = sum(scores) / len(scores)
    for cat in ("rotation", "around", "among"):
        cat_scores = [r["score"] for r in rows if r["mc_type"] == cat]
        if cat_scores:
            out[f"{cat}_accuracy"] = sum(cat_scores) / len(cat_scores)
    return out


def _aggregate_vsibench(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    by_qt: Dict[str, List[float]] = defaultdict(list)
    metric_name: Dict[str, str] = {}
    for r in rows:
        by_qt[r["question_type"]].append(r["score"])
        metric_name[r["question_type"]] = (
            "accuracy" if r["metric"] == "accuracy" else "MRA:.5:.95:.05"
        )

    per_type: Dict[str, float] = {qt: sum(v) / len(v) for qt, v in by_qt.items()}

    # Merge object_rel_direction_{easy,medium,hard} → object_rel_direction
    dir_subs = [k for k in list(per_type.keys()) if k.startswith("object_rel_direction_")]
    if dir_subs:
        vals = [per_type.pop(k) for k in dir_subs]
        per_type["object_rel_direction"] = sum(vals) / len(vals)
        for k in dir_subs:
            metric_name.pop(k, None)
        metric_name["object_rel_direction"] = "accuracy"

    out: Dict[str, float] = {}
    for qt, v in per_type.items():
        suffix = metric_name.get(qt, "score")
        out[f"{qt}_{suffix}"] = v
    out["overall"] = sum(per_type.values()) / len(per_type) if per_type else 0.0
    return out


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------

class EvalMetricsCallback(TrainerCallback):  # type: ignore[misc]
    """
    On ``on_evaluate``, read ``_EVAL_BUCKET``, compute MindCube / VSI-Bench
    metrics, log to wandb (rank-0 only), and clear the bucket.
    """

    def __init__(self, only_rank_zero: bool = True):
        self.only_rank_zero = only_rank_zero

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401
        if self.only_rank_zero:
            local_rank = getattr(args, "local_rank", -1)
            if local_rank not in (-1, 0):
                return

        with _BUCKET_LOCK:
            bucket = {k: list(v) for k, v in _EVAL_BUCKET.items()}
            _EVAL_BUCKET.clear()

        if not bucket:
            logger.info("[EvalMetrics] bucket empty, nothing to log")
            return

        agg_results: Dict[str, Dict[str, float]] = {}
        for task, rows in bucket.items():
            if task == "mindcube_tiny":
                agg_results[task] = _aggregate_mindcube(rows)
            elif task == "vsibench_tiny":
                agg_results[task] = _aggregate_vsibench(rows)
            else:
                scores = [r.get("score", 0.0) for r in rows]
                agg_results[task] = (
                    {"overall": sum(scores) / len(scores)} if scores else {}
                )

        metrics: Dict[str, float] = {}
        for task, m in agg_results.items():
            for k, v in m.items():
                metrics[f"eval/{task}/{k}"] = v
                if ":" in k:
                    metrics[f"eval/{task}/{k.replace(':', '_')}"] = v

        mc_overall = agg_results.get("mindcube_tiny", {}).get("overall_accuracy")
        vs_overall = agg_results.get("vsibench_tiny", {}).get("overall")
        if mc_overall is not None:
            metrics["eval/summary/mindcube_tiny_overall"] = mc_overall
        if vs_overall is not None:
            metrics["eval/summary/vsibench_tiny_overall"] = vs_overall

        if not metrics:
            return

        try:
            import wandb
            if wandb.run is not None:
                # Define a custom step axis the FIRST time we log. Without this,
                # passing step=state.global_step to wandb.log would be silently
                # dropped on resumed runs whose internal step counter has
                # already advanced past state.global_step (the well-known
                # "Tried to log to step X that is less than current step Y"
                # warning). See https://wandb.me/define-metric.
                if not getattr(self, "_wandb_metric_defined", False):
                    wandb.define_metric("eval/step")
                    wandb.define_metric("eval/*", step_metric="eval/step")
                    self._wandb_metric_defined = True
                wandb.log({**metrics, "eval/step": state.global_step})
        except ImportError:
            pass

        logger.info(
            "[EvalMetrics] step=%d | %s",
            state.global_step,
            " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )


# ---------------------------------------------------------------------------
# Register reward func + auto-install callback
# ---------------------------------------------------------------------------

if orms is not None:
    orms["raptor_val_collector"] = ValMetricsCollectorReward


def _install_into_grpo_trainer():
    if os.environ.get("RAPTOR_INLINE_EVAL", "1") != "1":
        return
    try:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    except Exception as e:
        logger.warning(f"[EvalMetrics] failed to import GRPOTrainer ({e}); skipping install")
        return

    original_init = GRPOTrainer.__init__
    if getattr(original_init, "_raptor_eval_metrics_patched", False):
        return

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            cb = EvalMetricsCallback(only_rank_zero=True)
            self.add_callback(cb)
            logger.info("[EvalMetrics] callback installed")
        except Exception as e:  # pragma: no cover
            logger.error(f"[EvalMetrics] install failed: {e}")

    patched_init._raptor_eval_metrics_patched = True  # type: ignore
    GRPOTrainer.__init__ = patched_init


_install_into_grpo_trainer()
