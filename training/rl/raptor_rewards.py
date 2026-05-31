"""
Raptor-R1 GRPO Reward Functions (v2).

Three reward signals for spatial reasoning with tool usage:
1. StructureReward: validates reasoning chain format (Step1-5 protocol) AND
                    tool call format (parameter completeness, <tool_call> wrapping)
2. AnswerAccuracyReward: answer correctness (MCA exact match / NA MRA continuous)
3. ToolUsageReward: pure positive signal for (tool + correct answer)

Design philosophy (v2):
- StructureReward owns ALL format-related signals (reasoning steps + tool format).
  Penalizes: missing reasoning steps, malformed tool calls, fake tool JSON
  (model writing tool JSON outside <tool_call> tags), repetition loops.
- ToolUsageReward is purely positive: tool + correct answer = bonus.
  No negative signals here; those belong to StructureReward.
- This separation prevents the v1 failure mode where malformed tool call penalties
  in ToolUsageReward taught the model that "no call (0) > malformed call (negative)".

Register as ms-swift external plugins via --external_plugins raptor_rewards.py

Usage:
    swift rlhf --rlhf_type grpo \\
        --external_plugins raptor_rewards.py \\
        --reward_funcs raptor_structure raptor_accuracy raptor_tool_use \\
        --reward_weights 0.2 0.6 0.2
"""

import json
import os
import re
import logging
from typing import List, Optional

import numpy as np

from swift.rewards import ORM, orms

logger = logging.getLogger(__name__)

# Hard gate: AnswerAccuracyReward is forced to 0 when StructureReward < this
# threshold. Format-then-correct (DeepSeek-R1 style) prevents long, format-broken
# completions from accruing accuracy signal and driving the format-collapse
# failure mode observed in run-20260425_172736-5cxsw01q (structure 0.87→0.46).
_STRUCTURE_GATE = float(os.environ.get('RAPTOR_STRUCTURE_GATE', '0.85'))

# ---------------------------------------------------------------------------
# Step structure detection
# ---------------------------------------------------------------------------

def _step_pattern(label: str) -> re.Pattern:
    """Compile a regex for a given step label (e.g. '1', '2.1', '4')."""
    return re.compile(
        r"(?:#{1,3}\s*)?(?:\*{1,2})?[Ss]tep\s*"
        + re.escape(label)
        + r"(?:\*{1,2})?(?:[:\s]|$)",
        re.MULTILINE,
    )


_STEP_PATTERNS = {
    "1":   _step_pattern("1"),
    "2":   _step_pattern("2"),
    "2.1": _step_pattern("2.1"),
    "2.2": _step_pattern("2.2"),
    "3":   _step_pattern("3"),
    "3.1": _step_pattern("3.1"),
    "3.2": _step_pattern("3.2"),
    "4":   _step_pattern("4"),
    "5":   _step_pattern("5"),
}


def _has_step(text: str, label: str) -> bool:
    pat = _STEP_PATTERNS.get(label)
    return bool(pat.search(text)) if pat else False


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.IGNORECASE | re.DOTALL)

# Required fields for a valid world2mind tool call
WORLD2MIND_REQUIRED_FIELDS = {"categories", "scene_type", "knowledge_type", "output_format"}

# Turn separator pattern used by raptor_scheduler in the rollout log
_TURN_SEP_RE = re.compile(r'---\s*\[Turn\s*\d+:\s*Assistant\]\s*---')

# ---------------------------------------------------------------------------
# Length penalty helpers (used by LengthPenaltyReward)
# ---------------------------------------------------------------------------

# Strip tool-return sections from merged completion text so that
# LengthPenaltyReward measures model-generated text only.
_TOOL_RESULT_RE = re.compile(
    r'---\s*\[Tool Result\]\s*---.*?(?=---\s*\[Turn|$)', re.DOTALL)
_ROLLOUT_HEADER_RE = re.compile(r'^\[Rollout:.*?\]\n', re.MULTILINE)

# ---------------------------------------------------------------------------
# Extra-step detection (reward-hacking defence)
# ---------------------------------------------------------------------------

# Matches step labels whose base number is > 5 (e.g. Step 6, Step 7, Step 13).
# These are signs of reward hacking where the model invents extra steps.
_EXTRA_STEP_RE = re.compile(
    r'(?:#{1,3}\s*)?(?:✅\s*)?[Ss]tep\s*([6-9]|\d{2,})(?:\.\d+)?(?:\s|:|$)',
    re.MULTILINE,
)


def _count_extra_steps(text: str) -> int:
    """Count step labels whose base number is > 5."""
    return len(_EXTRA_STEP_RE.findall(text))


def parse_tool_calls_from_text(text: str) -> List[dict]:
    """Parse <tool_call> blocks from completion text."""
    matches = _TOOL_CALL_PATTERN.findall(text)
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if 'name' in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            tool_calls.append({'name': 'malformed', 'arguments': {}, '_parse_error': True})
    return tool_calls


def get_tool_names_from_text(text: str) -> List[str]:
    """Extract tool names from actual <tool_call> blocks."""
    return [tc['name'] for tc in parse_tool_calls_from_text(text)]


def count_fake_tool_json(text: str) -> int:
    """
    Count JSON-style tool calls that appear OUTSIDE <tool_call> tags.

    This detects the reward-hacking pattern where the model writes tool call
    JSON in plain text or code blocks to mimic tool usage without actually
    calling the tool (which would trigger the scheduler and require correct params).

    Returns: number of fake tool JSON patterns found.
    """
    # Remove everything inside <tool_call>...</tool_call> to avoid false positives
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    w2m_pat = re.compile(r'["\']name["\']\s*:\s*["\']world2mind["\']')
    vi_pat  = re.compile(r'["\']name["\']\s*:\s*["\']view_image["\']')
    return len(w2m_pat.findall(cleaned)) + len(vi_pat.findall(cleaned))


# ---------------------------------------------------------------------------
# Loop / repetition detection
# ---------------------------------------------------------------------------

def detect_loop(text: str, min_length: int = 3000, snippet_start: int = 200,
                snippet_end: int = 400, repeat_threshold: int = 3) -> bool:
    """
    Detect degenerate repetition loops in completions.

    A loop is identified when:
    1. The completion is longer than min_length characters
    2. There is no <Answer> tag (the loop never resolves)
    3. A 200-char snippet from position [snippet_start:snippet_end] appears
       at least repeat_threshold times in the full text

    This catches the observed failure mode where the model generates 38k+ chars
    repeating the same reasoning paragraph indefinitely.
    """
    if len(text) < min_length:
        return False
    if _ANSWER_TAG_RE.search(text):
        return False
    snippet = text[snippet_start:snippet_end]
    if len(snippet) < 50:  # snippet too short to be meaningful
        return False
    return text.count(snippet) >= repeat_threshold


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_LETTER_PATTERNS = [
    re.compile(r"<Answer>\s*([A-Ea-e])\s*</Answer>", re.IGNORECASE),
    re.compile(r"\b([A-Ea-e])\.\s"),
    re.compile(r"answer[:\s]+([A-Ea-e])\b", re.IGNORECASE),
]


def extract_mca_letter(text: str) -> Optional[str]:
    """Extract MCA letter answer from text."""
    for pat in _LETTER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract numeric answer from <Answer> tag."""
    m = _ANSWER_TAG_RE.search(text)
    if m:
        raw = m.group(1).strip().replace(",", "")
        num_match = re.search(r"[-+]?\d*\.?\d+", raw)
        if num_match:
            try:
                return float(num_match.group())
            except ValueError:
                pass
    return None


def mean_relative_accuracy(pred: float, target: float,
                           start: float = 0.5, end: float = 0.95,
                           interval: float = 0.05) -> float:
    """
    Compute MRA (Mean Relative Accuracy) as used in VSI-Bench evaluation.
    Returns a continuous reward in [0, 1].
    """
    if target == 0:
        return 1.0 if abs(pred) < 1e-6 else 0.0
    num_pts = int((end - start) / interval + 2)
    conf_intervals = np.linspace(start, end, num_pts)
    rel_error = abs(pred - target) / abs(target)
    accuracy = rel_error <= (1 - conf_intervals)
    return float(accuracy.mean())


# ---------------------------------------------------------------------------
# Reward 1: StructureReward (v2 — includes tool format checks)
# ---------------------------------------------------------------------------

class StructureReward(ORM):
    """
    Validates BOTH reasoning chain format AND tool call format.

    Two-part scoring:
      Part A (Reasoning format): Step1-5 completeness protocol
      Part B (Tool format): parameter completeness + fake-call detection

    Hard prerequisites (either → 0.0):
      - Missing <Answer> tag
      - Repetition loop detected (3000+ chars, no Answer, snippet repeats 3×)

    Deductions from 1.0 (clamped to [0, 1]):
      Part A:
        - Missing Step1: -0.30
        - Missing Step5 (always required): -0.20
        - Used world2mind but missing Step2/2.1/2.2: -0.10
        - Used view_image but missing Step3/3.1/3.2: -0.10
        - Used any tool but missing Step4: -0.10
        - Turn 2+ repeats Step1 after a tool call: -0.15 (once)
      Part B (per tool call):
        - Malformed JSON in <tool_call>: -0.15/call
        - world2mind missing any required field: -0.20/call (flat penalty)
        - world2mind empty categories: -0.05/call
      Part B (fake detection):
        - Fake tool JSON outside <tool_call> (1 occurrence): -0.15
        - Fake tool JSON outside <tool_call> (2+ occurrences): -0.30
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            reward = self._score_single(completion)
            rewards.append(max(0.0, min(1.0, reward)))

        if logger.isEnabledFor(logging.DEBUG):
            for i, (comp, rew) in enumerate(zip(completions, rewards)):
                has_answer = bool(_ANSWER_TAG_RE.search(comp))
                tools = get_tool_names_from_text(comp)
                fake_count = count_fake_tool_json(comp)
                logger.debug(f"StructureReward[{i}]: reward={rew:.2f}, "
                             f"has_answer={has_answer}, tools={tools}, "
                             f"fake_json={fake_count}, len={len(comp)}")
        return rewards

    def _score_single(self, text: str) -> float:
        # --- Hard prerequisites ---
        if not _ANSWER_TAG_RE.search(text):
            return 0.0

        if detect_loop(text):
            return 0.0

        score = 1.0

        # --- Part A: Reasoning format ---
        if not _has_step(text, "1"):
            score -= 0.30

        # Step5 is always required (final answer synthesis)
        if not _has_step(text, "5"):
            score -= 0.20

        # Detect actual tool usage (from <tool_call> blocks only)
        tool_names = get_tool_names_from_text(text)
        used_world2mind = "world2mind" in tool_names
        used_view_image = "view_image" in tool_names
        used_any_tool = used_world2mind or used_view_image

        # --- Rule C (jump-in): no tool + missing Step1 + later steps present ---
        # SFT no-tool path has Step1 in 100% of samples; missing it while
        # writing Step2.x/3.x/4/5 is the canonical "skip the intro and pretend
        # to have already done deep reasoning" hack. Hard zero.
        if not used_any_tool and not _has_step(text, "1"):
            if any(_has_step(text, lbl) for lbl in
                   ("2", "2.1", "2.2", "3", "3.1", "3.2", "4", "5")):
                return 0.0

        # --- Rule A (cosplay): tool-output steps must accompany their tool ---
        # SFT: Step2.2 (interpret world2mind output) appears in ~0% of no-tool
        # samples but ~78–99% when world2mind is used; same for Step3.2 vs
        # view_image. Writing them without the corresponding tool call is
        # narrating fabricated tool output.
        # Step4 is allowed without tools (SFT: 11% of no-tool samples).
        if not used_world2mind and _has_step(text, "2.2"):
            score -= 0.35
        if not used_view_image and _has_step(text, "3.2"):
            score -= 0.35

        if used_world2mind:
            if not (_has_step(text, "2.1") or _has_step(text, "2.2") or _has_step(text, "2")):
                score -= 0.10

        if used_view_image:
            if not (_has_step(text, "3.1") or _has_step(text, "3.2") or _has_step(text, "3")):
                score -= 0.10

        if used_any_tool:
            if not _has_step(text, "4"):
                score -= 0.10

        # Repeated Step1 detection: after a tool call Turn 2+ should NOT re-state Step1.
        # Normal flow: Turn1(Step1→tool) → Turn2(Step2.1/2.2→Step4→Step5)
        # Abnormal:    Turn1(Step1→tool) → Turn2(Step1 again → ...)
        if used_any_tool:
            parts = _TURN_SEP_RE.split(text)
            # parts[0]=header, parts[1]=Turn1 content, parts[2]=Turn2 content, ...
            if len(parts) >= 3:
                for later_turn in parts[2:]:
                    if _has_step(later_turn, "1"):
                        score -= 0.15
                        break  # penalize once regardless of how many later turns repeat

        # --- Extra step numbers (reward-hacking defence) ---
        # Steps numbered > 5 (Step 6, Step 7, Step 13, …) are a hallmark of the
        # degenerate behaviour where the model pads with invented steps to score
        # higher on structure reward checks.
        extra = _count_extra_steps(text)
        if extra > 0:
            score -= min(0.20 * extra, 0.40)  # -0.20 per extra step, capped at -0.40

        # --- Step naming validation ---
        # Step 1 should be the "Visual Clues" step (always present).
        step1_line = re.search(
            r'(?:#{1,3}\s*)?[Ss]tep\s*1(?:\s|:)(.*)', text)
        if step1_line and 'visual' not in step1_line.group(1).lower()[:80]:
            score -= 0.10

        # Step 5 should be the "Final Answer" step (always present).
        step5_line = re.search(
            r'(?:#{1,3}\s*)?[Ss]tep\s*5(?:\s|:)(.*)', text)
        if step5_line:
            ln5 = step5_line.group(1).lower()[:80]
            if 'final' not in ln5 and 'answer' not in ln5:
                score -= 0.10

        # Step 4 should be "Cross-Validation" (only required when tools were used).
        if used_any_tool:
            step4_line = re.search(
                r'(?:#{1,3}\s*)?[Ss]tep\s*4(?:\s|:)(.*)', text)
            if step4_line:
                ln4 = step4_line.group(1).lower()[:80]
                if not any(k in ln4 for k in ('cross', 'valid', 'verif')):
                    score -= 0.05

        # --- Part B: Tool format ---
        tool_calls = parse_tool_calls_from_text(text)
        for tc in tool_calls:
            if tc.get('_parse_error'):
                score -= 0.15
                continue
            if tc['name'] == 'world2mind':
                args = tc.get('arguments', {})
                if not isinstance(args, dict):
                    score -= 0.15
                    continue
                # Missing any required field → flat penalty
                missing = WORLD2MIND_REQUIRED_FIELDS - set(args.keys())
                if missing:
                    score -= 0.20
                # Empty categories
                cats = args.get('categories', [])
                if not isinstance(cats, list) or len(cats) == 0:
                    score -= 0.05

        # Fake tool JSON detection (model writes tool JSON outside <tool_call> tags)
        fake_count = count_fake_tool_json(text)
        if fake_count >= 2:
            score -= 0.30
        elif fake_count == 1:
            score -= 0.15

        return score


# ---------------------------------------------------------------------------
# Reward 2: AnswerAccuracyReward
# ---------------------------------------------------------------------------

class AnswerAccuracyReward(ORM):
    """
    Answer correctness reward, hard-gated by structure score.

    - MCA: exact letter match → 1.0 / 0.0
    - NA: MRA (Mean Relative Accuracy) → continuous [0, 1]

    Hard gate: if StructureReward(completion) < RAPTOR_STRUCTURE_GATE
    (default 0.8), the accuracy is forced to 0 regardless of pred/gt match.
    Cascade: ToolUsageReward calls this class internally for its acc_score, so
    a structure failure also zeros the tool-bonus part automatically.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        solutions = kwargs.get('solution', [])
        answer_types = kwargs.get('answer_type', [])

        # Compute structure score for gating (regex-only; ~negligible cost).
        struct_scores = StructureReward()(completions)

        raw_rewards: List[float] = []
        for i, completion in enumerate(completions):
            sol = solutions[i] if i < len(solutions) else ""
            at = answer_types[i] if i < len(answer_types) else "mca"
            raw = max(0.0, min(1.0, self._score_single(completion, sol, at)))
            raw_rewards.append(raw)

        rewards: List[float] = []
        gated_flags: List[int] = []
        for raw, struct in zip(raw_rewards, struct_scores):
            if struct < _STRUCTURE_GATE:
                rewards.append(0.0)
                gated_flags.append(1)
            else:
                rewards.append(raw)
                gated_flags.append(0)

        if not kwargs.get('_skip_gate_metrics', False):
            self._log_gate_metrics(raw_rewards, gated_flags)

        if logger.isEnabledFor(logging.DEBUG):
            for i, (comp, rew, raw, struct) in enumerate(
                zip(completions, rewards, raw_rewards, struct_scores)
            ):
                sol = solutions[i] if i < len(solutions) else ""
                at = answer_types[i] if i < len(answer_types) else "mca"
                pred = (extract_numeric_answer(comp) if at == "na"
                        else extract_mca_letter(comp))
                logger.debug(
                    f"AccuracyReward[{i}]: reward={rew:.3f} raw={raw:.3f} "
                    f"struct={struct:.3f} gated={struct < _STRUCTURE_GATE} "
                    f"type={at} pred={pred} gt={sol}"
                )
        return rewards

    @staticmethod
    def _log_gate_metrics(raw_rewards: List[float], gated_flags: List[int]) -> None:
        if not raw_rewards:
            return
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
            import wandb  # type: ignore
            if wandb.run is None:
                return
            n = len(raw_rewards)
            wandb.log({
                "rewards/AccuracyReward/raw_mean": sum(raw_rewards) / n,
                "rewards/AccuracyReward/gated_ratio": sum(gated_flags) / n,
            })
        except Exception as e:
            logger.debug(f"[AccuracyReward] wandb metric log skipped: {e}")

    def _score_single(self, text: str, solution: str, answer_type: str) -> float:
        if not solution:
            return 0.0

        if answer_type == "na":
            pred = extract_numeric_answer(text)
            if pred is None:
                return 0.0
            try:
                gt = float(solution.strip().replace(",", ""))
            except ValueError:
                return 0.0
            return mean_relative_accuracy(pred, gt)
        else:
            # MCA
            pred = extract_mca_letter(text)
            if pred is None:
                return 0.0
            gt = solution.strip().upper()
            return 1.0 if pred == gt else 0.0


# ---------------------------------------------------------------------------
# Reward 3: ToolUsageReward (v2 — pure positive signal)
# ---------------------------------------------------------------------------

class ToolUsageReward(ORM):
    """
    Pure positive reward for tool usage + correct answer.

    Design (v2): All negative signals (malformed params, fake calls, loops)
    have been moved to StructureReward. This reward ONLY gives bonuses for
    the synergy of (actually calling a tool correctly) + (answer quality).

    Key properties:
    - Only VALID world2mind calls (all required fields present, non-empty categories,
      no parse error) count toward the bonus. A broken call that happens to produce
      a correct answer does NOT get the tool bonus — we reward correct tool USE,
      not lucky outcomes despite broken calls.
    - Answer quality is continuous: bonus = base × acc_score.
      For MCA (acc_score ∈ {0, 1}) this is equivalent to the old binary gate.
      For NA (acc_score ∈ [0, 1] via MRA) this scales the bonus proportionally,
      rewarding partial numeric accuracy rather than hard thresholding.
    - Excess calls (total w2m > 2) are penalized regardless of validity.

    Scoring:
      - valid world2mind (1-2 calls): +0.50 × acc_score
      - view_image (1-4 calls):       +0.20 × acc_score
      - world2mind calls > 2:         -0.15 × (total_count - 2)
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        answer_types = kwargs.get('answer_type', [])

        # Pre-compute continuous accuracy scores (MRA for NA, 0/1 for MCA).
        # Gate metrics are suppressed here to avoid duplicate wandb writes;
        # the gate itself still applies, so tool bonus auto-cascades to 0
        # whenever the structure score is below RAPTOR_STRUCTURE_GATE.
        accuracy_reward = AnswerAccuracyReward()
        acc_scores = accuracy_reward(completions, **kwargs, _skip_gate_metrics=True)

        rewards = []
        for i, completion in enumerate(completions):
            at = answer_types[i] if i < len(answer_types) else "mca"
            reward = self._score_single(completion, acc_scores[i], at)
            rewards.append(max(-1.0, min(1.0, reward)))

        if logger.isEnabledFor(logging.DEBUG):
            for i, (comp, rew) in enumerate(zip(completions, rewards)):
                tools = get_tool_names_from_text(comp)
                at = answer_types[i] if i < len(answer_types) else "mca"
                logger.debug(f"ToolUsageReward[{i}]: reward={rew:.2f}, tools={tools}, "
                             f"acc={acc_scores[i]:.3f}, type={at}")
        return rewards

    def _score_single(self, text: str, acc_score: float, answer_type: str) -> float:
        tool_calls = parse_tool_calls_from_text(text)

        # Total world2mind calls (for excess penalty)
        w2m_count = sum(1 for tc in tool_calls if tc['name'] == 'world2mind')
        vi_count  = sum(1 for tc in tool_calls if tc['name'] == 'view_image')

        # Valid world2mind calls only (for bonus):
        # must be parseable, have all required fields, and non-empty categories.
        valid_w2m = 0
        for tc in tool_calls:
            if tc['name'] != 'world2mind':
                continue
            if tc.get('_parse_error'):
                continue
            args = tc.get('arguments', {})
            if not isinstance(args, dict):
                continue
            if WORLD2MIND_REQUIRED_FIELDS - set(args.keys()):
                continue  # missing required field(s)
            cats = args.get('categories', [])
            if not isinstance(cats, list) or len(cats) == 0:
                continue  # empty categories
            valid_w2m += 1

        score = 0.0

        # Bonus: valid tool usage × answer quality (continuous for NA, 0/1 for MCA)
        if 1 <= valid_w2m <= 2:
            score += 0.50 * acc_score
        if 1 <= vi_count <= 4:
            score += 0.20 * acc_score

        # Penalty: too many total world2mind calls (prevents repetition hacking)
        if w2m_count > 2:
            score -= 0.15 * (w2m_count - 2)

        return score


# ---------------------------------------------------------------------------
# Reward 4: LengthPenaltyReward
# ---------------------------------------------------------------------------

class LengthPenaltyReward(ORM):
    """
    Three-segment length penalty on model-generated tokens only.

    Uses kwargs['response_loss_mask'] (populated by swift rollout_mixin) to
    isolate model-generated tokens. The raptor scheduler appends tool-return
    tokens into response_token_ids with loss_mask=0, so we count only tokens
    where loss_mask == 1 to exclude tool tokens from the length budget.

    Penalty schedule (token-based, aligned with wandb `completion_length`):
      tokens <= _BUDGET:        0.0
      _BUDGET < tokens <= _MAX: -(tokens - _BUDGET) / (_MAX - _BUDGET)   (linear 0 → -1)
      tokens > _MAX:           -1.0

    Budget/max chosen relative to --max_completion_length 8192:
      _BUDGET=4000 (typical good completions stay well below this)
      _MAX=8000    (≈ max_completion_length; full penalty at hard limit)

    Fallback order:
      1. response_loss_mask → sum(loss_mask==1)  (model tokens only, preferred)
      2. response_token_ids → sum(len)           (model + tool tokens; may over-count)
      3. char count on cleaned completion string × 4x char budget
    """

    _BUDGET: int = 2000         # tokens
    _MAX: int = 4000            # tokens
    _BUDGET_CHARS: int = 16000   # fallback: ~4x token budget
    _MAX_CHARS: int = 32000

    @staticmethod
    def _count_model_tokens(per_sample_mask) -> Optional[int]:
        """Sum loss_mask==1 across turns. Handles list[int] or list[list[int]]."""
        if per_sample_mask is None:
            return None
        if not per_sample_mask:
            return 0
        first = per_sample_mask[0]
        if isinstance(first, (list, tuple)):
            return sum(sum(1 for m in turn if m == 1) for turn in per_sample_mask)
        # flat list[int]
        return sum(1 for m in per_sample_mask if m == 1)

    @staticmethod
    def _count_all_tokens(per_sample_rti) -> Optional[int]:
        """Sum token counts across turns (includes tool tokens). Fallback only."""
        if per_sample_rti is None:
            return None
        if not per_sample_rti:
            return 0
        first = per_sample_rti[0]
        if isinstance(first, (list, tuple)):
            return sum(len(turn) for turn in per_sample_rti)
        return len(per_sample_rti)

    def _penalty(self, n: int, budget: int, max_n: int) -> float:
        if n <= budget:
            return 0.0
        if n <= max_n:
            return -(n - budget) / float(max_n - budget)
        return -1.0

    def __call__(self, completions, **kwargs) -> List[float]:
        mask_list = kwargs.get('response_loss_mask')
        rti_list = kwargs.get('response_token_ids')
        rewards: List[float] = []
        n_tokens: List[Optional[int]] = []

        for i, completion in enumerate(completions):
            mask = mask_list[i] if mask_list is not None and i < len(mask_list) else None
            n_tok = self._count_model_tokens(mask)
            if n_tok is None:
                rti = rti_list[i] if rti_list is not None and i < len(rti_list) else None
                n_tok = self._count_all_tokens(rti)
            n_tokens.append(n_tok)

            if n_tok is not None:
                rewards.append(self._penalty(n_tok, self._BUDGET, self._MAX))
            else:
                clean = _TOOL_RESULT_RE.sub('', completion)
                clean = _ROLLOUT_HEADER_RE.sub('', clean)
                rewards.append(self._penalty(len(clean), self._BUDGET_CHARS, self._MAX_CHARS))

        if logger.isEnabledFor(logging.DEBUG):
            for i, (comp, rew, n_tok) in enumerate(zip(completions, rewards, n_tokens)):
                logger.debug(f"LengthPenaltyReward[{i}]: reward={rew:.3f}, "
                             f"model_tokens={n_tok}, total_chars={len(comp)}")
        return rewards


# ---------------------------------------------------------------------------
# Register reward functions
# ---------------------------------------------------------------------------

orms['raptor_structure'] = StructureReward
orms['raptor_accuracy'] = AnswerAccuracyReward
orms['raptor_tool_use'] = ToolUsageReward
orms['raptor_length'] = LengthPenaltyReward
