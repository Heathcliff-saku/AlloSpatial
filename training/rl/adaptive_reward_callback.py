"""
Signal-adaptive reward-weight scheduling for Raptor-R1 GRPO training.

Replaces the no-op phase-based curriculum that used to live in
``raptor_rewards.py::_get_curriculum_multiplier`` with a data-driven
controller that reads recent reward statistics and rescales the weights
on ``raptor_length`` and ``raptor_tool_use`` only. ``raptor_structure`` and
``raptor_accuracy`` stay at their base weights — they carry the primary
signal for GRPO advantages and shouldn't drift under us.

Three factors, multiplied together, produce a raw weight:
  * headroom h_i = clip(sigma_i / sigma_target_i, 0.1, 2.0)
      signal-still-moving check; if std collapses (reward converged), drop.
  * saturation discount d_i = 1 - max(0, s_i - 0.3) / 0.7
      s_i = fraction of recent samples with |r - ref| < eps
      (ref = max possible for length penalty, 0 otherwise)
      — fully saturated reward → d_i ~ 0.
  * trend compensation t_i = 1 + clip(-trend_i / (|mu_i| + eps), 0, 1.0)
      — reward dropping → boost weight to rescue the signal.

Raw weights are renormalised *within the adaptive subset* so
sum(raptor_length, raptor_tool_use) stays equal to its base sum
(0.15 + 0.10 = 0.25 in the default v2 config). Then EMA-smoothed into the
live ``trainer.reward_weights`` tensor in-place.

Environment flag: ``RAPTOR_ADAPTIVE_WEIGHTS=1`` enables the callback.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:  # pragma: no cover
    TrainerCallback = object  # type: ignore


ADAPTIVE_SET = {"raptor_length", "raptor_tool_use"}

# Target stds — the headroom we expect a healthy reward to still have. If
# the observed std is well below this, the reward is converging and gets
# downweighted.
DEFAULT_SIGMA_TARGET = {
    "raptor_length": 0.15,
    "raptor_tool_use": 0.10,
}


@dataclass
class RewardStats:
    """Rolling statistics for a single reward component."""

    window: int = 50
    means: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    stds: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    saturations: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def push(self, mean: float, std: float, saturation: float):
        self.means.append(float(mean))
        self.stds.append(float(std))
        self.saturations.append(float(saturation))

    def current_mean(self) -> float:
        return self.means[-1] if self.means else 0.0

    def current_std(self) -> float:
        return self.stds[-1] if self.stds else 0.0

    def current_sat(self) -> float:
        return self.saturations[-1] if self.saturations else 0.0

    def trend(self) -> float:
        """Linear slope (per step) over the current window. Zero if <5 points."""
        n = len(self.means)
        if n < 5:
            return 0.0
        xs = list(range(n))
        ys = list(self.means)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0


class AdaptiveRewardCallback(TrainerCallback):
    """Update ``trainer.reward_weights`` every ``update_every`` steps."""

    def __init__(
        self,
        trainer,
        reward_names: List[str],
        warmup_steps: int = 50,
        update_every: int = 10,
        ema_alpha: float = 0.2,
        window: int = 50,
        sigma_target: Optional[Dict[str, float]] = None,
    ):
        self.trainer = trainer
        self.reward_names = list(reward_names)
        self.warmup_steps = int(warmup_steps)
        self.update_every = max(1, int(update_every))
        self.ema_alpha = float(ema_alpha)
        self.window = int(window)
        self.sigma_target = dict(DEFAULT_SIGMA_TARGET)
        if sigma_target:
            self.sigma_target.update(sigma_target)

        # Snapshot base weights — frozen components and the total of the
        # adaptive subset are both anchored here.
        import torch

        base = trainer.reward_weights.detach().cpu().tolist()
        self.base_weights: Dict[str, float] = {
            name: float(base[i]) for i, name in enumerate(self.reward_names)
        }
        self.adaptive_base_sum = sum(
            self.base_weights[n] for n in self.reward_names if n in ADAPTIVE_SET
        )
        self.stats: Dict[str, RewardStats] = {
            n: RewardStats(window=self.window) for n in self.reward_names
        }

        logger.info(
            f"[AdaptiveReward] base_weights={self.base_weights}, "
            f"adaptive_set={ADAPTIVE_SET & set(self.reward_names)}, "
            f"sigma_target={self.sigma_target}"
        )

    # -- stats ingestion -----------------------------------------------------

    def _ingest_logs(self, logs: Dict[str, float]):
        """Pick up per-reward mean/std from the log dict swift publishes every step."""
        for name in self.reward_names:
            mean_k = f"rewards/{name}/mean"
            std_k = f"rewards/{name}/std"
            if mean_k not in logs or std_k not in logs:
                continue
            mean = float(logs[mean_k])
            std = float(logs[std_k])

            # Saturation heuristic: near-zero std AND near-reference mean.
            # For length penalty the "fully satisfied" state is mean ~ 0.0
            # (no penalty triggered). For tool_use, mean near the ceiling
            # (~1.0) counts as saturated upward.
            ref = 1.0 if name == "raptor_tool_use" else 0.0
            eps_close = 0.05
            saturated = 1.0 if (std < 0.02 and abs(mean - ref) < eps_close) else 0.0

            self.stats[name].push(mean, std, saturated)

    # -- weight computation --------------------------------------------------

    def _compute_new_weights(self) -> Optional[Dict[str, float]]:
        import math

        raw: Dict[str, float] = {}
        for name in self.reward_names:
            base_w = self.base_weights[name]
            if name not in ADAPTIVE_SET:
                raw[name] = base_w
                continue

            stats = self.stats[name]
            if len(stats.means) < 5:
                raw[name] = base_w
                continue

            sigma = stats.current_std()
            sigma_target = self.sigma_target.get(name, 0.1)
            headroom = max(0.1, min(2.0, sigma / max(1e-6, sigma_target)))

            # Longer-horizon saturation: fraction of window that looked saturated.
            sat_frac = sum(stats.saturations) / len(stats.saturations)
            sat_discount = 1.0 - max(0.0, sat_frac - 0.3) / 0.7

            mu = stats.current_mean()
            trend = stats.trend()
            # Scale trend by |mu|+eps; negative trend (dropping) boosts weight.
            trend_norm = -trend / (abs(mu) + 0.05)
            trend_boost = 1.0 + max(0.0, min(1.0, trend_norm))

            raw[name] = base_w * headroom * sat_discount * trend_boost

        # Renormalise within the adaptive subset so the subset sum matches
        # the base subset sum (preserves overall advantage scale).
        adaptive_raw_sum = sum(
            raw[n] for n in self.reward_names if n in ADAPTIVE_SET
        )
        if adaptive_raw_sum <= 0:
            return None
        scale = self.adaptive_base_sum / adaptive_raw_sum
        for n in self.reward_names:
            if n in ADAPTIVE_SET:
                raw[n] *= scale

        return raw

    # -- HF hooks ------------------------------------------------------------

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        self._ingest_logs(logs)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0))
        if step < self.warmup_steps:
            return control
        if step % self.update_every != 0:
            return control

        new_w = self._compute_new_weights()
        if new_w is None:
            return control

        import torch

        prev = self.trainer.reward_weights.detach().cpu().tolist()
        smoothed: List[float] = []
        for i, name in enumerate(self.reward_names):
            target = new_w[name]
            ema = (1.0 - self.ema_alpha) * prev[i] + self.ema_alpha * target
            smoothed.append(ema)

        # In-place copy to preserve device / dtype on the training tensor.
        with torch.no_grad():
            tensor = torch.tensor(
                smoothed,
                dtype=self.trainer.reward_weights.dtype,
                device=self.trainer.reward_weights.device,
            )
            self.trainer.reward_weights.copy_(tensor)

        logger.info(
            f"[AdaptiveReward] step={step} weights={dict(zip(self.reward_names, smoothed))}"
        )
        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                log_payload = {
                    f"reward_weights/{name}": smoothed[i]
                    for i, name in enumerate(self.reward_names)
                }
                log_payload["reward_weights/sum"] = sum(smoothed)
                wandb.log(log_payload, step=step)
        except Exception as e:
            logger.debug(f"[AdaptiveReward] wandb.log skipped: {e}")
        return control


# ---------------------------------------------------------------------------
# Swift external_plugins hook: monkey-patch GRPOTrainer to auto-install us.
# ---------------------------------------------------------------------------

def _install_into_grpo_trainer():
    if os.environ.get("RAPTOR_ADAPTIVE_WEIGHTS", "0") != "1":
        return

    try:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    except Exception as e:
        logger.warning(
            f"[AdaptiveReward] failed to import GRPOTrainer ({e}); skipping install"
        )
        return

    warmup = int(os.environ.get("RAPTOR_ADAPTIVE_WARMUP", "50"))
    every = int(os.environ.get("RAPTOR_ADAPTIVE_EVERY", "10"))
    alpha = float(os.environ.get("RAPTOR_ADAPTIVE_EMA", "0.2"))
    window = int(os.environ.get("RAPTOR_ADAPTIVE_WINDOW", "50"))

    sigma_target: Dict[str, float] = {}
    env_st_len = os.environ.get("RAPTOR_ADAPTIVE_SIGMA_LENGTH")
    env_st_tool = os.environ.get("RAPTOR_ADAPTIVE_SIGMA_TOOL")
    if env_st_len:
        sigma_target["raptor_length"] = float(env_st_len)
    if env_st_tool:
        sigma_target["raptor_tool_use"] = float(env_st_tool)

    original_init = GRPOTrainer.__init__
    if getattr(original_init, "_raptor_adaptive_patched", False):
        return

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            reward_names = list(getattr(self, "reward_func_names", []))
            if not reward_names:
                logger.warning(
                    "[AdaptiveReward] trainer.reward_func_names empty; skipping"
                )
                return
            cb = AdaptiveRewardCallback(
                trainer=self,
                reward_names=reward_names,
                warmup_steps=warmup,
                update_every=every,
                ema_alpha=alpha,
                window=window,
                sigma_target=sigma_target,
            )
            self.add_callback(cb)
            logger.info(
                f"[AdaptiveReward] installed (warmup={warmup}, every={every}, "
                f"alpha={alpha}, window={window})"
            )
        except Exception as e:
            logger.error(f"[AdaptiveReward] install failed: {e}")

    patched_init._raptor_adaptive_patched = True  # type: ignore
    GRPOTrainer.__init__ = patched_init


_install_into_grpo_trainer()
