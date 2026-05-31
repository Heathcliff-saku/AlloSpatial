# ms-swift patches

AlloSpatial trains with [ms-swift](https://github.com/modelscope/ms-swift). The reward
functions and the multi-turn tool scheduler are loaded as **external plugins**
(`--external_plugins ...`), so ms-swift itself does **not** need to be forked. The only
upstream change we rely on is a small fix to GRPO completion logging for multi-turn,
tool-using rollouts.

## `ms-swift-grpo_trainer-multiturn.patch`

**What it does.** In multi-turn rollouts the final assistant message contains the full
text (reasoning + tool calls + tool results), but `_prepare_batch_inputs` rewrites that
message in place with model-generated token ids only. The patch captures the human-readable
completion **before** that rewrite, so `--log_completions` records the full reasoning/tool
trace (used by `eval_metrics_callback.py` and W&B logging).

**Base revision.** Generated against ms-swift commit `b6f20f61b`. It touches a single file,
`swift/rlhf_trainers/grpo_trainer.py`. If your ms-swift is on a different revision and the
patch does not apply cleanly, apply the change by hand — it just moves the
`log_messages` / `log_completions` extraction earlier in `_compute_loss` / the logging block.

## How to apply

```bash
# 1. Install ms-swift (editable is convenient for patching)
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout b6f20f61b          # optional: pin to the tested revision
pip install -e .

# 2. Apply the patch
git apply /path/to/AlloSpatial/training/patches/ms-swift-grpo_trainer-multiturn.patch
# or, if not a git checkout:
patch -p1 < /path/to/AlloSpatial/training/patches/ms-swift-grpo_trainer-multiturn.patch
```

The SFT stage (`training/sft/`) needs **no** patch — stock ms-swift `swift sft` works.
