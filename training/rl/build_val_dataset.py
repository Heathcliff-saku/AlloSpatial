"""
Build a mixed validation JSONL (MindCube tiny + VSI-Bench tiny) for swift GRPO
native evaluation.

Output schema per row (ms-swift GRPO + extra metadata for eval aggregation):
    {
      "messages":      [system, user],
      "images": [...] | "videos": [...],     # absolute paths
      "solution":      <gt>,
      "answer_type":   "mca" | "na",
      "source":        "mindcube" | "vsibench",
      "task":          "mindcube_tiny" | "vsibench_tiny",  # bucket key
      "question_type": <qt>,                 # for VSI sub-aggregation
      "val_id":        <str>,                # stable per-sample id
    }

Data sources:
  - MindCube tiny (combined): HF parquet at
      $HF_HOME/hub/datasets--oscarqjh--MindCube_lmmseval/.../tiny/combined-*.parquet
    (embeds images as bytes; extracted here to val_cache/mindcube/)
  - VSI-Bench tiny: test.jsonl filtered by TINY_SUBSET_INDEXES; videos loaded
    from $VSIBENCH_VIDEO_BASE/{dataset}/{scene_name}.mp4

Usage:
    HF_HOME=/path/to/hf_cache VSIBENCH_VIDEO_BASE=/path/to/vsibench \\
      python build_val_dataset.py [--output grpo_val_tiny.jsonl]
"""

import argparse
import io
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_val_dataset")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.prompts import TRAINING_SYSTEM_PROMPT

# Parse three constants (TINY_SUBSET_INDEXES, MCA_QUESTION_TYPES, NA_QUESTION_TYPES)
# out of lmms-eval/vsibench/utils.py via AST, avoiding the lmms_eval import graph
# (which pulls in tenacity etc.).
import ast
_LMMS_VSI_UTILS = (
    _PROJECT_ROOT.parent / "lmms-eval" / "lmms_eval" / "tasks" / "vsibench" / "utils.py"
)
_src = _LMMS_VSI_UTILS.read_text(encoding="utf-8")
_tree = ast.parse(_src)
_wanted = {"TINY_SUBSET_INDEXES", "MCA_QUESTION_TYPES", "NA_QUESTION_TYPES"}
_consts: Dict[str, Any] = {}
for node in _tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]
        if isinstance(t, ast.Name) and t.id in _wanted:
            _consts[t.id] = ast.literal_eval(node.value)
TINY_SUBSET_INDEXES = _consts["TINY_SUBSET_INDEXES"]
VSI_MCA = set(_consts["MCA_QUESTION_TYPES"])
VSI_NA = set(_consts["NA_QUESTION_TYPES"])

HF_HOME = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))

MINDCUBE_REPO_DIR = os.path.join(HF_HOME, "hub", "datasets--oscarqjh--MindCube_lmmseval")
VSIBENCH_REPO_DIR = os.path.join(HF_HOME, "hub", "datasets--nyu-visionx--VSI-Bench")
VSIBENCH_VIDEO_BASE = os.environ.get("VSIBENCH_VIDEO_BASE", "/path/to/vsibench")  # {dataset}/{scene}.mp4
VSIBENCH_SCENEPP_ALIAS = "scannetpp"                            # HF uses "scannetpp"

OUTPUT_DEFAULT = Path(__file__).parent / "grpo_val_tiny.jsonl"
IMG_CACHE_DEFAULT = Path(__file__).parent / "val_cache" / "mindcube"


def _first_file(pattern_dir: str, suffix: str) -> str | None:
    root = Path(pattern_dir)
    if not root.exists():
        return None
    for p in root.rglob(f"*{suffix}"):
        return str(p)
    return None


def load_mindcube_tiny() -> List[Dict[str, Any]]:
    import pandas as pd
    parquet = _first_file(
        os.path.join(MINDCUBE_REPO_DIR, "snapshots"),
        "tiny/combined-00000-of-00001.parquet",
    )
    if parquet is None:
        raise FileNotFoundError(
            f"MindCube tiny parquet not found under {MINDCUBE_REPO_DIR}. "
            f"Set HF_HOME so that $HF_HOME/hub contains the downloaded dataset."
        )
    df = pd.read_parquet(parquet)
    logger.info(f"MindCube tiny: loaded {len(df)} rows from {parquet}")
    return df.to_dict(orient="records")


def load_vsibench_tiny() -> List[Dict[str, Any]]:
    jsonl = _first_file(os.path.join(VSIBENCH_REPO_DIR, "snapshots"), "test.jsonl")
    if jsonl is None:
        raise FileNotFoundError(
            f"VSI-Bench test.jsonl not found under {VSIBENCH_REPO_DIR}"
        )
    tiny_set = set(TINY_SUBSET_INDEXES)
    rows: List[Dict[str, Any]] = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if int(r["id"]) in tiny_set:
                rows.append(r)
    logger.info(f"VSI-Bench tiny: filtered {len(rows)} / {len(tiny_set)} expected")
    if len(rows) < len(tiny_set):
        logger.warning(f"{len(tiny_set) - len(rows)} tiny ids missing from test.jsonl")
    return rows


def extract_mindcube_images(row: Dict[str, Any], cache_dir: Path) -> List[str]:
    """Write PIL bytes to disk, return absolute paths."""
    row_dir = cache_dir / row["id"]
    row_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for idx, img_blob in enumerate(row["images"]):
        if not isinstance(img_blob, dict) or img_blob.get("bytes") is None:
            raise ValueError(f"MindCube {row['id']} image {idx} has no bytes payload")
        out = row_dir / f"img_{idx}.png"
        if not out.exists():
            # Re-encode through PIL to guarantee PNG
            from PIL import Image
            img = Image.open(io.BytesIO(img_blob["bytes"])).convert("RGB")
            img.save(out, format="PNG")
        paths.append(str(out.resolve()))
    return paths


def build_mindcube_row(row: Dict[str, Any], cache_dir: Path) -> Dict[str, Any]:
    image_paths = extract_mindcube_images(row, cache_dir)
    prompt = row["input_prompt"]
    user_content = "<image>" * len(image_paths) + "\n" + prompt
    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": image_paths,
        "solution": row["gt_answer"],
        "answer_type": "mca",
        "source": "mindcube",
        "task": "mindcube_tiny",
        "question_type": "mindcube",
        "val_id": row["id"],
    }


def build_vsibench_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    dataset = row["dataset"]
    scene = row["scene_name"]
    qt = row["question_type"]
    # VSI-590K naming difference (if any) — HF VSI-Bench uses: arkitscenes / scannet / scannetpp
    video_path = Path(VSIBENCH_VIDEO_BASE) / dataset / f"{scene}.mp4"
    if not video_path.exists():
        logger.warning(f"VSI video missing: {video_path}  (id={row['id']}, skipping)")
        return None

    question = row["question"]
    if qt in VSI_NA:
        user_content = (
            "<video>\nThese are frames of a video.\n"
            f"{question}\nPlease answer the question using a single word or phrase."
        )
        answer_type = "na"
    elif qt in VSI_MCA:
        opts = row.get("options")
        options_str = "\nOptions:\n" + "\n".join(list(opts)) if opts is not None else ""
        user_content = (
            "<video>\nThese are frames of a video.\n"
            f"{question}{options_str}\n"
            "Answer with the option's letter from the given choices directly."
        )
        answer_type = "mca"
    else:
        logger.warning(f"VSI unknown question_type={qt} for id={row['id']}, skipping")
        return None

    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "videos": [str(video_path.resolve())],
        "solution": str(row["ground_truth"]),
        "answer_type": answer_type,
        "source": "vsibench",
        "task": "vsibench_tiny",
        "question_type": qt,
        "val_id": f"vsi_{row['id']}",
    }


def _stratified_sample(
    rows: List[Dict[str, Any]], key_fn, target: int, seed: int
) -> List[Dict[str, Any]]:
    """Cap total sample count at `target`, stratified by `key_fn`, preserving type ratios."""
    import random
    if target <= 0 or target >= len(rows):
        return rows
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)

    total = len(rows)
    out: List[Dict[str, Any]] = []
    remaining = target
    bucket_items = list(buckets.items())
    for idx, (k, bucket) in enumerate(bucket_items):
        is_last = idx == len(bucket_items) - 1
        if is_last:
            take = remaining
        else:
            take = max(1, round(target * len(bucket) / total))
            take = min(take, len(bucket), remaining)
        out.extend(rng.sample(bucket, min(take, len(bucket))))
        remaining -= take
        if remaining <= 0:
            break
    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--img-cache", default=str(IMG_CACHE_DEFAULT))
    parser.add_argument(
        "--mindcube-n", type=int, default=100,
        help="Stratified sample size for MindCube (by category: rotation/around/among). "
             "Set 0 to keep all tiny samples.",
    )
    parser.add_argument(
        "--vsibench-n", type=int, default=100,
        help="Stratified sample size for VSI-Bench (by question_type). "
             "Set 0 to keep all available tiny samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-mindcube", action="store_true")
    parser.add_argument("--skip-vsibench", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.img_cache)
    output_path = Path(args.output)

    all_rows: List[Dict[str, Any]] = []

    if not args.skip_mindcube:
        mc_raw = load_mindcube_tiny()
        # Stratify by id prefix (among/around/rotation) BEFORE image extraction
        # so we don't pay PIL decode cost for samples we'll discard.
        sampled_raw = _stratified_sample(
            mc_raw,
            key_fn=lambda r: r["id"].split("_", 1)[0],
            target=args.mindcube_n,
            seed=args.seed,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        for r in sampled_raw:
            all_rows.append(build_mindcube_row(r, cache_dir))

    if not args.skip_vsibench:
        vsi_raw = load_vsibench_tiny()
        vsi_rows_all: List[Dict[str, Any]] = []
        for r in vsi_raw:
            row = build_vsibench_row(r)
            if row is not None:
                vsi_rows_all.append(row)
        vsi_rows = _stratified_sample(
            vsi_rows_all,
            key_fn=lambda r: r["question_type"],
            target=args.vsibench_n,
            seed=args.seed,
        )
        all_rows.extend(vsi_rows)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_task: Dict[str, int] = {}
    for r in all_rows:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    logger.info(f"Wrote {len(all_rows)} val rows to {output_path}")
    for t, n in by_task.items():
        logger.info(f"  {t}: {n}")


if __name__ == "__main__":
    main()
