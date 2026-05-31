"""
Build GRPO training dataset from VSI-590K and MindCube.

Performs stratified sampling from VSI-590K (~50K samples) and includes
all MindCube training samples (~10K). Outputs ms-swift GRPO JSONL format.

Usage:
    python build_grpo_dataset.py
    python build_grpo_dataset.py --vsi-target 50000 --output grpo_dataset.jsonl
    python build_grpo_dataset.py --vsi-only --vsi-target 5000  # small test set
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_grpo_dataset")

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.prompts import TRAINING_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Local dataset roots. Override via the matching environment variables, or edit here.
# VSI-590K: https://huggingface.co/datasets/nyu-visionx/VSI-590K
# MindCube: https://huggingface.co/datasets/MLL-Lab/MindCube
VSI_DATA_PATH = os.environ.get("VSI_DATA_PATH", "/path/to/VSI-590K/vsi_arkitscenes_scannet_scannetppv2.jsonl")
VSI_VIDEO_BASE = os.environ.get("VSI_VIDEO_BASE", "/path/to/VSI-590K")

MINDCUBE_DATA_PATH = os.environ.get("MINDCUBE_DATA_PATH", "/path/to/MindCube/data/raw/MindCube_train.jsonl")
MINDCUBE_IMAGE_BASE = os.environ.get("MINDCUBE_IMAGE_BASE", "/path/to/MindCube/data")

# Question types and their answer types
MCA_QUESTION_TYPES = {
    "relative_direction_object",
    "relative_distance_object",
    "relative_size_object",
    "relative_count",
    "appearance_order",
}
NA_QUESTION_TYPES = {
    "absolute_distance_object",
    "absolute_direction_object",
    "absolute_size_object",
    "absolute_size_room",
    "absolute_count",
}

# Target sample counts per question type (VSI-590K)
DEFAULT_VSI_TARGETS = {
    "relative_direction_object": 8000,
    "relative_distance_object": 7000,
    "relative_size_object": 5000,
    "absolute_distance_object": 5000,
    "absolute_direction_object": 5000,
    "absolute_size_object": 5000,
    "appearance_order": 5000,
    "absolute_count": 5000,
    "absolute_size_room": 3000,
    "relative_count": 2000,
}

# NOTE: SFT training data uses simplified prompts — just the raw question text
# (no REASONING PROTOCOL). The model learned Step1-5 format from SFT demonstrations,
# so GRPO prompts must match SFT format to avoid distribution mismatch.


# ---------------------------------------------------------------------------
# VSI-590K Processing
# ---------------------------------------------------------------------------

def load_vsi_data(path: str) -> List[Dict]:
    """Load VSI-590K JSONL data."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} VSI-590K records from {path}")
    return records


def stratified_sample_vsi(
    records: List[Dict],
    targets: Dict[str, int],
    seed: int = 42,
) -> List[Dict]:
    """
    Stratified sampling: by question_type, within each type by video source.
    """
    rng = random.Random(seed)

    # Group by question_type and video source
    groups = defaultdict(lambda: defaultdict(list))
    for rec in records:
        qt = rec.get("question_type", "unknown")
        src = rec["video"].split("/")[0]  # e.g., scannet, arkitscenes, scannetppv2
        groups[qt][src].append(rec)

    sampled = []
    for qt, target in targets.items():
        src_groups = groups.get(qt, {})
        if not src_groups:
            logger.warning(f"No data for question_type={qt}")
            continue

        total_available = sum(len(v) for v in src_groups.values())
        actual_target = min(target, total_available)

        # Proportional sampling within each source
        qt_sampled = []
        for src, src_records in src_groups.items():
            src_ratio = len(src_records) / total_available
            src_target = max(1, int(actual_target * src_ratio))
            src_target = min(src_target, len(src_records))
            qt_sampled.extend(rng.sample(src_records, src_target))

        # Adjust to exact target if needed
        if len(qt_sampled) > actual_target:
            qt_sampled = rng.sample(qt_sampled, actual_target)

        sampled.extend(qt_sampled)
        logger.info(
            f"  {qt}: {len(qt_sampled)}/{total_available} sampled "
            f"(target={target}, sources={list(src_groups.keys())})"
        )

    rng.shuffle(sampled)
    logger.info(f"Total VSI-590K sampled: {len(sampled)}")
    return sampled


def convert_vsi_to_grpo(record: Dict) -> Dict:
    """Convert a VSI-590K record to GRPO format.

    User message format matches SFT training data exactly:
        <video>\n{question text with options/instructions}
    No reasoning protocol — the model learned Step1-5 from SFT.
    """
    # Extract question text from conversation
    question_text = record["conversations"][0]["value"]
    ground_truth = record["conversations"][1]["value"]
    question_type = record.get("question_type", "unknown")
    video_rel = record["video"]

    # Determine answer type
    answer_type = "mca" if question_type in MCA_QUESTION_TYPES else "na"

    # Build user message: <video> + raw question (matches SFT format)
    # VSI-590K uses <image> prefix for video frames, replace with <video>
    query = question_text
    if query.startswith("<image>\n"):
        query = query[len("<image>\n"):]

    user_content = f"<video>\n{query}"

    # Build full video path
    video_path = os.path.join(VSI_VIDEO_BASE, video_rel)

    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "videos": [video_path],
        "solution": ground_truth,
        "question_type": question_type,
        "answer_type": answer_type,
        "source": "vsibench",
    }


# ---------------------------------------------------------------------------
# MindCube Processing
# ---------------------------------------------------------------------------

def load_mindcube_data(path: str) -> List[Dict]:
    """Load MindCube training JSONL data."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} MindCube records from {path}")
    return records


def convert_mindcube_to_grpo(record: Dict) -> Dict:
    """Convert a MindCube record to GRPO format.

    User message format matches SFT training data exactly:
        <image><image><image><image>\n{question text}
    """
    question_text = record["question"]
    ground_truth = record["gt_answer"]
    image_rels = record["images"]

    # Build image placeholders (concatenated, matching SFT format)
    image_placeholders = "<image>" * len(image_rels)

    # Build user message: image placeholders + raw question
    user_content = f"{image_placeholders}\n{question_text}"

    # Build full image paths
    image_paths = [os.path.join(MINDCUBE_IMAGE_BASE, img) for img in image_rels]

    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": image_paths,
        "solution": ground_truth,
        "question_type": "mindcube",
        "answer_type": "mca",
        "source": "mindcube",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_media_paths(record: Dict) -> bool:
    """Check that media files exist."""
    for vp in record.get("videos", []):
        if not os.path.exists(vp):
            return False
    for ip in record.get("images", []):
        if not os.path.exists(ip):
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build GRPO training dataset")
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: grpo/grpo_dataset.jsonl)"
    )
    parser.add_argument("--vsi-target", type=int, default=50000,
                        help="Total target samples from VSI-590K (default: 50000)")
    parser.add_argument("--vsi-only", action="store_true",
                        help="Only include VSI-590K data")
    parser.add_argument("--mindcube-only", action="store_true",
                        help="Only include MindCube data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip media path validation")
    args = parser.parse_args()

    output_path = args.output or str(
        Path(__file__).parent / "grpo_dataset.jsonl"
    )

    # Scale targets proportionally
    default_total = sum(DEFAULT_VSI_TARGETS.values())
    scale = args.vsi_target / default_total
    targets = {k: max(1, int(v * scale)) for k, v in DEFAULT_VSI_TARGETS.items()}

    all_samples = []

    # --- VSI-590K ---
    if not args.mindcube_only:
        logger.info("=== Processing VSI-590K ===")
        vsi_records = load_vsi_data(VSI_DATA_PATH)
        vsi_sampled = stratified_sample_vsi(vsi_records, targets, seed=args.seed)

        for rec in vsi_sampled:
            grpo_rec = convert_vsi_to_grpo(rec)
            all_samples.append(grpo_rec)

        logger.info(f"VSI-590K: {len(vsi_sampled)} samples converted")

    # --- MindCube ---
    if not args.vsi_only:
        logger.info("=== Processing MindCube ===")
        mc_records = load_mindcube_data(MINDCUBE_DATA_PATH)

        for rec in mc_records:
            grpo_rec = convert_mindcube_to_grpo(rec)
            all_samples.append(grpo_rec)

        logger.info(f"MindCube: {len(mc_records)} samples converted")

    # --- Validation ---
    if not args.skip_validation:
        logger.info("=== Validating media paths ===")
        valid_samples = []
        missing_count = 0
        for rec in all_samples:
            if validate_media_paths(rec):
                valid_samples.append(rec)
            else:
                missing_count += 1
        if missing_count > 0:
            logger.warning(f"Skipped {missing_count} samples with missing media files")
        all_samples = valid_samples

    # --- Shuffle and write ---
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Statistics ---
    stats = defaultdict(int)
    for rec in all_samples:
        stats[f"{rec['source']}/{rec['question_type']}"] += 1
        stats[f"total_{rec['source']}"] += 1

    logger.info(f"\n=== Dataset Summary ===")
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Output: {output_path}")
    for key in sorted(stats.keys()):
        logger.info(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
