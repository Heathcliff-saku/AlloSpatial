"""
SceneMemoryCache — in-memory transport between pipeline stages.

When enabled (config.use_memory_cache = True), DA3 and SAM3 intermediate
results are kept in memory rather than written to disk and read back.
The final AST yaml + visualisations are still written to the workspace as
usual; only the intermediate arrays are bypassed.

Lifecycle (model_service.py):
    cache = SceneMemoryCache()
    run_depth_estimation(..., memory_cache=cache)
    run_segmentation(..., memory_cache=cache)
    run_mapping(..., memory_cache=cache)
    run_ast_generation(..., memory_cache=cache)
    cache.clear()          # ← explicit release; GC is immediate
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SceneMemoryCache:
    # ── DA3 outputs (keyed by frame_name, e.g. "frame_000001") ───────────────
    da3_depths:     Dict[str, np.ndarray] = field(default_factory=dict)
    # [H,W] float32 — confidence map (None when model produces no conf)
    da3_confs:      Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    da3_intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)   # [3,3]
    da3_poses:      Dict[str, np.ndarray] = field(default_factory=dict)   # [4,4] c2w

    # Replaces conf_summary.npy
    conf_summary: Optional[dict] = None

    # ── SAM3 outputs ──────────────────────────────────────────────────────────
    # [C, H, W] uint8  (one entry per non-skipped frame)
    sam3_masks:      Dict[str, np.ndarray] = field(default_factory=dict)
    sam3_class_info: Optional[dict] = None      # replaces class_info.npy

    # ── Point-cloud (GlobalMapping → generate_cognitive_ast) ─────────────────
    pc_points:       Optional[np.ndarray] = None  # [N,3] float32
    pc_colors:       Optional[np.ndarray] = None  # [N,3] float32
    pc_labels:       Optional[np.ndarray] = None  # [N,]  int32
    pc_class_names:  Optional[List[str]] = None
    pc_scene_bounds: Optional[dict] = None

    # ── Book-keeping ──────────────────────────────────────────────────────────
    def has_da3(self) -> bool:
        return bool(self.da3_depths)

    def has_sam3(self) -> bool:
        return bool(self.sam3_masks)

    def has_pointcloud(self) -> bool:
        return self.pc_points is not None

    def clear(self) -> None:
        """Explicitly release all held numpy arrays (do not wait for GC)."""
        self.da3_depths.clear()
        self.da3_confs.clear()
        self.da3_intrinsics.clear()
        self.da3_poses.clear()
        self.conf_summary = None
        self.sam3_masks.clear()
        self.sam3_class_info = None
        self.pc_points = None
        self.pc_colors = None
        self.pc_labels = None
        self.pc_class_names = None
        self.pc_scene_bounds = None
        gc.collect()
