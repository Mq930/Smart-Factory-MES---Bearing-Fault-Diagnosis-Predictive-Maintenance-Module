"""
Wraps the trained MultiScaleCNNTransformer checkpoint to classify a BATCH
of consecutive vibration windows (simulating N seconds of rolling sensor
telemetry) and return per-window predictions plus Grad-CAM saliency.

This module has no LangGraph dependency - it's a plain inference utility
that graph.py's ingest node calls.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model          # noqa: E402
from grad_cam import GradCAM1D         # noqa: E402
from dataset import (                  # noqa: E402
    build_recordings, CLASS_NAMES, WINDOW_SIZE, DEFAULT_STRIDE,
)


@dataclass
class WindowPrediction:
    window_index: int          # position within the batch (0-indexed, chronological)
    predicted_class: str
    confidence: float
    all_probs: dict            # class_name -> probability, for the full distribution
    cam: np.ndarray = field(repr=False)  # (window_size,) saliency, kept for potential dashboard use


class BearingClassifierTool:
    """
    Loads a trained checkpoint once, then classifies batches of windows on
    demand. Designed to be instantiated once per agent session and reused
    across graph invocations (loading the checkpoint is the expensive part).
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        self.model = build_model(num_classes=ckpt["model_config"]["num_classes"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.norm_mean = None
        self.norm_std = None
        self._load_norm_stats(checkpoint_path)

        self.cam_tool = GradCAM1D(self.model)

    def _load_norm_stats(self, checkpoint_path: str):
        """norm_stats.json is saved alongside best_model.pt by train.py."""
        import json
        norm_path = os.path.join(os.path.dirname(checkpoint_path), "norm_stats.json")
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                stats = json.load(f)
            self.norm_mean = stats["mean"]
            self.norm_std = stats["std"]
        else:
            raise FileNotFoundError(
                f"norm_stats.json not found next to {checkpoint_path}. "
                "This file is required to normalize raw signal windows the same "
                "way they were normalized during training - re-run train.py if missing."
            )

    def classify_batch(self, windows: np.ndarray) -> List[WindowPrediction]:
        """
        windows: (N, window_size) raw (un-normalized) vibration windows,
                 in chronological order.
        Returns: list of N WindowPrediction, one per window, in the same order.
        """
        results = []
        normalized = (windows - self.norm_mean) / self.norm_std

        for i, w in enumerate(normalized):
            x = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            cam, pred_idx, probs = self.cam_tool.generate(x)

            all_probs = {self.class_names[j]: float(probs[j]) for j in range(len(self.class_names))}
            results.append(WindowPrediction(
                window_index=i,
                predicted_class=self.class_names[pred_idx],
                confidence=float(probs[pred_idx]),
                all_probs=all_probs,
                cam=cam,
            ))
        return results


def load_windows_from_recording(raw_dir: str, class_name: str, load: int,
                                 n_windows: int = 10, start_at: int = 0,
                                 window_size: int = WINDOW_SIZE, stride: int = DEFAULT_STRIDE
                                 ) -> np.ndarray:
    """
    Convenience loader for DEMO/TEST purposes: pulls N consecutive windows
    from a specific real CWRU recording, simulating what a rolling sensor
    feed would deliver over time. In a real deployment this function would
    be replaced by a live sensor ingestion pipeline - the rest of the agent
    (classify_batch onward) is agnostic to where the windows came from.

    class_name: e.g. "inner_race_014", must match a CLASS_NAMES entry.
    load: which HP load's recording to pull from (0-3).
    """
    recordings = build_recordings(raw_dir, window_size, stride)
    matches = [r for r in recordings if r.class_name == class_name and r.load == load]
    if not matches:
        available = sorted({(r.class_name, r.load) for r in recordings})
        raise ValueError(
            f"No recording found for class={class_name}, load={load}. "
            f"Available (class, load) combos: {available}"
        )
    rec = matches[0]
    if start_at + n_windows > len(rec.windows):
        raise ValueError(
            f"Requested windows [{start_at}:{start_at + n_windows}] exceed "
            f"recording length ({len(rec.windows)} windows available)."
        )
    return rec.windows[start_at: start_at + n_windows]
