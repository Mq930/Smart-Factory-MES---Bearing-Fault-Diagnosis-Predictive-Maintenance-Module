"""
CWRU Bearing Dataset loader.

Key design decisions:
1. GROUP-WISE SPLIT: windows are split into train/val/test by SOURCE FILE,
   not by individual window. Adjacent windows from the same recording are
   highly correlated (overlapping or near-overlapping vibration segments),
   so splitting at the window level leaks information between train and
   test and inflates accuracy. Here, whole recordings (i.e. one class at
   one load) are assigned entirely to one split.
2. Each .mat file becomes many windows via a sliding window over the
   1D vibration signal (default: window=1024, stride=256, i.e. 75% overlap
   within a file - overlap is fine WITHIN a split, just not ACROSS splits).
3. Per-channel (DE accelerometer) z-score normalization, fit on train only.

Expected input layout (produced by download_cwru.py):
    data/raw/normal_load0_97.mat
    data/raw/inner_race_load0_105.mat
    data/raw/outer_race_load0_130.mat
    data/raw/ball_load0_118.mat
    ... etc for load 0-3
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["normal", "inner_race", "outer_race", "ball"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

WINDOW_SIZE = 1024
DEFAULT_STRIDE = 256  # 75% overlap within a recording


def _find_de_key(mat_dict: dict) -> str:
    """CWRU .mat files store the drive-end channel under a key like
    'X097_DE_time'. Find it regardless of the numeric prefix."""
    candidates = [k for k in mat_dict.keys() if k.endswith("_DE_time")]
    if not candidates:
        raise KeyError(
            f"No '*_DE_time' key found in mat file. Keys present: {list(mat_dict.keys())}"
        )
    return candidates[0]


def load_signal(mat_path: str) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    key = _find_de_key(mat)
    signal = mat[key].squeeze().astype(np.float32)
    return signal


def parse_filename(path: str) -> Tuple[str, int]:
    """'normal_load0_97.mat' -> ('normal', 0)"""
    base = os.path.basename(path)
    m = re.match(r"([a-z_]+)_load(\d)_\d+\.mat", base)
    if not m:
        raise ValueError(f"Unexpected filename format: {base}")
    cls_name, load = m.group(1), int(m.group(2))
    if cls_name not in CLASS_TO_IDX:
        raise ValueError(f"Unknown class '{cls_name}' parsed from {base}")
    return cls_name, load


def sliding_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    n_windows = (len(signal) - window_size) // stride + 1
    if n_windows <= 0:
        raise ValueError(f"Signal too short ({len(signal)}) for window {window_size}")
    windows = np.stack(
        [signal[i * stride: i * stride + window_size] for i in range(n_windows)]
    )
    return windows  # (n_windows, window_size)


@dataclass
class RecordingWindows:
    file_id: str            # unique id for the source recording, e.g. "inner_race_load2_107"
    class_name: str
    load: int
    windows: np.ndarray      # (n_windows, window_size)


def build_recordings(raw_dir: str, window_size: int = WINDOW_SIZE,
                      stride: int = DEFAULT_STRIDE) -> List[RecordingWindows]:
    mat_files = sorted(glob.glob(os.path.join(raw_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {raw_dir}. Run download_cwru.py first."
        )

    recordings = []
    for path in mat_files:
        cls_name, load = parse_filename(path)
        signal = load_signal(path)
        windows = sliding_windows(signal, window_size, stride)
        file_id = os.path.splitext(os.path.basename(path))[0]
        recordings.append(RecordingWindows(file_id, cls_name, load, windows))
    return recordings


def group_split(recordings: List[RecordingWindows], seed: int = 42,
                 val_frac: float = 0.15, test_frac: float = 0.15
                 ) -> Tuple[List[RecordingWindows], List[RecordingWindows], List[RecordingWindows]]:
    """
    Splits whole recordings (files) into train/val/test, stratified by class
    so every split sees every fault type, but no recording's windows appear
    in more than one split.

    With 4 loads per class, this typically yields ~2-3 train / ~0-1 val /
    ~0-1 test recordings per class - stratification below ensures at least
    one recording per class lands in val and test when possible.
    """
    rng = np.random.RandomState(seed)
    by_class = {}
    for rec in recordings:
        by_class.setdefault(rec.class_name, []).append(rec)

    train, val, test = [], [], []
    for cls_name, recs in by_class.items():
        recs = list(recs)
        rng.shuffle(recs)
        n = len(recs)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        # guard against over-allocating on small groups (e.g. n=4)
        n_val = min(n_val, n - n_test - 1) if n - n_test - 1 > 0 else max(0, n - n_test - 1)
        n_val = max(n_val, 1) if n - n_test >= 2 else 0

        test.extend(recs[:n_test])
        val.extend(recs[n_test:n_test + n_val])
        train.extend(recs[n_test + n_val:])

    return train, val, test


def cross_load_split(recordings: List[RecordingWindows], test_load: int = 3,
                      val_frac_of_train: float = 0.2, seed: int = 42):
    """
    Cross-load generalization split: a much harder, more honest protocol
    than group_split().

    - TEST  = all recordings at `test_load` HP, for every class. The model
      never sees this operating condition during training at all.
    - TRAIN/VAL = all recordings at the remaining loads, split by recording
      (never by window). Val is drawn from the SAME load range as train, so
      val accuracy will typically look better than test accuracy here.
      That gap is expected and is the point: it separates "did it memorize
      this load's fingerprint" from "did it learn the fault physics".

    With 4 loads per class (0-3 HP) and test_load=3, this leaves 3
    recordings per class for train/val: val gets 1 recording per class,
    train gets 2.
    """
    rng = np.random.RandomState(seed)
    by_class = {}
    for rec in recordings:
        by_class.setdefault(rec.class_name, []).append(rec)

    train, val, test = [], [], []
    for cls_name, recs in by_class.items():
        test_recs = [r for r in recs if r.load == test_load]
        remaining = [r for r in recs if r.load != test_load]
        rng.shuffle(remaining)

        n_val = max(1, round(len(remaining) * val_frac_of_train)) if len(remaining) > 1 else 0

        test.extend(test_recs)
        val.extend(remaining[:n_val])
        train.extend(remaining[n_val:])

    return train, val, test


class BearingWindowDataset(Dataset):
    """Flattened window-level dataset built from a list of RecordingWindows."""

    def __init__(self, recordings: List[RecordingWindows], mean: float = None, std: float = None):
        xs, ys, groups = [], [], []
        for rec in recordings:
            xs.append(rec.windows)
            ys.append(np.full(len(rec.windows), CLASS_TO_IDX[rec.class_name], dtype=np.int64))
            groups.extend([rec.file_id] * len(rec.windows))

        self.X = np.concatenate(xs, axis=0).astype(np.float32)  # (N, window_size)
        self.y = np.concatenate(ys, axis=0)
        self.groups = groups

        if mean is None or std is None:
            self.mean = float(self.X.mean())
            self.std = float(self.X.std() + 1e-8)
        else:
            self.mean = mean
            self.std = std

        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # (1, window_size) - single channel
        y = int(self.y[idx])
        return x, y


def make_dataloaders(raw_dir: str, batch_size: int = 64, window_size: int = WINDOW_SIZE,
                      stride: int = DEFAULT_STRIDE, seed: int = 42, num_workers: int = 2,
                      split_strategy: str = "group", test_load: int = 3
                      ) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    split_strategy:
        "group"       - random split by recording, stratified by class.
                         Easier protocol; loads are mixed across train/val/test.
        "cross_load"  - test set is an entirely unseen load (see cross_load_split).
                         Harder, more honest measure of generalization.
    """
    recordings = build_recordings(raw_dir, window_size, stride)

    if split_strategy == "group":
        train_recs, val_recs, test_recs = group_split(recordings, seed=seed)
    elif split_strategy == "cross_load":
        train_recs, val_recs, test_recs = cross_load_split(recordings, test_load=test_load, seed=seed)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}. Use 'group' or 'cross_load'.")

    print(f"Split strategy: {split_strategy}"
          + (f" (test_load={test_load})" if split_strategy == "cross_load" else ""))
    print("Split summary (by recording / source file):")
    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        ids = [r.file_id for r in recs]
        print(f"  {name}: {len(recs)} recordings -> {ids}")

    train_ds = BearingWindowDataset(train_recs)
    val_ds = BearingWindowDataset(val_recs, mean=train_ds.mean, std=train_ds.std)
    test_ds = BearingWindowDataset(test_recs, mean=train_ds.mean, std=train_ds.std)

    print(f"\nWindow counts -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    norm_stats = {"mean": train_ds.mean, "std": train_ds.std}
    return train_loader, val_loader, test_loader, norm_stats


if __name__ == "__main__":
    # Quick smoke test
    train_loader, val_loader, test_loader, stats = make_dataloaders(
        raw_dir="../data/raw", batch_size=32
    )
    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape, "Labels:", yb.shape)
    print("Normalization stats:", stats)
