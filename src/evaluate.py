"""
Evaluates the best checkpoint on the held-out test set.

Usage:
    python evaluate.py --raw_dir ../data/raw --checkpoint ../checkpoints/best_model.pt
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataset import make_dataloaders, make_noisy_test_loader, CLASS_NAMES
from model import build_model
from train import get_device


@torch.no_grad()
def collect_predictions(model, loader, device, temperature: float = 1.0):
    """
    temperature: divides logits before softmax (see calibration.py). 1.0 is
    a no-op (uncalibrated, original behavior). Predictions (argmax) are
    identical regardless of temperature - only the confidence VALUES in
    all_probs change.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb) / temperature
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
        all_probs.append(probs.cpu().numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def print_confusion_matrix(cm: np.ndarray, class_names):
    header = "gt\\pred".ljust(14) + "".join(c[:10].ljust(12) for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        line = class_names[i][:12].ljust(14) + "".join(str(v).ljust(12) for v in row)
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window_size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None,
                     help="overrides the seed stored in the checkpoint, if provided")
    ap.add_argument("--split_strategy", choices=["group", "cross_load"], default=None,
                     help="overrides the split strategy stored in the checkpoint, if provided")
    ap.add_argument("--test_load", type=int, default=None,
                     help="overrides the test_load stored in the checkpoint, if provided")
    ap.add_argument("--noise_levels", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.8],
                     help="additional Gaussian noise std levels (normalized units) to sweep over "
                          "for a robustness curve, evaluated on top of the clean test result. "
                          "Pass --noise_levels with no values to skip this sweep.")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    # Rebuild loaders with the SAME split settings used during training so the
    # test split matches exactly. Older checkpoints (pre split_strategy) fall
    # back to "group" for backward compatibility. CLI flags override if given.
    seed = args.seed if args.seed is not None else ckpt.get("seed", 42)
    split_strategy = args.split_strategy if args.split_strategy is not None else ckpt.get("split_strategy", "group")
    test_load = args.test_load if args.test_load is not None else ckpt.get("test_load", 3)

    _, _, test_loader, norm_stats = make_dataloaders(
        raw_dir=args.raw_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        seed=seed,
        split_strategy=split_strategy,
        test_load=test_load,
    )

    model = build_model(num_classes=ckpt["model_config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = ckpt.get("class_names", CLASS_NAMES)

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})\n")

    # Apply temperature scaling if calibration.py has been run for this
    # checkpoint (calibration.json sits alongside best_model.pt). Falls back
    # to T=1.0 (uncalibrated, original behavior) if not found - so this is
    # backward compatible with checkpoints that haven't been calibrated yet.
    temperature = 1.0
    calib_path = os.path.join(os.path.dirname(args.checkpoint), "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f)
        temperature = calib["temperature"]
        print(f"Applying temperature scaling: T={temperature:.4f} "
              f"(from {calib_path}, fit ECE {calib['ece_before']:.4f} -> {calib['ece_after']:.4f})\n")
    else:
        print(f"No calibration.json found at {calib_path} - using uncalibrated "
              f"confidence (T=1.0). Run calibration.py to fit temperature scaling.\n")

    preds, labels, probs = collect_predictions(model, test_loader, device, temperature=temperature)

    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    print("Confusion matrix (rows=ground truth, cols=predicted):")
    print_confusion_matrix(cm, class_names)

    # --- Noise robustness sweep ---
    # Clean test accuracy alone doesn't tell you how the model behaves on a
    # real, noisier factory floor. Re-evaluate the SAME test recordings with
    # injected Gaussian noise at increasing severity to get a degradation
    # curve. Uses the training set's mean/std (norm_stats) - never refit on
    # noisy data, since that would defeat the point.
    noise_results = {}
    if args.noise_levels:
        print("\nNoise robustness sweep (Gaussian noise added at eval time only):")
        print(f"  {'noise_std':<12}{'accuracy':<12}{'macro_f1':<12}")
        print(f"  {'0.0 (clean)':<12}{acc:<12.4f}{macro_f1:<12.4f}")
        for noise_std in args.noise_levels:
            noisy_loader = make_noisy_test_loader(
                raw_dir=args.raw_dir,
                mean=norm_stats["mean"],
                std=norm_stats["std"],
                noise_std=noise_std,
                batch_size=args.batch_size,
                window_size=args.window_size,
                stride=args.stride,
                seed=seed,
                split_strategy=split_strategy,
                test_load=test_load,
            )
            n_preds, n_labels, _ = collect_predictions(model, noisy_loader, device, temperature=temperature)
            n_acc = (n_preds == n_labels).mean()
            n_f1 = f1_score(n_labels, n_preds, average="macro")
            noise_results[noise_std] = {"accuracy": float(n_acc), "macro_f1": float(n_f1)}
            print(f"  {noise_std:<12}{n_acc:<12.4f}{n_f1:<12.4f}")

    results = {
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "noise_robustness": noise_results,
        "temperature_applied": temperature,
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
