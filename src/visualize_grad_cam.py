"""
Generates Grad-CAM saliency visualizations for the trained bearing fault
classifier, using real held-out test signals.

Usage:
    python visualize_grad_cam.py --checkpoint ../checkpoints/best_model.pt \
        --n_per_class 2 --out_dir ../grad_cam_outputs

Produces:
    - One PNG per visualized window: raw waveform with the Grad-CAM
      saliency curve overlaid (color-mapped), predicted vs. true class,
      and confidence.
    - grad_cam_summary.json: for each visualized window, the predicted
      class, true class, confidence, and the raw CAM array - this is the
      structured payload the React MES dashboard will consume later to
      render saliency on the live telemetry chart.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless - no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import make_dataloaders, CLASS_NAMES
from grad_cam import GradCAM1D
from model import build_model
from train import get_device


def plot_saliency(signal: np.ndarray, cam: np.ndarray, true_class: str, pred_class: str,
                   confidence: float, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    t = np.arange(len(signal))

    # Top: raw waveform, colored by saliency intensity via a scatter overlay
    axes[0].plot(t, signal, color="#4a4a4a", linewidth=0.6, zorder=1)
    sc = axes[0].scatter(t, signal, c=cam, cmap="inferno", s=4, zorder=2, vmin=0, vmax=1)
    axes[0].set_ylabel("Normalized amplitude")
    correctness = "CORRECT" if true_class == pred_class else "MISCLASSIFIED"
    axes[0].set_title(
        f"True: {true_class}  |  Predicted: {pred_class} ({confidence:.1%} confidence)  [{correctness}]",
        fontsize=11,
    )
    fig.colorbar(sc, ax=axes[0], label="Grad-CAM saliency", pad=0.01)

    # Bottom: saliency curve alone, easier to read the localization
    axes[1].fill_between(t, cam, color="#d62728", alpha=0.6)
    axes[1].set_ylabel("Saliency")
    axes[1].set_xlabel("Sample index (within 1024-sample window)")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    ap.add_argument("--out_dir", default="../grad_cam_outputs")
    ap.add_argument("--n_per_class", type=int, default=2,
                     help="how many test windows per class to visualize")
    ap.add_argument("--window_size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split_strategy", choices=["group", "cross_load"], default=None)
    ap.add_argument("--test_load", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    seed = args.seed if args.seed is not None else ckpt.get("seed", 42)
    split_strategy = args.split_strategy if args.split_strategy is not None else ckpt.get("split_strategy", "group")
    test_load = args.test_load if args.test_load is not None else ckpt.get("test_load", 3)

    _, _, test_loader, _ = make_dataloaders(
        raw_dir=args.raw_dir,
        batch_size=1,  # Grad-CAM here processes one window at a time
        window_size=args.window_size,
        stride=args.stride,
        seed=seed,
        split_strategy=split_strategy,
        test_load=test_load,
    )

    model = build_model(num_classes=ckpt["model_config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    class_names = ckpt.get("class_names", CLASS_NAMES)

    cam_tool = GradCAM1D(model)

    # collect up to n_per_class examples per true class
    per_class_count = {c: 0 for c in class_names}
    summary = []

    for xb, yb in test_loader:
        true_idx = int(yb.item())
        true_name = class_names[true_idx]
        if per_class_count[true_name] >= args.n_per_class:
            continue

        xb = xb.to(device)
        cam, pred_idx, probs = cam_tool.generate(xb)  # requires grad internally; do NOT wrap in no_grad
        pred_name = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        signal = xb.detach().cpu().numpy().squeeze()  # (window_size,) normalized signal

        fname = f"{true_name}_{'correct' if pred_name == true_name else 'WRONG_pred_' + pred_name}_{per_class_count[true_name]}.png"
        out_path = os.path.join(args.out_dir, fname)
        plot_saliency(signal, cam, true_name, pred_name, confidence, out_path)

        summary.append({
            "file": fname,
            "true_class": true_name,
            "predicted_class": pred_name,
            "confidence": confidence,
            "correct": pred_name == true_name,
            "cam": cam.tolist(),
        })

        per_class_count[true_name] += 1
        print(f"  saved {fname}  (confidence={confidence:.3f})")

        if all(count >= args.n_per_class for count in per_class_count.values()):
            break

    summary_path = os.path.join(args.out_dir, "grad_cam_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(summary)} visualizations to {args.out_dir}")
    print(f"Structured summary (for dashboard integration): {summary_path}")


if __name__ == "__main__":
    main()
