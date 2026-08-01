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
from scipy.signal import spectrogram

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


def plot_spectrogram_overlay(signal: np.ndarray, cam: np.ndarray, true_class: str,
                              pred_class: str, confidence: float, out_path: str,
                              fs: int = 12000):
    """
    IMPORTANT - what this plot actually is:
    The saliency values (`cam`) come from the SAME 1D Grad-CAM computed on
    the raw waveform - nothing here is a new/separate "2D Grad-CAM" pass
    through a spectrogram-trained model. We are only changing the visual
    backdrop the existing 1D saliency curve is painted on, from a plain
    waveform to a spectrogram, because a spectrogram makes periodic
    fault-impact structure easier to see at a glance (impacts show up as
    vertical energy stripes at regular intervals). The saliency heatmap
    below has no independent frequency-axis information - each time-bin's
    color is the same 1D saliency value repeated down every frequency row.
    Do not describe this output as "2D Grad-CAM" - it is 1D Grad-CAM
    saliency overlaid on a spectrogram visualization.

    fs: sampling rate in Hz. CWRU 12k Drive End data is sampled at 12,000 Hz.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1]})

    # Use sample index (0..window_size) as the x-axis on ALL THREE panels,
    # matching plot_saliency()'s x-axis exactly, so the two PNGs for the
    # same window are directly, visually comparable feature-for-feature.
    # scipy.signal.spectrogram natively reports its time bins in seconds,
    # so we convert those bin centers back to sample-index units below
    # rather than switching the waveform panel to seconds.
    sample_idx = np.arange(len(signal))

    # --- Panel 1: raw waveform for reference ---
    axes[0].plot(sample_idx, signal, color="#333333", linewidth=0.7)
    correctness = "CORRECT" if true_class == pred_class else "MISCLASSIFIED"
    axes[0].set_title(
        f"True: {true_class}  |  Predicted: {pred_class} ({confidence:.1%} confidence)  [{correctness}]\n"
        f"1D Grad-CAM saliency overlaid on a spectrogram (saliency has no independent frequency info)",
        fontsize=10,
    )
    axes[0].set_ylabel("Amplitude")

    # --- Panel 2: spectrogram with saliency-colored overlay ---
    # nperseg chosen so we get a reasonable number of time bins across a
    # 1024-sample window without over-smoothing; noverlap for smoother
    # time resolution.
    freqs, times_spec, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=96)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    spec_sample_idx = times_spec * fs  # convert spectrogram's seconds back to sample-index units

    # background: the actual spectrogram, grayscale so the saliency overlay
    # (in color) is what draws the eye
    axes[1].pcolormesh(spec_sample_idx, freqs, Sxx_db, shading="gouraud", cmap="gray")

    # resample the 1D saliency curve (length = len(signal)) onto the
    # spectrogram's time-bin centers, then broadcast across all frequency
    # rows to build a 2D overlay - see docstring: this does NOT add new
    # frequency-axis information, it's the same 1D curve repainted.
    cam_resampled = np.interp(spec_sample_idx, sample_idx, cam)  # (n_time_bins,)
    cam_2d = np.tile(cam_resampled, (len(freqs), 1))  # (n_freqs, n_time_bins)

    overlay = axes[1].pcolormesh(spec_sample_idx, freqs, cam_2d, shading="gouraud",
                                  cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(overlay, ax=axes[1], label="Grad-CAM saliency (from 1D signal)", pad=0.01)

    # --- Panel 3: saliency curve alone, same as the standard plot ---
    axes[2].fill_between(sample_idx, cam, color="#d62728", alpha=0.6)
    axes[2].set_ylabel("Saliency")
    axes[2].set_xlabel("Sample index (within 1024-sample window) - matches the standard plot's x-axis")
    axes[2].set_ylim(0, 1.05)

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
    ap.add_argument("--sample_rate", type=int, default=12000,
                     help="sampling rate in Hz, for spectrogram axis labeling (CWRU 12k DE = 12000)")
    ap.add_argument("--no_spectrogram", action="store_true",
                     help="skip the extra spectrogram-overlay plot, only generate the standard waveform plot")
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

        spec_fname = None
        if not args.no_spectrogram:
            spec_fname = fname.replace(".png", "_spectrogram.png")
            spec_out_path = os.path.join(args.out_dir, spec_fname)
            plot_spectrogram_overlay(signal, cam, true_name, pred_name, confidence,
                                      spec_out_path, fs=args.sample_rate)

        summary.append({
            "file": fname,
            "spectrogram_file": spec_fname,
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
