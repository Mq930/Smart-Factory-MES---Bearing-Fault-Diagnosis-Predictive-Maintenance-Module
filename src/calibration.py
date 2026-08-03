"""
Confidence calibration for the bearing fault classifier via temperature
scaling (Guo et al., 2017, "On Calibration of Modern Neural Networks").

WHY THIS MATTERS FOR THIS PROJECT SPECIFICALLY:
The MOM agent's route_severity() node makes a hard decision using
mean_confidence < HIGH_CONFIDENCE_THRESHOLD (0.9) to distinguish a
"watch" alert from "alert"/"critical". If the model's raw softmax
confidence is systematically overconfident (a well-documented property of
modern deep nets - see the paper above), that threshold doesn't mean what
it looks like it means: predictions the model calls "95% confident" might
only be correct 80% of the time. Temperature scaling fixes this WITHOUT
changing any prediction (argmax is invariant to temperature), so accuracy
numbers you already reported are untouched - only the confidence VALUES
become trustworthy probabilities.

WHAT TEMPERATURE SCALING IS:
A single scalar T > 0, fit on the VALIDATION set (never test - that would
leak calibration information into your final reported numbers), that
divides the logits before softmax:
    calibrated_probs = softmax(logits / T)
T is found by minimizing negative log-likelihood (NLL) on validation
logits/labels, holding the trained model's weights completely fixed.
T > 1 softens (reduces) overconfident predictions; T < 1 sharpens them;
T = 1 is a no-op (equivalent to the original uncalibrated model).

RELIABILITY DIAGRAM + ECE:
Bins predictions by confidence into ~10 bins, and for each bin plots
(mean predicted confidence) vs. (actual accuracy of predictions in that
bin). A perfectly calibrated model's bars sit exactly on the y=x diagonal.
Expected Calibration Error (ECE) is the bin-size-weighted average gap
between confidence and accuracy - the standard scalar summary of
calibration quality (lower is better, 0 = perfect).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import make_dataloaders, CLASS_NAMES
from model import build_model
from train import get_device


class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and learns a single scalar temperature T.
    The base model's weights are NOT touched - only T is optimized.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # start at T=1.0 (no-op) and optimize from there
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        # parameterize in log-space so T is always positive (T = exp(log_T)),
        # which keeps the optimizer well-behaved near T=1
        return torch.exp(self.log_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature

    @torch.no_grad()
    def collect_logits(self, loader, device) -> tuple:
        """Runs the base model (uncalibrated) over a loader and collects
        raw logits + labels, for fitting or evaluating calibration."""
        self.model.eval()
        all_logits, all_labels = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            logits = self.model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb)
        return torch.cat(all_logits), torch.cat(all_labels)

    def fit(self, val_loader, device, max_iter: int = 100, lr: float = 0.01):
        """
        Fits self.log_temperature by minimizing NLL on the validation set.
        Uses LBFGS (standard choice for this - it's a 1-parameter convex-ish
        optimization, converges in a handful of steps).
        """
        self.to(device)
        val_logits, val_labels = self.collect_logits(val_loader, device)
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss

        before_nll = nll_criterion(val_logits, val_labels).item()
        optimizer.step(closure)
        after_nll = nll_criterion(val_logits / self.temperature, val_labels).item()

        print(f"Temperature scaling fit complete:")
        print(f"  T = {self.temperature.item():.4f}")
        print(f"  Validation NLL before: {before_nll:.4f}  ->  after: {after_nll:.4f}")
        return self.temperature.item()


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
                 n_bins: int = 10) -> tuple:
    """
    Returns (ece, bin_confidences, bin_accuracies, bin_counts) for a
    reliability diagram. Bins are equal-width over [0, 1] confidence range.
    """
    correct = (predictions == labels).astype(np.float32)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_confidences, bin_accuracies, bin_counts = [], [], []
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin is inclusive on the right edge
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            bin_confidences.append(np.nan)
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        bin_counts.append(int(count))
        ece += (count / n) * abs(bin_acc - bin_conf)

    return ece, np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)


def plot_reliability_diagram(bin_confidences: np.ndarray, bin_accuracies: np.ndarray,
                              bin_counts: np.ndarray, ece: float, title: str, out_path: str):
    n_bins = len(bin_confidences)
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})

    # perfect calibration reference line
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    valid = ~np.isnan(bin_accuracies)
    ax1.bar(bin_centers[valid], bin_accuracies[valid], width=1 / n_bins, alpha=0.7,
            edgecolor="black", color="#4c72b0", label="Model accuracy per bin")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"{title}\nECE = {ece:.4f}")
    ax1.legend(loc="upper left")

    ax2.bar(bin_centers[valid], bin_counts[valid], width=1 / n_bins, color="#888888")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("# samples")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    ap.add_argument("--out_dir", default="../checkpoints")
    ap.add_argument("--window_size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split_strategy", choices=["group", "cross_load"], default=None)
    ap.add_argument("--test_load", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    seed = args.seed if args.seed is not None else ckpt.get("seed", 42)
    split_strategy = args.split_strategy if args.split_strategy is not None else ckpt.get("split_strategy", "group")
    test_load = args.test_load if args.test_load is not None else ckpt.get("test_load", 3)

    train_loader, val_loader, test_loader, _ = make_dataloaders(
        raw_dir=args.raw_dir, batch_size=args.batch_size,
        window_size=args.window_size, stride=args.stride,
        seed=seed, split_strategy=split_strategy, test_load=test_load,
    )

    model = build_model(num_classes=ckpt["model_config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    class_names = ckpt.get("class_names", CLASS_NAMES)

    scaler = TemperatureScaler(model)

    # --- BEFORE calibration: reliability diagram on TEST set using raw softmax ---
    print("\nEvaluating BEFORE calibration (T=1.0, raw softmax) on test set...")
    test_logits_raw, test_labels = scaler.collect_logits(test_loader, device)
    probs_before = F.softmax(test_logits_raw, dim=1).numpy()
    preds_before = probs_before.argmax(axis=1)
    conf_before = probs_before.max(axis=1)
    labels_np = test_labels.numpy()

    ece_before, bc_before, ba_before, cnt_before = compute_ece(conf_before, preds_before, labels_np)
    print(f"  Test accuracy: {(preds_before == labels_np).mean():.4f}")
    print(f"  ECE before calibration: {ece_before:.4f}")

    plot_reliability_diagram(
        bc_before, ba_before, cnt_before, ece_before,
        "Reliability Diagram - BEFORE Calibration (T=1.0)",
        os.path.join(args.out_dir, "reliability_before.png"),
    )

    # --- Fit temperature on VALIDATION set ---
    print("\nFitting temperature scaling on validation set...")
    T = scaler.fit(val_loader, device)

    # --- AFTER calibration: reliability diagram on TEST set using calibrated softmax ---
    print("\nEvaluating AFTER calibration on test set...")
    probs_after = F.softmax(test_logits_raw.to(device) / scaler.temperature, dim=1).detach().cpu().numpy()
    preds_after = probs_after.argmax(axis=1)  # identical to preds_before - argmax is temperature-invariant
    conf_after = probs_after.max(axis=1)

    assert np.array_equal(preds_before, preds_after), (
        "Predictions changed after temperature scaling - this should be impossible "
        "(temperature scaling only rescales logits uniformly, argmax is invariant). "
        "If this assertion fires, something is wrong with the implementation."
    )

    ece_after, bc_after, ba_after, cnt_after = compute_ece(conf_after, preds_after, labels_np)
    print(f"  Test accuracy: {(preds_after == labels_np).mean():.4f}  (unchanged, as expected)")
    print(f"  ECE after calibration: {ece_after:.4f}")

    plot_reliability_diagram(
        bc_after, ba_after, cnt_after, ece_after,
        f"Reliability Diagram - AFTER Calibration (T={T:.3f})",
        os.path.join(args.out_dir, "reliability_after.png"),
    )

    print(f"\nECE improvement: {ece_before:.4f} -> {ece_after:.4f} "
          f"({'better' if ece_after < ece_before else 'WORSE - check val/test split size'})")

    # --- Save temperature into the checkpoint's sidecar file, so evaluate.py
    # and the MOM agent's classifier_tool.py can both apply it consistently
    # without needing to refit every time. ---
    calib_path = os.path.join(args.out_dir, "calibration.json")
    with open(calib_path, "w") as f:
        json.dump({
            "temperature": T,
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "fit_on_split": split_strategy,
            "fit_seed": seed,
        }, f, indent=2)
    print(f"\nSaved temperature + ECE results to {calib_path}")
    print("Reliability diagrams saved to reliability_before.png / reliability_after.png")


if __name__ == "__main__":
    main()
