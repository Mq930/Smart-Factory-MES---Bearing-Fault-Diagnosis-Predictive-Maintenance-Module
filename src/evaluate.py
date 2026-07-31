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

from dataset import make_dataloaders, CLASS_NAMES
from model import build_model
from train import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # NOTE: must rebuild loaders with the same seed/window/stride used during
    # training so the test split matches exactly.
    _, _, test_loader, _ = make_dataloaders(
        raw_dir=args.raw_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        seed=args.seed,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(num_classes=ckpt["model_config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = ckpt.get("class_names", CLASS_NAMES)

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})\n")

    preds, labels, probs = collect_predictions(model, test_loader, device)

    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    print("Confusion matrix (rows=ground truth, cols=predicted):")
    print_confusion_matrix(cm, class_names)

    results = {
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
