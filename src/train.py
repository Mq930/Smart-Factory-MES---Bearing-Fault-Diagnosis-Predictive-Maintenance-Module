"""
Training script for the Multi-Scale CNN-Transformer bearing fault classifier.

Usage:
    python train.py --raw_dir ../data/raw --epochs 40 --batch_size 64

Saves:
    ../checkpoints/best_model.pt   - best val-accuracy checkpoint (full state)
    ../checkpoints/norm_stats.json - mean/std used for input normalization
                                     (needed at inference time)
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_dataloaders, CLASS_NAMES
from model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_n += xb.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--checkpoint_dir", default="../checkpoints")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--window_size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8, help="early stopping patience (epochs)")
    ap.add_argument("--split_strategy", choices=["group", "cross_load"], default="group",
                     help="'group' = random file-level split (easier). "
                          "'cross_load' = held-out load for test (harder, more honest).")
    ap.add_argument("--test_load", type=int, default=3,
                     help="which HP load (0-3) to hold out entirely when split_strategy=cross_load")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, norm_stats = make_dataloaders(
        raw_dir=args.raw_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        seed=args.seed,
        split_strategy=args.split_strategy,
        test_load=args.test_load,
    )

    with open(os.path.join(args.checkpoint_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
                "model_config": {
                    "num_classes": len(CLASS_NAMES),
                },
                "split_strategy": args.split_strategy,
                "test_load": args.test_load,
                "seed": args.seed,
            }, os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs.")
                break

    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print("Run evaluate.py to score the held-out test set with the best checkpoint.")


if __name__ == "__main__":
    main()
