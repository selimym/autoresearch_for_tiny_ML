"""
subset_sanity_check.py — One-time data-volume sanity check (pre-Phase 1).

Trains the Phase-1 baseline architecture for EPOCHS epochs on 5K, 10K, and
20K COCO-person subsets, evaluates mAP50, and prints a recommendation.

Usage:
    uv run subset_sanity_check.py [--epochs N] [--sizes 5000,10000,20000]

Defaults: 2 epochs, sizes 5000/10000/20000.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import the baseline build_model / fcos_loss from the gen-0 seed program
# ---------------------------------------------------------------------------
_GEN0 = Path(__file__).parent / "results" / "phase1" / "gen_0"
sys.path.insert(0, str(_GEN0))
from main import build_model, fcos_loss  # type: ignore[import]

from train_utils import load_backbone, make_dataloader, export_onnx
from evaluate_core import evaluate_model

# ---------------------------------------------------------------------------
# Baseline hyper-parameters (mirror gen-0 main.py)
# ---------------------------------------------------------------------------
NECK_CHANNELS = [96, 96, 96]
UIB_CONFIGS = [
    {"dw_kernel_size_start": 0, "dw_kernel_size_mid": 3, "dw_kernel_size_end": 0, "exp_ratio": 2.0},
    {"dw_kernel_size_start": 3, "dw_kernel_size_mid": 3, "dw_kernel_size_end": 0, "exp_ratio": 2.0},
    {"dw_kernel_size_start": 3, "dw_kernel_size_mid": 0, "dw_kernel_size_end": 0, "exp_ratio": 2.0},
]
HEAD_CHANNELS = 64
HEAD_STACKS    = 3
FROZEN_STAGES  = 4
LR             = 1e-3
BATCH_SIZE     = 8
IMG_SIZE       = 320
STRIDES        = [8, 16, 32]


# ---------------------------------------------------------------------------
# Training + evaluation for a single subset size
# ---------------------------------------------------------------------------

def run_for_size(subset_size: int, epochs: int, device: torch.device) -> dict:
    """Train baseline for `epochs` epochs on `subset_size` images, return metrics."""
    print(f"\n{'='*60}")
    print(f"  subset_size={subset_size:,}  epochs={epochs}")
    print(f"{'='*60}")

    backbone, backbone_channels = load_backbone(frozen_stages=FROZEN_STAGES)
    FPNNeck, FCOSHead, TinyDetector = build_model(backbone_channels)
    neck  = FPNNeck(backbone_channels, NECK_CHANNELS, UIB_CONFIGS)
    head  = FCOSHead(NECK_CHANNELS, HEAD_CHANNELS, HEAD_STACKS)
    model = TinyDetector(backbone, neck, head).to(device)

    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    train_loader = make_dataloader("train", BATCH_SIZE, IMG_SIZE, subset_size=subset_size)

    warmup_epochs = 0.5
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=epochs * len(train_loader),
        pct_start=warmup_epochs / epochs,
    )

    model.train()
    total_steps = epochs * len(train_loader)
    step = 0
    t0 = time.perf_counter()

    with tqdm(total=total_steps, unit="batch",
              desc=f"Training [{subset_size//1000}K]") as pbar:
        for epoch in range(epochs):
            for imgs, targets in train_loader:
                imgs_t = torch.stack(imgs).to(device)
                tgts   = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                cls_preds, reg_preds, ctr_preds = model(imgs_t)
                loss = fcos_loss(cls_preds, reg_preds, ctr_preds, tgts, STRIDES)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                if step % 50 == 0 or step == total_steps:
                    pbar.set_postfix(
                        epoch=f"{epoch+1}/{epochs}",
                        loss=f"{loss.item():.4f}",
                    )
                pbar.update(1)

    train_time = time.perf_counter() - t0
    print(f"  Training done in {train_time/60:.1f} min")

    # Export to a temp ONNX and evaluate
    onnx_path = f"/tmp/sanity_check_{subset_size}.onnx"
    model.eval()
    export_onnx(model, onnx_path, IMG_SIZE)
    metrics = evaluate_model(onnx_path)
    metrics["train_time_min"] = train_time / 60
    metrics["subset_size"]    = subset_size
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Subset size sanity check")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per subset (default: 2)")
    parser.add_argument("--sizes", type=str, default="5000,10000,20000",
                        help="Comma-separated subset sizes (default: 5000,10000,20000)")
    args = parser.parse_args()

    sizes  = [int(s) for s in args.sizes.split(",")]
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Subset sizes: {sizes}")
    print(f"Epochs per run: {epochs}")

    results = []
    for sz in sizes:
        metrics = run_for_size(sz, epochs, device)
        results.append(metrics)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUBSET SIZE SANITY CHECK — RESULTS")
    print(f"{'='*60}")
    header = f"{'Size':>8}  {'mAP50':>7}  {'Size MB':>8}  {'Train min':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['subset_size']:>8,}  "
            f"{r['mAP50']:>7.4f}  "
            f"{r['model_size_mb']:>8.2f}  "
            f"{r['train_time_min']:>10.1f}"
        )

    # ---------------------------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------------------------
    if len(results) >= 2:
        print()
        maps = [r["mAP50"] for r in results]
        gains = [maps[i+1] - maps[i] for i in range(len(maps) - 1)]

        print("  mAP50 gains between sizes:")
        for i, g in enumerate(gains):
            print(f"    {sizes[i]:,} → {sizes[i+1]:,}: {g:+.4f}")

        # Decision: if gain from 5K→10K > half the gain from 10K→20K's
        # diminishing-returns threshold, recommend bumping.
        first_gain = gains[0]
        if len(gains) >= 2:
            second_gain = gains[1]
            if first_gain > 0.01 and first_gain > second_gain * 0.8:
                print("\n  RECOMMENDATION: 5K is on the steep part of the curve.")
                print("  → Bump to 10K for Phase 1 experiments.")
            else:
                print("\n  RECOMMENDATION: 5K and 10K are close.")
                print("  → Proceed with 5K.")
        else:
            if first_gain > 0.01:
                print(f"\n  RECOMMENDATION: mAP50 gain {first_gain:+.4f} suggests bumping to {sizes[1]:,}.")
            else:
                print(f"\n  RECOMMENDATION: mAP50 gain {first_gain:+.4f} is small — proceed with {sizes[0]:,}.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
