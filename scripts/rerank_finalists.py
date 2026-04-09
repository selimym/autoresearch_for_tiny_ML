"""
rerank_finalists.py — Phase 1.5: re-evaluate top Phase 1 candidates with a
longer, controlled training schedule and multiple seeds.

Addresses the undertraining bias identified in the feedback: Phase 1 uses only
4 epochs on a frozen backbone and a 5K subset, which can produce noisy
architecture rankings.  This script retrains each finalist with:
  - a configurable number of epochs (default: 10)
  - partial backbone unfreezing (default: FROZEN_STAGES=2)
  - N random seeds (default: 2)

and logs mean ± std mAP50 per candidate to results/phase15.tsv.

Usage:
    uv run scripts/rerank_finalists.py [--top-k 5] [--seeds 2] [--epochs 10]
                                       [--frozen-stages 2] [--batch-size 8]

Reads:  results/phase1.tsv  (or --tsv path)
Writes: results/phase15.tsv
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Project root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from train_utils import load_backbone, make_dataloader, export_onnx
from evaluate_core import evaluate_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tsv(path: Path) -> list[dict]:
    """Read a TSV file (with header) into a list of dicts."""
    rows = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    return rows


def _load_program_from_results(gen_dir: Path):
    """Import build_model and fcos_loss from a Phase 1 generation directory.

    Each generation directory should contain a file called 'main.py' (the
    ShinkaEvolve convention) or 'initial_compress.py'.
    """
    for candidate in ["main.py", "initial_compress.py"]:
        prog = gen_dir / candidate
        if prog.exists():
            spec = importlib.util.spec_from_file_location("_phase1_program", prog)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise FileNotFoundError(f"No program file found in {gen_dir}")


def _find_candidate_dir(results_dir: Path, commit_hash: str) -> Path | None:
    """Search Phase 1 results directory for a generation containing the commit."""
    for gen_dir in sorted(results_dir.glob("gen_*")):
        for prog in gen_dir.glob("*.py"):
            if commit_hash[:8] in prog.stem or commit_hash[:8] in gen_dir.name:
                return gen_dir
        # Also try commit hash as subdirectory name
        candidate = gen_dir / commit_hash[:8]
        if candidate.is_dir():
            return candidate
    return None


def _train_candidate(
    mod,
    epochs: int,
    frozen_stages: int,
    batch_size: int,
    lr: float,
    warmup_epochs: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Retrain a Phase 1 candidate with a controlled recipe and evaluate it.

    Returns the evaluate_model metrics dict.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load architecture constants from the candidate module
    img_size = getattr(mod, "IMG_SIZE", 320)
    neck_channels = getattr(mod, "NECK_CHANNELS", [96, 96, 96])
    uib_configs = getattr(mod, "UIB_CONFIGS", [])
    head_channels = getattr(mod, "HEAD_CHANNELS", 64)
    head_stacks = getattr(mod, "HEAD_STACKS", 3)

    backbone, backbone_channels = load_backbone(frozen_stages=frozen_stages)
    FPNNeck, FCOSHead, TinyDetector = mod.build_model(backbone_channels)
    neck = FPNNeck(backbone_channels, neck_channels, uib_configs)
    head = FCOSHead(neck_channels, head_channels, head_stacks)
    model = TinyDetector(backbone, neck, head).to(device)

    strides = [8, 16, 32]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    train_loader = make_dataloader("train", batch_size, img_size)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * len(train_loader),
        pct_start=warmup_epochs / max(epochs, 1),
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"  epoch {epoch+1}/{epochs}", leave=False):
            imgs_t = torch.stack(imgs).to(device)
            targets = [{k: v.to(device) if hasattr(v, "to") else v
                        for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            cls_preds, reg_preds, ctr_preds = model(imgs_t)
            loss = mod.fcos_loss(cls_preds, reg_preds, ctr_preds, targets, strides)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

    del train_loader, optimizer, scheduler
    gc.collect()

    model.eval()
    onnx_path = f"checkpoints/rerank_seed{seed}.onnx"
    export_onnx(model, onnx_path, img_size)
    metrics = evaluate_model(onnx_path)

    del model, backbone
    gc.collect()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5: rerank finalist candidates")
    parser.add_argument("--tsv", default="results/phase1.tsv",
                        help="Path to Phase 1 results TSV (default: results/phase1.tsv)")
    parser.add_argument("--results-dir", default="results/phase1",
                        help="Phase 1 ShinkaEvolve results directory")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top Phase 1 candidates to rerank (default: 5)")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Number of random seeds per candidate (default: 2)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs for reranking (default: 10)")
    parser.add_argument("--frozen-stages", type=int, default=2,
                        help="Backbone frozen stages (default: 2; allows more unfreezing)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for reranking (default: 5e-4)")
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--out", default="results/phase15.tsv",
                        help="Output TSV path (default: results/phase15.tsv)")
    args = parser.parse_args()

    tsv_path = _ROOT / args.tsv
    results_dir = _ROOT / args.results_dir
    out_path = _ROOT / args.out

    if not tsv_path.exists():
        print(f"ERROR: Phase 1 TSV not found: {tsv_path}")
        print("Run Phase 1 first, or pass --tsv to point to the right file.")
        sys.exit(1)

    rows = _load_tsv(tsv_path)
    if not rows:
        print(f"ERROR: No data in {tsv_path}")
        sys.exit(1)

    # Sort by score descending, take top-k
    def _score(r: dict) -> float:
        try:
            return float(r.get("score", 0) or 0)
        except ValueError:
            return 0.0

    candidates = sorted(rows, key=_score, reverse=True)[: args.top_k]
    print(f"Reranking top-{len(candidates)} Phase 1 candidates "
          f"({args.seeds} seeds × {args.epochs} epochs each)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Write TSV header
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("rank\tcommit\tseed\tmAP50\tmAP\tAP_small\tmodel_size_mb\tcpu_latency_ms\tscore\n")

    summary: list[dict] = []

    for rank, cand in enumerate(candidates, 1):
        commit = cand.get("commit", f"candidate_{rank}")
        phase1_score = cand.get("score", "?")
        phase1_map50 = cand.get("mAP50", "?")
        print(f"[{rank}/{len(candidates)}] commit={commit}  "
              f"phase1_score={phase1_score}  phase1_mAP50={phase1_map50}")

        # Locate the program file for this candidate
        cand_dir = _find_candidate_dir(results_dir, commit)
        if cand_dir is None:
            # Fall back: use the seed program (initial_compress.py)
            cand_dir = _ROOT
            print(f"  WARNING: could not find generation dir for {commit[:8]}, "
                  f"falling back to initial_compress.py")
        try:
            mod = _load_program_from_results(cand_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        seed_maps: list[float] = []
        for seed in range(1, args.seeds + 1):
            print(f"  seed {seed}/{args.seeds} ...")
            t0 = time.perf_counter()
            try:
                metrics = _train_candidate(
                    mod=mod,
                    epochs=args.epochs,
                    frozen_stages=args.frozen_stages,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    warmup_epochs=args.warmup_epochs,
                    seed=seed,
                    device=device,
                )
            except Exception as exc:
                print(f"  ERROR (seed {seed}): {exc}")
                continue
            elapsed = time.perf_counter() - t0
            seed_maps.append(metrics["mAP50"])
            print(f"    mAP50={metrics['mAP50']:.4f}  mAP={metrics['mAP']:.4f}  "
                  f"AP_small={metrics['AP_small']:.4f}  "
                  f"score={metrics['score']:.4f}  ({elapsed:.0f}s)")
            with open(out_path, "a") as f:
                f.write(
                    f"{rank}\t{commit}\t{seed}\t"
                    f"{metrics['mAP50']:.6f}\t{metrics['mAP']:.6f}\t"
                    f"{metrics['AP_small']:.6f}\t{metrics['model_size_mb']:.3f}\t"
                    f"{metrics['cpu_latency_ms']:.1f}\t{metrics['score']:.6f}\n"
                )

        if seed_maps:
            mean_map = statistics.mean(seed_maps)
            std_map = statistics.stdev(seed_maps) if len(seed_maps) > 1 else 0.0
            summary.append({"rank": rank, "commit": commit,
                             "mean_mAP50": mean_map, "std_mAP50": std_map})
            print(f"  → mean mAP50 = {mean_map:.4f} ± {std_map:.4f}\n")

    # Final ranked summary
    print("\n" + "=" * 60)
    print("Phase 1.5 reranking summary (sorted by mean mAP50)")
    print("=" * 60)
    summary.sort(key=lambda r: r["mean_mAP50"], reverse=True)
    for i, r in enumerate(summary, 1):
        print(f"  {i}. commit={r['commit'][:12]}  "
              f"mean_mAP50={r['mean_mAP50']:.4f} ± {r['std_mAP50']:.4f}")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
