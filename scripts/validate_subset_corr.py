"""
validate_subset_corr.py — Validate that Phase 1's cheap 5K-subset screening
preserves the correct architecture rank order compared to 10K and 20K subsets.

Retrains top-K Phase 1 candidates at multiple subset sizes and computes
Spearman rank correlation between the 5K ranking and each larger subset ranking.

Usage:
    uv run scripts/validate_subset_corr.py \
      [--tsv results/phase1.tsv] \
      [--results-dir results/phase1] \
      [--top-k 5] \
      [--subsets 5000 10000 20000] \
      [--epochs 6] \
      [--frozen-stages 4] \
      [--batch-size 8] \
      [--lr 1e-3] \
      [--seeds 1] \
      [--out results/subset_corr.tsv]

Reads:  results/phase1.tsv  (or --tsv path)
Writes: results/subset_corr.tsv  (or --out path)
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
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
    for candidate in ["main.py", "initial_compress.py"]:
        prog = gen_dir / candidate
        if prog.exists():
            spec = importlib.util.spec_from_file_location("_phase1_program", prog)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise FileNotFoundError(f"No program file found in {gen_dir}")


def _find_candidate_dir(results_dir: Path, commit_hash: str) -> Path | None:
    for gen_dir in sorted(results_dir.glob("gen_*")):
        for prog in gen_dir.glob("*.py"):
            if commit_hash[:8] in prog.stem or commit_hash[:8] in gen_dir.name:
                return gen_dir
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
    subset_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    from tinydet.repro import set_seed
    set_seed(seed)

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
    train_loader = make_dataloader("train", batch_size, img_size, subset_size=subset_size)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * len(train_loader),
        pct_start=0.1,
    )

    model.train()
    for epoch in range(epochs):
        for imgs, targets in tqdm(
            train_loader,
            desc=f"  epoch {epoch+1}/{epochs} (subset={subset_size})",
            leave=False,
        ):
            imgs_t = torch.stack(imgs).to(device)
            targets = [
                {k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()}
                for t in targets
            ]
            optimizer.zero_grad()
            cls_preds, reg_preds, ctr_preds = model(imgs_t)
            loss = mod.fcos_loss(cls_preds, reg_preds, ctr_preds, targets, strides)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

    del train_loader, optimizer, scheduler
    gc.collect()

    model.eval()
    onnx_path = f"checkpoints/subset_corr_s{subset_size}_seed{seed}.onnx"
    export_onnx(model, onnx_path, img_size)
    metrics = evaluate_model(onnx_path)

    del model, backbone
    gc.collect()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 5K subset rank correlation vs larger subsets"
    )
    parser.add_argument("--tsv", default="results/phase1.tsv",
                        help="Path to Phase 1 results TSV (default: results/phase1.tsv)")
    parser.add_argument("--results-dir", default="results/phase1",
                        help="Phase 1 ShinkaEvolve results directory")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top Phase 1 candidates to evaluate (default: 5)")
    parser.add_argument("--subsets", type=int, nargs="+", default=[5000, 10000, 20000],
                        help="Subset sizes to evaluate (default: 5000 10000 20000)")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Training epochs per run (default: 6)")
    parser.add_argument("--frozen-stages", type=int, default=4,
                        help="Backbone frozen stages — matches Phase 1 (default: 4)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Number of random seeds per (candidate, subset) pair (default: 2)")
    parser.add_argument("--out", default="results/subset_corr.tsv",
                        help="Output TSV path (default: results/subset_corr.tsv)")
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

    def _score(r: dict) -> float:
        try:
            return float(r.get("score", 0) or 0)
        except ValueError:
            return 0.0

    candidates = sorted(rows, key=_score, reverse=True)[: args.top_k]
    print(
        f"Validating subset rank correlation for top-{len(candidates)} Phase 1 candidates\n"
        f"Subsets: {args.subsets}  seeds: {args.seeds}  epochs: {args.epochs}\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("commit\tsubset_size\tseed\tmAP50\trank\n")

    # results[subset_size][commit] = list of mAP50 values across seeds
    results: dict[int, dict[str, list[float]]] = {s: {} for s in args.subsets}

    total_runs = len(candidates) * len(args.subsets) * args.seeds
    run_idx = 0

    for cand in candidates:
        commit = cand.get("commit", "unknown")
        phase1_score = cand.get("score", "?")

        cand_dir = _find_candidate_dir(results_dir, commit)
        if cand_dir is None:
            print(f"  ERROR: could not resolve artifact dir for commit {commit[:8]} — "
                  f"skipping. Check that results_dir={results_dir} contains the "
                  f"candidate's evolved program files.")
            continue
        try:
            mod = _load_program_from_results(cand_dir)
        except FileNotFoundError as e:
            print(f"  SKIP {commit[:12]}: {e}")
            continue

        for subset_size in args.subsets:
            results[subset_size].setdefault(commit, [])
            for seed in range(1, args.seeds + 1):
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] commit={commit[:12]}  "
                    f"subset={subset_size}  seed={seed}  "
                    f"phase1_score={phase1_score}"
                )
                t0 = time.perf_counter()
                try:
                    metrics = _train_candidate(
                        mod=mod,
                        epochs=args.epochs,
                        frozen_stages=args.frozen_stages,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        subset_size=subset_size,
                        seed=seed,
                        device=device,
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    gc.collect()
                    continue
                elapsed = time.perf_counter() - t0
                map50 = metrics["mAP50"]
                results[subset_size][commit].append(map50)
                print(
                    f"  mAP50={map50:.4f}  mAP={metrics['mAP']:.4f}  "
                    f"score={metrics['score']:.4f}  ({elapsed:.0f}s)"
                )
                gc.collect()

    # Build mean mAP50 per (subset_size, commit), assign ranks, write TSV
    # rank_table[subset_size] = list of (commit, mean_mAP50) sorted desc
    rank_table: dict[int, list[tuple[str, float]]] = {}
    for subset_size in args.subsets:
        entries = []
        for commit, maps in results[subset_size].items():
            if maps:
                mean_map = sum(maps) / len(maps)
                entries.append((commit, mean_map))
        entries.sort(key=lambda x: x[1], reverse=True)
        rank_table[subset_size] = entries

    # Write TSV with ranks
    with open(out_path, "a") as f:
        for subset_size in args.subsets:
            for rank, (commit, mean_map) in enumerate(rank_table[subset_size], 1):
                seed_maps = results[subset_size][commit]
                for seed_idx, map50 in enumerate(seed_maps, 1):
                    f.write(f"{commit}\t{subset_size}\t{seed_idx}\t{map50:.6f}\t{rank}\n")

    # Compute Spearman correlation between 5K ranks and each larger subset
    baseline_size = args.subsets[0]
    baseline_entries = rank_table.get(baseline_size, [])
    baseline_commits = [c for c, _ in baseline_entries]

    print("\n" + "=" * 60)
    print("Subset correlation with 5K baseline")
    print("=" * 60)

    all_trustworthy = True
    all_uncertain = True
    conclusions: list[str] = []

    for subset_size in args.subsets[1:]:
        other_entries = rank_table.get(subset_size, [])
        other_commits = [c for c, _ in other_entries]

        # Build aligned rank vectors for commits present in both
        common_commits = [c for c in baseline_commits if c in other_commits]
        if len(common_commits) < 2:
            print(f"{subset_size // 1000}K: insufficient data (need ≥2 common commits)")
            continue

        baseline_ranks = {c: i + 1 for i, (c, _) in enumerate(baseline_entries)}
        other_ranks = {c: i + 1 for i, (c, _) in enumerate(other_entries)}

        x = [baseline_ranks[c] for c in common_commits]
        y = [other_ranks[c] for c in common_commits]

        rho, pval = spearmanr(x, y)

        label = f"{subset_size // 1000}K"
        print(f"{label}: ρ={rho:.2f}  p={pval:.3f}")

        if rho < 0.8:
            all_trustworthy = False
        if rho < 0.6:
            all_uncertain = False

        conclusions.append((rho, pval, label))

    if conclusions:
        min_rho = min(r for r, _, _ in conclusions)
        if min_rho >= 0.8:
            verdict = "trustworthy"
        elif min_rho >= 0.6:
            verdict = "uncertain"
        else:
            verdict = "unreliable"
        print(f"Conclusion: Phase 1 5K ranking is {verdict}")
    else:
        print("Conclusion: insufficient data to determine trustworthiness")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
