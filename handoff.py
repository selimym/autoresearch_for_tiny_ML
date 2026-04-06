"""
handoff.py — Phase transition utility. Human-run between phases.

Usage:
  python handoff.py --phase 1              # Phase 1→2 transition (default: highest score)
  python handoff.py --phase 1 --select abc1234   # Use specific commit
  python handoff.py --phase 2              # Phase 2→3 transition
"""
import argparse, json, subprocess, shutil
from pathlib import Path


RESULTS_FILES = {1: "results_phase1.tsv", 2: "results_phase2.tsv"}
HANDOFF_PATH = Path("handoff.json")


def run(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def read_results(phase: int) -> list[dict]:
    results_path = Path(RESULTS_FILES[phase])
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found. Has Phase {phase} run?")

    rows = []
    with open(results_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"{results_path} has no data rows.")

    header = lines[0].strip().split("\t")
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        if row.get("status") == "keep":
            try:
                row["score"] = float(row["score"])
                row["mAP50"] = float(row["mAP50"])
                row["model_size_mb"] = float(row["model_size_mb"])
                row["cpu_latency_ms"] = float(row["cpu_latency_ms"])
                rows.append(row)
            except (ValueError, KeyError):
                pass
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    """Return Pareto-optimal rows maximizing mAP50 and minimizing model_size_mb."""
    front = []
    for r in rows:
        dominated = False
        for other in rows:
            if other is r:
                continue
            if (other["mAP50"] >= r["mAP50"] and other["model_size_mb"] <= r["model_size_mb"]
                    and (other["mAP50"] > r["mAP50"] or other["model_size_mb"] < r["model_size_mb"])):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda x: x["score"], reverse=True)


def handoff_phase1(select_commit: str | None) -> None:
    rows = read_results(1)
    front = pareto_front(rows)

    print("\n=== Phase 1 Top-5 Pareto Candidates ===")
    print(f"{'#':>3}  {'commit':>10}  {'score':>8}  {'mAP50':>8}  {'size_mb':>8}  {'latency_ms':>12}  description")
    print("-" * 85)
    for i, row in enumerate(front[:5]):
        print(f"{i+1:>3}  {row['commit']:>10}  {row['score']:>8.4f}  "
              f"{row['mAP50']:>8.4f}  {row['model_size_mb']:>8.2f}  "
              f"{row['cpu_latency_ms']:>12.1f}  {row.get('description','')}")

    selected = select_commit or front[0]["commit"]
    print(f"\nSelected commit: {selected}")

    run(f"git checkout {selected}")
    src = Path("checkpoints/phase1_candidate.pt")
    dst = Path("checkpoints/phase1_best.pt")
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied {src} → {dst}")

    data = {}
    if HANDOFF_PATH.exists():
        with open(HANDOFF_PATH) as f:
            data = json.load(f)
    data["phase1_best"] = str(dst)
    with open(HANDOFF_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated handoff.json: {data}")

    tag = selected[:8]
    branch = f"phase2/quant-{tag}"
    run(f"git checkout -b {branch}")
    print(f"Created branch: {branch}")

    Path("results_phase2.tsv").unlink(missing_ok=True)
    print("Ready for Phase 2.")


def handoff_phase2(select_commit: str | None) -> None:
    rows = read_results(2)
    front = pareto_front(rows)

    print("\n=== Phase 2 Top-5 Pareto Candidates ===")
    print(f"{'#':>3}  {'commit':>10}  {'score':>8}  {'mAP50':>8}  {'size_mb':>8}  {'latency_ms':>12}  description")
    print("-" * 85)
    for i, row in enumerate(front[:5]):
        print(f"{i+1:>3}  {row['commit']:>10}  {row['score']:>8.4f}  "
              f"{row['mAP50']:>8.4f}  {row['model_size_mb']:>8.2f}  "
              f"{row['cpu_latency_ms']:>12.1f}  {row.get('description','')}")

    selected = select_commit or front[0]["commit"]
    print(f"\nSelected commit: {selected}")

    run(f"git checkout {selected}")

    pt_src = Path("checkpoints/phase2_candidate.pt")
    onnx_src = next(Path("checkpoints").glob("phase2_*.onnx"), None)

    data = {}
    if HANDOFF_PATH.exists():
        with open(HANDOFF_PATH) as f:
            data = json.load(f)

    if pt_src.exists():
        dst_pt = Path("checkpoints/phase2_best.pt")
        shutil.copy(pt_src, dst_pt)
        data["phase2_best_pt"] = str(dst_pt)

    if onnx_src:
        dst_onnx = Path("checkpoints/phase2_best.onnx")
        shutil.copy(onnx_src, dst_onnx)
        data["phase2_best_onnx"] = str(dst_onnx)

    with open(HANDOFF_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated handoff.json: {data}")

    tag = selected[:8]
    branch = f"phase3/prune-{tag}"
    run(f"git checkout -b {branch}")
    print(f"Created branch: {branch}")

    Path("results_phase3.tsv").unlink(missing_ok=True)
    print("Ready for Phase 3.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--select", type=str, default=None,
                        help="Override default (highest score) with specific commit hash")
    args = parser.parse_args()

    if args.phase == 1:
        handoff_phase1(args.select)
    elif args.phase == 2:
        handoff_phase2(args.select)


if __name__ == "__main__":
    main()
