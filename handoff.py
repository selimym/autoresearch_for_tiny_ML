"""
handoff.py — Phase transition utility. Human-run between phases.

Usage:
  python handoff.py --phase 1              # Phase 1→2 transition (default: highest score)
  python handoff.py --phase 1 --select abc1234   # Use specific commit
  python handoff.py --phase 2              # Phase 2→3 transition
  python handoff.py --phase 2 --baseline yolov8n_person.onnx  # with baseline comparison
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


def _eval_baseline_onnx(onnx_path: str) -> dict:
    """Evaluate a standard post-NMS detection ONNX against COCO val (person class).

    Expected output format: one or more tensors where the first tensor has shape
    (1, N, 6) or (N, 6) with columns [x1, y1, x2, y2, score, class_id] in
    pixel coordinates for the input resolution.  This is the format produced by
    most export pipelines (Ultralytics, NanoDet-plus, etc.) after NMS.

    Coordinates must be at the same scale as the model input (e.g. 320×320).
    """
    import gc
    import numpy as np
    import onnxruntime as ort
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from torchvision.ops import nms
    import torch

    from evaluate_core import _measure_cpu_latency_ms, _model_size_mb, IMG_SIZE
    from train_utils import make_dataloader, CACHE_DIR

    ann_path = CACHE_DIR / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Val annotations not found at {ann_path}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    # Resolve dynamic axes to concrete values
    input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
    img_size = input_shape[2] if len(input_shape) >= 3 else IMG_SIZE

    coco_gt = COCO(str(ann_path))
    val_loader = make_dataloader("val", batch_size=1, img_size=img_size)
    results = []

    for imgs, targets in val_loader:
        img_np = imgs[0].numpy()[None].astype(np.float32)  # (1, 3, H, W)
        img_id = int(targets[0]["image_id"].item())
        outputs = sess.run(None, {input_name: img_np})

        # Accept (1, N, 6), (N, 6), or (1, 6, N) — normalise to (N, 6)
        det = outputs[0]
        if det.ndim == 3:
            if det.shape[-1] == 6:
                det = det[0]                    # (1, N, 6) → (N, 6)
            elif det.shape[1] == 6:
                det = det[0].T                  # (1, 6, N) → (N, 6)
        if det.ndim != 2 or det.shape[1] < 6 or det.shape[0] == 0:
            continue

        boxes = torch.tensor(det[:, :4], dtype=torch.float32)
        scores = torch.tensor(det[:, 4], dtype=torch.float32)
        class_ids = det[:, 5].astype(int)

        # Keep only person class (COCO id 1 or zero-indexed 0)
        person_mask = (class_ids == 0) | (class_ids == 1)
        if not person_mask.any():
            continue
        boxes = boxes[person_mask]
        scores = scores[person_mask]

        keep = nms(boxes, scores, iou_threshold=0.5)
        for idx in keep.tolist():
            x1, y1, x2, y2 = boxes[idx].tolist()
            results.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[idx]),
            })

    size_mb = _model_size_mb(onnx_path)
    latency_ms = _measure_cpu_latency_ms(onnx_path)

    if not results:
        del coco_gt, val_loader
        gc.collect()
        return {"mAP50": 0.0, "mAP": 0.0, "AP_small": 0.0,
                "model_size_mb": size_mb, "cpu_latency_ms": latency_ms}

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = [1]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    metrics = {
        "mAP50":          float(ev.stats[1]),
        "mAP":            float(ev.stats[0]),
        "AP_small":       float(ev.stats[3]),
        "model_size_mb":  size_mb,
        "cpu_latency_ms": latency_ms,
    }
    del coco_gt, coco_dt, ev, val_loader
    gc.collect()
    return metrics


def _print_comparison(our_metrics: dict, baseline_metrics: dict,
                       our_label: str, baseline_label: str) -> None:
    """Print a side-by-side comparison table."""
    cols = ["mAP50", "mAP", "AP_small", "model_size_mb", "cpu_latency_ms"]
    fmt =  [".4f",   ".4f", ".4f",      ".2f",           ".1f"]
    w = max(len(our_label), len(baseline_label), 12)
    header = f"\n{'Metric':<20}  {our_label:>{w}}  {baseline_label:>{w}}"
    print(header)
    print("-" * len(header))
    for col, f in zip(cols, fmt):
        ours = our_metrics.get(col, float("nan"))
        theirs = baseline_metrics.get(col, float("nan"))
        print(f"  {col:<18}  {ours:>{w}{f}}  {theirs:>{w}{f}}")
    print()


def handoff_phase2(select_commit: str | None, baseline_onnx: str | None = None) -> None:
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

    # Baseline comparison (optional) — evaluate our model vs reference side-by-side
    if baseline_onnx:
        print(f"\nRunning baseline comparison against {baseline_onnx} ...")
        from evaluate_core import evaluate_model
        our_onnx = str(onnx_src or dst_onnx if onnx_src else None)
        if our_onnx:
            our_metrics = evaluate_model(our_onnx)
            baseline_metrics = _eval_baseline_onnx(baseline_onnx)
            our_label = f"ours ({selected[:8]})"
            baseline_label = Path(baseline_onnx).stem
            _print_comparison(our_metrics, baseline_metrics, our_label, baseline_label)
            data["baseline_comparison"] = {
                "baseline_path": baseline_onnx,
                "our_metrics": our_metrics,
                "baseline_metrics": baseline_metrics,
            }
        else:
            print("WARNING: no Phase 2 ONNX found, skipping baseline comparison.")

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
    parser.add_argument("--baseline", type=str, default=None,
                        help="(Phase 2 only) Path to a reference ONNX for side-by-side "
                             "comparison. Expected output: (1,N,6) or (N,6) post-NMS "
                             "detections [x1,y1,x2,y2,score,class_id].")
    args = parser.parse_args()

    if args.phase == 1:
        handoff_phase1(args.select)
    elif args.phase == 2:
        handoff_phase2(args.select, baseline_onnx=args.baseline)


if __name__ == "__main__":
    main()
