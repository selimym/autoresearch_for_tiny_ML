"""
evaluate_core.py — Shared evaluation logic for all phase wrappers.

Fixed file: never modified by agents.

Provides:
  evaluate_model(model_or_path) -> dict[str, float]
    Returns: {score, mAP50, model_size_mb, cpu_latency_ms, params_M}
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

MLFLOW_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:////.mlruns/mlruns.db"
)

# Spatial resolution used for mAP evaluation (must match initial_compress.py)
IMG_SIZE: int = 320


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_map50(onnx_path: str) -> float:
    """Decode FCOS ONNX outputs and compute mAP50 on COCO val.

    ONNX output format (9 tensors from TinyDetector):
        PyTorch exports (cls_list, reg_list, ctr_list) as a flat sequence.
        The exact order depends on the return structure of forward():
          - Grouped by type: cls0, cls1, cls2, reg0, reg1, reg2, ctr0, ctr1, ctr2
        Output shapes (for 320×320 input):
          cls_i: (1, 1, H_i, W_i)  — raw logits
          reg_i: (1, 4, H_i, W_i)  — positive distances (already exp'd in model)
          ctr_i: (1, 1, H_i, W_i)  — centerness logits

    This function auto-detects layout by inspecting output channel counts at
    runtime: channel==1 → cls or ctr, channel==4 → reg.
    """
    import numpy as np
    import onnxruntime as ort
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from torchvision.ops import nms
    from train_utils import make_dataloader, CACHE_DIR

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    all_inputs = sess.get_inputs()
    input_name = all_inputs[0].name if all_inputs else "images"
    strides = [8, 16, 32]  # P3, P4, P5 strides for 320×320 input
    num_levels = len(strides)

    ann_path = CACHE_DIR / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        print(f"[evaluate_core] Val annotations not found at {ann_path}. Returning 0.0")
        return 0.0

    coco_gt = COCO(str(ann_path))
    val_loader = make_dataloader("val", batch_size=1, img_size=IMG_SIZE)

    # Determine ONNX output layout by running once on a dummy input.
    # Layout A (interleaved): cls0, reg0, ctr0, cls1, reg1, ctr1, cls2, reg2, ctr2
    # Layout B (grouped):     cls0, cls1, cls2, reg0, reg1, reg2, ctr0, ctr1, ctr2
    dummy_input = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    probe = sess.run(None, {input_name: dummy_input})
    if len(probe) == 9:
        # Detect by channel dim: reg outputs have 4 channels, cls/ctr have 1
        ch = [p.shape[1] for p in probe]
        if ch[1] == 4:
            # Layout A: cls, reg, ctr interleaved per level
            layout = "interleaved"
        else:
            # Layout B: all cls, then all reg, then all ctr
            layout = "grouped"
    else:
        layout = "interleaved"  # fallback

    def _decode_level(outputs, level_idx, layout):
        """Return (cls_out, reg_out, ctr_out) for the given level index."""
        if layout == "interleaved":
            base = level_idx * 3
            return outputs[base], outputs[base + 1], outputs[base + 2]
        else:
            # grouped: cls0..cls(n-1), reg0..reg(n-1), ctr0..ctr(n-1)
            cls_out = outputs[level_idx]
            reg_out = outputs[num_levels + level_idx]
            ctr_out = outputs[num_levels * 2 + level_idx]
            return cls_out, reg_out, ctr_out

    results = []
    for imgs, targets in val_loader:
        img_np = imgs[0].numpy()[None]  # (1, 3, H, W)
        img_id = int(targets[0]["image_id"].item())

        outputs = sess.run(None, {input_name: img_np})

        all_boxes, all_scores = [], []
        for level_idx, stride in enumerate(strides):
            if level_idx * 3 >= len(outputs) and level_idx >= len(outputs):
                break
            cls_out, reg_out, ctr_out = _decode_level(outputs, level_idx, layout)

            H, W = cls_out.shape[2], cls_out.shape[3]
            cls_sigmoid = 1 / (1 + np.exp(-cls_out[0, 0]))  # sigmoid, shape (H, W)
            ctr_sigmoid = 1 / (1 + np.exp(-ctr_out[0, 0]))  # sigmoid, shape (H, W)
            scores = np.sqrt(np.clip(cls_sigmoid * ctr_sigmoid, 0, 1))

            for r in range(H):
                for c in range(W):
                    score = float(scores[r, c])
                    if score < 0.05:  # confidence threshold
                        continue
                    cx = (c + 0.5) * stride
                    cy = (r + 0.5) * stride
                    l, t, r_dist, b = reg_out[0, :, r, c]
                    x1 = cx - l
                    y1 = cy - t
                    x2 = cx + r_dist
                    y2 = cy + b
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)

        if all_boxes:
            import torch
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
            scores_t = torch.tensor(all_scores, dtype=torch.float32)
            keep = nms(boxes_t, scores_t, iou_threshold=0.5)
            for idx in keep.tolist():
                x1, y1, x2, y2 = all_boxes[idx]
                results.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": all_scores[idx],
                })

    if not results:
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[1])  # AP @[.50]


def _measure_cpu_latency_ms(onnx_path: str) -> float:
    """Measure median CPU inference latency over 50 runs (10 warmup).

    Args:
        onnx_path: Path to an ONNX model file.

    Returns:
        Median latency in milliseconds.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # Build a dummy input matching the model's expected shape.
    input_meta = session.get_inputs()[0]
    # Shape may contain dynamic axes (None or symbolic strings); replace with 1.
    shape = [
        d if isinstance(d, int) and d > 0 else 1
        for d in input_meta.shape
    ]
    import numpy as np
    dummy_input = np.random.rand(*shape).astype(np.float32)
    feed = {input_meta.name: dummy_input}

    # Warmup
    for _ in range(10):
        session.run(None, feed)

    # Timed runs
    latencies: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return statistics.median(latencies)


def _model_size_mb(onnx_path: str) -> float:
    """Return file size of the ONNX model in megabytes."""
    return os.path.getsize(onnx_path) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_model(model_or_path: Union[nn.Module, str]) -> dict[str, float]:
    """Evaluate a model and return a metrics dictionary.

    Args:
        model_or_path:
            - ``str`` or ``Path``: path to an ``.onnx`` file.
            - ``nn.Module``: a PyTorch model; it will be exported to a
              temporary ONNX file automatically.

    Returns:
        A ``dict`` with the following keys:

        - ``score``          – composite score (mAP50 / model_size_mb if
                               mAP50 >= 0.15, else 0.0)
        - ``mAP50``          – AP at IoU=0.50 on COCO val (person class)
        - ``model_size_mb``  – size of the ONNX file in MB
        - ``cpu_latency_ms`` – median CPU inference latency in ms
        - ``params_M``       – parameter count in millions (0.0 if a path
                               was provided instead of an ``nn.Module``)
    """
    # ------------------------------------------------------------------ #
    # Resolve to an ONNX path                                             #
    # ------------------------------------------------------------------ #
    params_M: float = 0.0
    _tmp_dir = None  # keep reference so it isn't garbage-collected early

    if isinstance(model_or_path, (str, Path)):
        onnx_path = str(model_or_path)
    elif isinstance(model_or_path, nn.Module):
        from train_utils import export_onnx

        params_M = sum(p.numel() for p in model_or_path.parameters()) / 1e6

        _tmp_dir = tempfile.TemporaryDirectory()
        onnx_path = os.path.join(_tmp_dir.name, "model_eval.onnx")
        export_onnx(model_or_path, onnx_path)
    else:
        raise TypeError(
            f"evaluate_model expects nn.Module, str, or Path; got {type(model_or_path)}"
        )

    # ------------------------------------------------------------------ #
    # Compute metrics                                                     #
    # ------------------------------------------------------------------ #
    try:
        mAP50 = _compute_map50(onnx_path)
        size_mb = _model_size_mb(onnx_path)
        latency_ms = _measure_cpu_latency_ms(onnx_path)

        score = mAP50 / size_mb if mAP50 >= 0.15 and size_mb > 0 else 0.0

        metrics: dict[str, float] = {
            "score": score,
            "mAP50": mAP50,
            "model_size_mb": size_mb,
            "cpu_latency_ms": latency_ms,
            "params_M": params_M,
        }
    finally:
        # Clean up temp dir if we created one
        if _tmp_dir is not None:
            _tmp_dir.cleanup()

    # ------------------------------------------------------------------ #
    # MLflow logging (best-effort)                                        #
    # ------------------------------------------------------------------ #
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(nested=True):
            mlflow.log_metrics(metrics)
    except Exception:
        pass  # Never let MLflow errors propagate

    return metrics
