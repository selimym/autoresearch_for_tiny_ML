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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_map50(onnx_path: str) -> float:
    """Compute mAP50 on COCO 2017 val set (person class only).

    TODO: This is a stub returning 0.0 until the ONNX output format is
    finalised in initial_compress.py (Task 8).  The full implementation
    will:
      - Run the ONNX model on every COCO 2017 val image using
        onnxruntime CPUExecutionProvider.
      - Decode FPN-level outputs: cls (1, 1, H, W) and reg (1, 4, H, W)
        per stride level.
      - Apply sigmoid to cls logits; apply NMS with iou_threshold=0.5
        via torchvision.ops.nms.
      - Accumulate COCO-format detections and evaluate with
        pycocotools.cocoeval.COCOeval (catIds=[1]).
      - Return evaluator.stats[1] (AP @ IoU=0.50).

    Once Task 8 defines the exact ONNX output format, replace the
    ``return 0.0`` line below with the real decoder + evaluator.
    """
    # Stub: real mAP computation pending model output format from Task 8.
    return 0.0


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
