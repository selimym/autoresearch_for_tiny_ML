"""
compress.py — Phase 2-3 agent-editable file. Architecture locked from Phase 1.

Usage:
  uv run python compress.py > run.log 2>&1

Agent greps:
  grep "^score:\\|^mAP50:\\|^model_size_mb:\\|^cpu_latency_ms:" run.log
"""
import json
import gc
import os
from pathlib import Path

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# =============================================================================
# PHASE 2 — Quantization (backbone + neck architecture locked from Phase 1)
# Agent varies: quant mode, calibration strategy, per-channel vs per-tensor
# =============================================================================

QUANT_MODE    = "none"    # "none" | "ptq_int8_static"
CALIB_BATCHES = 16        # kept for compatibility; calibration uses 50 val images

# =============================================================================
# PHASE 3 — Pruning (loads Phase 2 best from handoff.json)
# Agent varies: prune type, ratio, recovery fine-tuning settings
# =============================================================================

PRUNE_RATIO   = 0.0              # fraction of channels pruned (0.0 = disabled)
PRUNE_TYPE    = "l1_structured"  # "l1_structured" | "magnitude_unstructured"

# =============================================================================
# Training recipe — agent can vary in any phase
# =============================================================================

LR             = 1e-3
EPOCHS         = 4
BATCH_SIZE     = int(os.environ.get("TRAINING_BATCH_SIZE", "8"))
WARMUP_EPOCHS  = 0.5
FROZEN_STAGES  = 2       # 2 = last 2 backbone stages unfrozen (Phases 2-3 default)
IMG_SIZE       = 320     # fixed — do not change


# ---------------------------------------------------------------------------
# Handoff loader
# ---------------------------------------------------------------------------

def load_phase_checkpoint() -> Path:
    """Read handoff.json and return the checkpoint path for this phase.

    Tries "phase2_best_pt" key first (Phase 3 entry point), then falls back
    to "phase1_best" (Phase 2 entry point).

    Raises:
        AssertionError: if handoff.json is missing or neither key is present.
    """
    handoff_path = Path("handoff.json")
    assert handoff_path.exists(), (
        "handoff.json not found. "
        "Run handoff.py --phase 1 (after Phase 1) or --phase 2 (after Phase 2) "
        "to generate it before running compress.py."
    )
    with open(handoff_path) as f:
        handoff = json.load(f)

    if "phase2_best_pt" in handoff:
        ckpt_path = Path(handoff["phase2_best_pt"])
    elif "phase1_best" in handoff:
        ckpt_path = Path(handoff["phase1_best"])
    else:
        raise AssertionError(
            "handoff.json exists but contains neither 'phase2_best_pt' nor "
            "'phase1_best' keys. Re-run handoff.py to regenerate it."
        )

    assert ckpt_path.exists(), (
        f"Checkpoint referenced in handoff.json does not exist: {ckpt_path}"
    )
    return ckpt_path


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def apply_pruning(model, prune_type: str, prune_ratio: float):
    """Apply structured or unstructured pruning to neck + head parameters.

    Args:
        model:       TinyDetector instance.
        prune_type:  "l1_structured" | "magnitude_unstructured"
        prune_ratio: fraction of weights/channels to prune (0.0 = no-op).

    Returns:
        model with pruning masks applied (in-place).
    """
    import torch
    import torch.nn.utils.prune as prune

    if prune_ratio <= 0.0:
        return model

    # Gather Conv2d layers from neck and head only (backbone stays intact)
    modules_to_prune = []
    for name, module in model.neck.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            modules_to_prune.append((module, "weight"))
    for name, module in model.head.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            modules_to_prune.append((module, "weight"))

    if not modules_to_prune:
        print("apply_pruning: no Conv2d modules found in neck/head — skipping.")
        return model

    if prune_type == "l1_structured":
        for module, param in modules_to_prune:
            # Prune output channels (dim=0) by L1 norm
            n_channels = module.weight.shape[0]
            amount = max(1, int(n_channels * prune_ratio))
            amount = min(amount, n_channels - 1)  # keep at least 1 channel
            prune.ln_structured(module, name=param, amount=amount, n=1, dim=0)
    elif prune_type == "magnitude_unstructured":
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_ratio,
        )
    else:
        raise ValueError(f"Unknown prune_type: {prune_type!r}")

    return model


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class _ValCalibrationDataReader(CalibrationDataReader):
    """Feeds ~50 val images to onnxruntime quantize_static as calibration data."""

    def __init__(self, input_name: str, img_size: int):
        from tinydet.data import make_dataloader
        import numpy as np

        calib_loader = make_dataloader("val", batch_size=1, img_size=img_size)
        self._data = []
        for imgs, _ in calib_loader:
            if len(self._data) >= 50:
                break
            img_np = imgs[0].numpy()[None].astype(np.float32)  # (1, 3, H, W)
            self._data.append({input_name: img_np})
        self._iter = iter(self._data)

    def get_next(self):
        return next(self._iter, None)


def quantize_onnx_static(float_onnx_path: str, quant_onnx_path: str) -> str:
    """Quantize a float ONNX model to INT8 using static PTQ with calibration data.

    Args:
        float_onnx_path: path to the float32 ONNX model produced by export_onnx.
        quant_onnx_path: output path for the quantized INT8 ONNX model.

    Returns:
        quant_onnx_path (the path to the quantized model).
    """
    import onnxruntime as ort

    # Determine the model's input name for the calibration reader.
    sess = ort.InferenceSession(float_onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    calibration_reader = _ValCalibrationDataReader(input_name, IMG_SIZE)

    quantize_static(
        float_onnx_path,
        quant_onnx_path,
        calibration_reader,
        weight_type=QuantType.QInt8,
    )
    return quant_onnx_path


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    """Load phase checkpoint, prune → fine-tune → quantize → evaluate."""
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import OneCycleLR
    from train_utils import load_backbone, make_dataloader, export_onnx, save_checkpoint
    from evaluate_core import evaluate_model
    from initial_compress import (
        build_model,
        NECK_CHANNELS,
        UIB_CONFIGS,
        HEAD_CHANNELS,
        HEAD_STACKS,
        fcos_loss,
    )

    # --- Load checkpoint from handoff ---
    ckpt_path = load_phase_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, backbone_channels = load_backbone(frozen_stages=FROZEN_STAGES)

    FPNNeck, FCOSHead, TinyDetector = build_model(backbone_channels)
    neck = FPNNeck(backbone_channels, NECK_CHANNELS, UIB_CONFIGS)
    head = FCOSHead(NECK_CHANNELS, HEAD_CHANNELS, HEAD_STACKS)
    model = TinyDetector(backbone, neck, head).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded OK.")

    # --- Phase 3: apply pruning before fine-tuning ---
    if PRUNE_RATIO > 0.0:
        print(f"Applying pruning: type={PRUNE_TYPE}, ratio={PRUNE_RATIO}")
        model = apply_pruning(model, PRUNE_TYPE, PRUNE_RATIO)

    # --- Fine-tuning (recovery after pruning, or standard Phase 2 warm-up) ---
    strides = [8, 16, 32]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    train_loader = make_dataloader("train", BATCH_SIZE, IMG_SIZE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=EPOCHS * len(train_loader),
        pct_start=WARMUP_EPOCHS / max(EPOCHS, 1),
    )

    model.train()
    for epoch in range(EPOCHS):
        for imgs, targets in train_loader:
            imgs_t = torch.stack(imgs).to(device)
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            cls_preds, reg_preds, ctr_preds = model(imgs_t)
            loss = fcos_loss(cls_preds, reg_preds, ctr_preds, tgts, strides)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} done, loss={loss.item():.4f}")

    # Free training objects before quantization — calibration data reader loads
    # the val annotation JSON, and train_loader holds the train JSON.
    # Having both alive simultaneously doubles the annotation RAM footprint.
    del train_loader, optimizer, scheduler
    gc.collect()

    # --- Export and evaluate ---
    float_onnx_path = "checkpoints/compress_candidate_float.onnx"
    onnx_path = "checkpoints/compress_candidate.onnx"

    model.eval()
    # If ONNX export or quantization fails, mark the run INVALID and abort.
    # Never silently fall back to evaluating an unquantized model — that
    # would produce misleading metrics for a "quantized" experiment.
    try:
        export_onnx(model, float_onnx_path, IMG_SIZE)
    except Exception as e:
        print(f"ERROR: ONNX export failed for QUANT_MODE={QUANT_MODE}: {e}")
        print("status:INVALID")
        raise SystemExit(1) from e

    # --- Phase 2: apply static PTQ quantization after export ---
    if QUANT_MODE == "ptq_int8_static":
        print(f"Applying quantization: mode={QUANT_MODE}")
        try:
            quantize_onnx_static(float_onnx_path, onnx_path)
        except Exception as e:
            print(f"ERROR: ONNX static quantization failed for QUANT_MODE={QUANT_MODE}: {e}")
            print("status:INVALID")
            raise SystemExit(1) from e
    elif QUANT_MODE == "none":
        import shutil
        shutil.copy2(float_onnx_path, onnx_path)
    else:
        raise ValueError(f"Unknown QUANT_MODE: {QUANT_MODE!r}")

    metrics = evaluate_model(onnx_path)

    save_checkpoint(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "config": {
                "QUANT_MODE": QUANT_MODE,
                "CALIB_BATCHES": CALIB_BATCHES,
                "PRUNE_RATIO": PRUNE_RATIO,
                "PRUNE_TYPE": PRUNE_TYPE,
                "NECK_CHANNELS": NECK_CHANNELS,
                "UIB_CONFIGS": UIB_CONFIGS,
                "HEAD_CHANNELS": HEAD_CHANNELS,
                "HEAD_STACKS": HEAD_STACKS,
            },
        },
        "checkpoints/compress_candidate.pt",
    )

    return metrics


if __name__ == "__main__":
    metrics = run_experiment()

    # --- Agent-greppable output block ---
    print("---")
    print(f"score:{metrics['score']:.6f}")
    print(f"mAP50:{metrics['mAP50']:.6f}")
    print(f"mAP:{metrics['mAP']:.6f}")
    print(f"AP_small:{metrics['AP_small']:.6f}")
    print(f"model_size_mb:{metrics['model_size_mb']:.3f}")
    print(f"cpu_latency_ms:{metrics['cpu_latency_ms']:.1f}")
    print("---")
