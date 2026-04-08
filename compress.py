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

# =============================================================================
# PHASE 2 — Quantization (backbone + neck architecture locked from Phase 1)
# Agent varies: quant mode, calibration strategy, per-channel vs per-tensor
# =============================================================================

QUANT_MODE    = "none"    # "none" | "ptq_int8" | "qat_int8" | "ptq_int4" | "dynamic"
CALIB_BATCHES = 16        # calibration batches for PTQ observers

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

def apply_quantization(model, quant_mode: str, calib_batches: int):
    """Apply post-training or quantization-aware quantization to the model.

    Args:
        model:         TinyDetector (float32, already fine-tuned).
        quant_mode:    one of "none" | "ptq_int8" | "qat_int8" | "ptq_int4" | "dynamic"
        calib_batches: number of calibration batches used for PTQ observers.

    Returns:
        Quantized (or unchanged) model in eval mode.
    """
    import torch
    from train_utils import make_dataloader

    if quant_mode == "none":
        return model.eval()

    device = next(model.parameters()).device

    if quant_mode == "dynamic":
        # Dynamic quantization: weights INT8, activations quantized at runtime
        import torch.quantization as tq
        model.eval().cpu()
        quantized = tq.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8,
        )
        return quantized

    if quant_mode in ("ptq_int8", "ptq_int4"):
        import torch.quantization as tq

        model.eval().cpu()

        # Use per-channel quantization for better accuracy
        if quant_mode == "ptq_int8":
            qconfig = tq.get_default_qconfig("fbgemm")
        else:
            # INT4: use reduce-range per-channel observer (approximates 4-bit)
            from torch.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
            qconfig = tq.QConfig(
                activation=MinMaxObserver.with_args(
                    dtype=torch.quint8,
                    reduce_range=True,
                ),
                weight=PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=True,
                ),
            )

        model.qconfig = qconfig
        tq.prepare(model, inplace=True)

        # Calibration pass
        calib_loader = make_dataloader("val", BATCH_SIZE, IMG_SIZE)
        model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(calib_loader):
                if batch_idx >= calib_batches:
                    break
                imgs_t = torch.stack(imgs).cpu()
                model(imgs_t)

        tq.convert(model, inplace=True)
        return model

    if quant_mode == "qat_int8":
        import torch.quantization as tq
        import torch.optim as optim
        from torch.optim.lr_scheduler import OneCycleLR
        from train_utils import make_dataloader

        model.train().cpu()
        qconfig = tq.get_default_qat_qconfig("fbgemm")
        model.qconfig = qconfig
        tq.prepare_qat(model, inplace=True)

        # Short QAT fine-tuning (uses same EPOCHS / LR / BATCH_SIZE settings)
        train_loader = make_dataloader("train", BATCH_SIZE, IMG_SIZE)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=LR * 0.1, weight_decay=1e-4)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR * 0.1,
            total_steps=EPOCHS * len(train_loader),
            pct_start=WARMUP_EPOCHS / max(EPOCHS, 1),
        )

        from initial_compress import fcos_loss
        strides = [8, 16, 32]

        for epoch in range(EPOCHS):
            for imgs, targets in train_loader:
                imgs_t = torch.stack(imgs).cpu()
                tgts = [{k: v.cpu() for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                cls_preds, reg_preds, ctr_preds = model(imgs_t)
                loss = fcos_loss(cls_preds, reg_preds, ctr_preds, tgts, strides)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
            print(f"QAT epoch {epoch + 1}/{EPOCHS} done, loss={loss.item():.4f}")

        tq.convert(model.eval(), inplace=True)
        return model

    raise ValueError(f"Unknown QUANT_MODE: {quant_mode!r}")


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

    # Free training objects before quantization — apply_quantization loads
    # the val annotation JSON, and train_loader holds the train JSON.
    # Having both alive simultaneously doubles the annotation RAM footprint.
    del train_loader, optimizer, scheduler
    gc.collect()

    # --- Phase 2: apply quantization after fine-tuning ---
    if QUANT_MODE != "none":
        print(f"Applying quantization: mode={QUANT_MODE}, calib_batches={CALIB_BATCHES}")
        model = apply_quantization(model, QUANT_MODE, CALIB_BATCHES)
    else:
        model.eval()

    # --- Export and evaluate ---
    onnx_path = "checkpoints/compress_candidate.onnx"
    # Quantized models may not export cleanly; fall back to float eval path
    try:
        export_onnx(model, onnx_path, IMG_SIZE)
    except Exception as e:
        print(f"WARNING: ONNX export failed ({e}); evaluating float model.")
        model_float = model
        # Re-export without quantization for evaluation only
        onnx_path_fallback = "checkpoints/compress_candidate_float.onnx"
        export_onnx(model_float, onnx_path_fallback, IMG_SIZE)
        onnx_path = onnx_path_fallback

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
    print(f"model_size_mb:{metrics['model_size_mb']:.3f}")
    print(f"cpu_latency_ms:{metrics['cpu_latency_ms']:.1f}")
    print("---")
