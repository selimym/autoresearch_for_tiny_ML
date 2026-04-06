# Phase 2: Quantization Guide

## Overview

Phase 2 is **non-advancing**: every experiment starts from the same Phase 1 float checkpoint (`phase1_best.pt`). The goal is to find the best quantization strategy — not to chain experiments.

## Quantization Modes

| QUANT_MODE | Size reduction | mAP50 impact | Speed on ARM |
|---|---|---|---|
| `none` | 0% (FP32 baseline) | 0 | slowest |
| `dynamic` | ~20% | minimal | fast |
| `ptq_int8` | ~75% | 0.5–2pp drop | fast |
| `qat_int8` | ~75% | minimal (QAT-tuned) | fast |
| `ptq_int4` | ~87% | 3–8pp drop | fastest |

## PTQ vs QAT

**PTQ (Post-Training Quantization)**: quantize a pretrained float model using a small calibration dataset. Fast — no retraining. `CALIB_BATCHES` controls how much calibration data is used (more = better observer statistics, diminishing returns past 32 batches).

**QAT (Quantization-Aware Training)**: simulate quantization during training using fake quantization nodes. Requires fine-tuning (4–8 epochs). Recovers 0.5–2pp mAP50 lost by PTQ. Use when PTQ drops accuracy below the `MAP_FLOOR = 0.15` threshold.

## Per-Channel vs Per-Tensor

ONNX Runtime's `quantize_static` supports both:
- **Per-tensor**: one scale factor per layer. Faster on ARM. May lose accuracy on layers with high weight variance.
- **Per-channel**: one scale factor per output channel. Better accuracy (+0.5–1pp mAP50 typically). Slightly slower.

To switch to per-channel in `apply_quantization()`:
```python
from onnxruntime.quantization import QuantType, QuantizationMode
oq.quantize_static(fp32_path, int8_path, CalibReader(calib_data),
                   weight_type=QuantType.QInt8,
                   per_channel=True)
```

## Calibration Strategies

- **MinMax**: default. Fast. Can be sensitive to outliers.
- **Histogram (percentile)**: more robust. Use `CalibrationMethod.Percentile` with `calibrate_method` arg.
- **Entropy**: minimizes KL divergence. Often best for int8 but slowest to calibrate.

## INT4 Expectations

INT4 quantization (4-bit weights) typically requires mixed-precision: INT4 weights but INT8/FP16 activations. ONNX Runtime supports this via `QuantType.QUInt4` / `QuantType.QInt4`. Expect:
- Model size: ~0.5 bytes/param → ~87% smaller than FP32
- mAP50 drop: 3–8pp for detection models (activations stay in higher precision)
- If mAP50 drops below 0.15: try QAT recovery or reduce INT4 scope to fewer layers

## Agent Workflow

1. Start with `QUANT_MODE="none"` (FP32 baseline) — records the score baseline
2. Try `dynamic` — quick sanity check
3. Try `ptq_int8` with CALIB_BATCHES=16, then 32 — usually the best tradeoff
4. If mAP50 drops too much: try `qat_int8` with 4+ recovery epochs
5. Try `ptq_int4` — only if INT8 score is already strong
