# Phase 3: Pruning Guide

## Overview

Phase 3 is **advancing**: the agent builds on the current best checkpoint, progressively pruning and fine-tuning. The goal is to maximize `score = mAP50 / model_size_mb` by reducing model size while recovering accuracy via fine-tuning.

## Pruning Types

| PRUNE_TYPE | What it prunes | Size reduction | Speed impact |
|---|---|---|---|
| `l1_structured` | Entire channels (output filters) | Yes — real ONNX size reduction | Yes — fewer multiplies |
| `magnitude_unstructured` | Individual weights | No† — sparsity only | Only with sparse runtime support |

†Unstructured pruning creates sparse tensors. Without ARM sparse runtime support, the ONNX model doesn't get smaller or faster — only the weight count drops. Use structured pruning for real compression gains.

## Ratio Ranges and Expected Impact

| PRUNE_RATIO | Expected mAP50 drop | Recovery epochs |
|---|---|---|
| 0.10 (10%) | 0.5–1pp | 2–3 |
| 0.20 (20%) | 1–3pp | 3–5 |
| 0.30 (30%) | 3–6pp | 5–8 |
| 0.40 (40%) | 6–10pp | 8–15 |
| > 0.50 | Often catastrophic | May not recover |

## Expected Failure Modes

1. **Over-pruning crash**: ratio > 0.5 structured on narrow layers (e.g., NECK_CHANNELS=[48,48,48]) may prune all channels in a layer → runtime error. Fix: cap ratio at 0.4, or exclude the last conv in each block.

2. **Score regression after recovery**: fine-tuning recovers mAP50 but model size stays the same → score unchanged. Cause: pruning mask was applied but `torch.nn.utils.prune.remove()` not called before ONNX export. Fix: call `prune.remove(module, 'weight')` for each pruned module.

3. **mAP50 below floor after pruning**: score = 0.0. Recovery with more epochs (10+) often helps. If not, reduce ratio.

## Progressive Pruning Schedule

Instead of one large prune step, pruning iteratively in smaller steps recovers better:
```python
for ratio in [0.05, 0.05, 0.10]:  # total: 20%
    model = apply_pruning(model, PRUNE_TYPE, ratio)
    # fine-tune 3 epochs
    # evaluate
```

The agent can implement this inside `apply_pruning()` or in the main loop.

## Important: Remove Pruning Masks Before ONNX Export

PyTorch pruning works by adding a `weight_mask` buffer and a `weight_orig` parameter. The actual `weight` tensor is recomputed via hook. ONNX export sees the underlying structure, not the masked weights. To actually make the model smaller:

1. For **structured** pruning: after removing zero-channels, rebuild the module without those channels (or use a library like `torch-pruning`).
2. Minimum: call `torch.nn.utils.prune.remove(module, 'weight')` to make the mask permanent before ONNX export.

Without this step, the ONNX file will be the same size as before pruning.
