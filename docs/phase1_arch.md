# Phase 1: Architecture Search Reference

## UIB Block Variants

The `UniversalInvertedResidual` class (from `blocks/uib.py`) supports four configurations via `dw_kernel_size_start`, `dw_kernel_size_mid`, `dw_kernel_size_end`:

| Variant | dw_start | dw_mid | dw_end | Notes |
|---|---|---|---|---|
| Inverted Bottleneck (IB) | 0 | 3 | 0 | Standard MobileNet block — best default |
| ConvNext-like | 3 | 0 | 0 | Larger receptive field, fewer ops |
| ExtraDW | 3 | 3 | 0 | Two depthwise convs, best for large objects |
| FFN/Pointwise | 0 | 0 | 0 | No depthwise at all — fastest, weakest |
| Full (all 3) | 3 | 3 | 3 | Most capacity, heaviest |

`exp_ratio` controls the expansion width in the pointwise stages: 2.0 (lightweight) → 8.0 (heavy).

## FPN vs PAN

The default `FPNNeck` does top-down only: P5 → P4 → P3 (each level gets context from coarser levels).

A PAN (Path Aggregation Network) adds a bottom-up path: P3 → P4 → P5 after top-down. This helps P4/P5 levels detect large-object context aggregated from fine-grained P3 features. LLMs often discover PAN is better than FPN for this task.

To implement PAN in the EVOLVE-BLOCK, after the top-down pass, add:
```python
# Bottom-up pass
for i in range(1, len(laterals)):
    down = nn.functional.avg_pool2d(laterals[i-1], kernel_size=2, stride=2)
    laterals[i] = laterals[i] + down
```

## FCOS Level Assignment

Each FPN level handles objects of a specific scale:
- P3 (stride 8): small persons (< 64px height)
- P4 (stride 16): medium persons (64–128px)
- P5 (stride 32): large persons (> 128px)

A point at P3 grid position (i,j) has center coordinates `(j*8 + 4, i*8 + 4)` in image pixels.

## Search Space Summary

| Hyperparameter | Default | Searchable range |
|---|---|---|
| NECK_CHANNELS (per level) | [96, 96, 96] | 48–192, can be asymmetric |
| UIB variant per level | IB, ExtraDW, mixed | Any of 5 variants |
| UIB exp_ratio | 4.0 | 2.0–8.0 |
| HEAD_CHANNELS | 64 | 32–128 |
| HEAD_STACKS | 3 | 2–4 |
| FROZEN_STAGES | 4 | 2–4 |
| LR | 1e-3 | 5e-4–3e-3 |
| EPOCHS | 4 | 3–6 |

## Reading the ShinkaEvolve Pareto Curve

After Phase 1, `results/phase1/` contains an archive sorted by `combined_score`. Look for:
1. **Dominant cluster**: high mAP50 + small model → high score
2. **mAP50-only winners**: large model, good accuracy — useful to understand accuracy ceiling
3. **size-only winners**: tiny model, low accuracy — useful to understand compression limit

`handoff.py --phase 1` automates Pareto selection.
