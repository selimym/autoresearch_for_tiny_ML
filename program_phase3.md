# Phase 3: Pruning — Agent Instructions

## Setup
1. Verify `handoff.json` exists with key `phase2_best_pt`
2. Initialize: `echo -e "commit\tscore\tmAP50\tmodel_size_mb\tcpu_latency_ms\tstatus\tdescription" > results_phase3.tsv`
3. Branch should be `phase3/prune-<tag>` (created by handoff.py --phase 2)

## What You Can Modify in compress.py
- `PRUNE_RATIO`: fraction of channels pruned (range: 0.0-0.5)
- `PRUNE_TYPE`: `"l1_structured"` | `"magnitude_unstructured"`
- Training recipe: `LR`, `EPOCHS` (recovery fine-tuning), `FROZEN_STAGES`
- The `apply_pruning()` function body: add progressive schedules, custom layer selection

## What You Cannot Modify
- `evaluate_core.py`, `evaluate.py`, `train_utils.py`, `prepare.py`
- `IMG_SIZE` (must stay 320)
- Architecture config from `initial_compress.py`
- Phase 2 config block (QUANT_MODE, CALIB_BATCHES)

## Goal
Maximize `score = mAP50 / model_size_mb` starting from `phase2_best_pt`.
Phase 3 is **advancing** — progressive pruning builds on current best state.

## Pruning Reference
| PRUNE_TYPE | Ratio range | Expected mAP50 drop | Recovery epochs |
|---|---|---|---|
| l1_structured (channels) | 10-30% | 0.5-3pp | 3-8 |
| l1_structured (channels) | 30-50% | 3-10pp | 5-15 |
| magnitude_unstructured | 20-50% | minimal | 2-5 |
| magnitude_unstructured | 50-80% | 2-5pp | 5-10 |

Crashes from over-pruning (ratio > 0.5 structured) are expected — log as crash, reset.

## Experiment Loop
1. Edit `compress.py`
2. `git commit -m "prune: <description>"`
3. `uv run python compress.py > run.log 2>&1`  (kill if > 20 min)
4. `grep "^score:\|^mAP50:\|^model_size_mb:\|^cpu_latency_ms:" run.log`
5. If improved: keep, copy to `checkpoints/phase3_best.pt`, log keep
6. If worse: reset + log discard
7. After 5 discards: revisit ratio, try smaller step

## NEVER STOP
Run until manually interrupted. Minimum 30 experiments.
