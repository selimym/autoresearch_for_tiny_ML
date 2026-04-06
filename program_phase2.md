# Phase 2: Quantization — Agent Instructions

## Setup
1. Verify `handoff.json` exists with key `phase1_best`
2. Verify COCO val loads: `python -c "from train_utils import make_dataloader; next(iter(make_dataloader('val', 2, 320))); print('OK')"`
3. Initialize results TSV: `echo -e "commit\tscore\tmAP50\tmodel_size_mb\tcpu_latency_ms\tstatus\tdescription" > results_phase2.tsv`
4. Branch should be `phase2/quant-<tag>` (created by handoff.py --phase 1)

## What You Can Modify in compress.py
- `QUANT_MODE`: try each of `"none"`, `"ptq_int8"`, `"qat_int8"`, `"ptq_int4"`, `"dynamic"`
- `CALIB_BATCHES`: calibration dataset size (range: 8-64)
- Training recipe: `LR`, `EPOCHS`, `BATCH_SIZE`, `WARMUP_EPOCHS`, `FROZEN_STAGES`
- The `apply_quantization()` function body: add calibration strategies, observer types

## What You Cannot Modify
- `evaluate_core.py`, `evaluate.py`, `train_utils.py`, `prepare.py`
- `IMG_SIZE` (must stay 320)
- Architecture config imported from `initial_compress.py`
- Phase 3 config block (PRUNE_RATIO, PRUNE_TYPE)

## Goal
Maximize `score = mAP50 / model_size_mb` where `mAP50 >= 0.15`.
Every experiment starts from the same `phase1_best.pt` float checkpoint — Phase 2 does NOT advance. This keeps all results directly comparable.

## Quantization Quick Reference
| QUANT_MODE | Size reduction | mAP50 impact | Speed |
|---|---|---|---|
| none | 0% (FP32 baseline) | 0 | slowest |
| dynamic | ~20% | minimal | fast |
| ptq_int8 | ~75% | small (0.5-2pp) | fast |
| qat_int8 | ~75% | minimal (tuned) | fast |
| ptq_int4 | ~87% | moderate (3-8pp) | fastest |

## Optional Literature Search
Before implementing an uncertain idea (e.g., mixed INT4/INT8 per-layer):
```python
from semanticscholar import SemanticScholar
ss = SemanticScholar()
results = ss.search_paper("INT4 quantization detection accuracy", limit=5)
for r in results: print(r.title, r.year, r.abstract[:200])
```
1-2 queries max, skim abstracts only, < 2 min total.

## Experiment Loop
1. Edit `compress.py`
2. `git commit -m "quant: <description>"`
3. `uv run python compress.py > run.log 2>&1`  (kill if > 12 min)
4. `grep "^score:\|^mAP50:\|^model_size_mb:\|^cpu_latency_ms:" run.log`
5. If improved: `echo -e "$(git rev-parse --short HEAD)\t<score>\t<mAP50>\t<size>\t<latency>\tkeep\t<desc>" >> results_phase2.tsv`
6. If worse or crash: `git reset --hard HEAD~1` + log as discard

## NEVER STOP
Run until manually interrupted. Minimum 80 experiments.
