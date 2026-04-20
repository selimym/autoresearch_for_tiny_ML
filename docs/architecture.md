# System Architecture

## Pipeline Overview

```
COCO 2017 (5K subset)
       |
       v
+-----------------------------------------------------+
|  Phase 1: Architecture Search (ShinkaEvolve)        |
|                                                     |
|  initial_compress.py --[EVOLVE-BLOCK]--> run_experiment() |
|         |                                           |
|         v                                           |
|  MNv4-Conv-S backbone (frozen)                      |
|         |                                           |
|  FPN neck (UIB blocks) <-- LLM mutates configs      |
|         |                                           |
|  FCOS head (cls/reg/ctr per level)                  |
|         |                                           |
|  evaluate_core.py -> score = mAP50/model_size_mb    |
|         |                                           |
|  ShinkaEvolve Pareto archive (40 slots)             |
+-----------------------------------------------------+
       |
       v handoff.py --phase 1
+-----------------------------------------------------+
|  Phase 2: Quantization (autoresearch loop)          |
|                                                     |
|  compress.py (QUANT_MODE block) <-- agent edits    |
|         |                                           |
|  PTQ/QAT/dynamic quantization                       |
|         |                                           |
|  evaluate.py -> score printed -> agent logs TSV     |
+-----------------------------------------------------+
       |
       v handoff.py --phase 2
+-----------------------------------------------------+
|  Phase 3: Pruning (autoresearch loop)               |
|                                                     |
|  compress.py (PRUNE_RATIO block) <-- agent edits   |
|         |                                           |
|  Structured/unstructured pruning + recovery         |
|         |                                           |
|  evaluate.py -> score -> agent keeps if improved    |
+-----------------------------------------------------+
       |
       v
  checkpoints/phase3_champion.onnx
  benchmark_pi.py (on Pi 3B+)
```

## Design Decisions

### Why FCOS (anchor-free)?
No anchor hyperparameter tuning. FCOS assigns positives via center-inside-box rule. centerness weighting suppresses off-center false positives. Well-suited to single-class person detection.

### Why UIB blocks in the neck?
Universal Inverted Bottleneck supports four variants (IB, ConvNext-like, ExtraDW, FFN) via the same class — one parameter set controls them all. This makes the search space discrete and interpretable for the LLM proposer.

### Why FPN (not PAN)?
FPN top-down is the simplest starting point. Phase 1 may discover that PAN (adding a bottom-up path) improves P3 detection of small persons. The evolve block's build_model() is free to implement PAN instead.

### Why 5K COCO subset?
4-epoch training runs in ~5 min on an RTX 3060. This enables ~80 Phase 1 experiments overnight. If mAP50(5K) ≈ mAP50(10K) within 1pp, 5K is sufficient.

### Why score = mAP50 / model_size_mb?
Directly rewards the compression objective: maximize accuracy per MB. The floor (mAP50 < 0.15 → score = 0) prevents degenerate models (e.g., 0.1 MB ONNX with 0.01 mAP50) from crowding the Pareto archive.

## Key Files

| File | Role | Who edits it |
|---|---|---|
| `initial_compress.py` | Phase 1 seed (EVOLVE-BLOCK) | ShinkaEvolve LLM |
| `compress.py` | Phase 2-3 template | autoresearch agent |
| `evaluate_core.py` | Shared evaluation | Never |
| `train_utils.py` | Backbone, dataloader, ONNX export | Never |
| `shinka_evaluate.py` | ShinkaEvolve adapter | Never |
| `evaluate.py` | Phase 2-3 CLI + OpenEvolve | Never |
| `handoff.py` | Phase transition utility | Human (between phases) |
| `benchmark_pi.py` | ARM latency measurement | Human (on Pi) |
