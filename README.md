# autoresearch-tinydet

Autonomous person-detection compression research for ARM CPU deployment.

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Same paradigm — give an agent a pipeline and let it explore the accuracy-efficiency tradeoff space overnight — applied to detection compression rather than language model pretraining.

## What it does

Three sequential overnight phases, each building on the previous best:

| Phase | Method | Tool | Time budget |
|---|---|---|---|
| Phase 1: Architecture Search | Evolutionary NAS (Pareto archive) | ShinkaEvolve | Night 1 (~80 experiments) |
| Phase 2: Quantization | PTQ/QAT exploration | autoresearch loop | Night 2 (~100 experiments) |
| Phase 3: Pruning | Structured/unstructured + recovery | autoresearch loop | Night 3 (~35 experiments) |

**Model:** MobileNetV4-Conv-S backbone (timm, frozen) + FPN neck with UIB blocks + FCOS head → INT8 ONNX for ARM CPU deployment.

## Results

| Model | mAP50 | Size (MB) | Score | Pi 3B+ latency |
|---|---|---|---|---|
| YOLOv8n baseline | — | — | — | — |
| Phase 1 champion | — | — | — | — |
| Phase 2 champion | — | — | — | — |
| Phase 3 champion | — | — | — | — |

*(Fills as experiments run)*

## Quick Start

```bash
# Install
uv sync

# Set your API key (Gemini is the default LLM — get a free key at aistudio.google.com)
cp .env.example .env   # then edit .env and fill in GEMINI_API_KEY

# One-time data setup (~20 min, downloads COCO 2017 val + annotations)
uv run prepare.py

# Smoke test architecture (no COCO required)
uv run python initial_compress.py

# Smoke test full evaluation loop
uv run python shinka_evaluate.py --program_path initial_compress.py --results_dir /tmp/smoke

# Phase 1: overnight run
uv run python run_phase1.py --config shinka_phase1.yaml

# Morning: review and select
python handoff.py --phase 1

# Phase 2: overnight
# Agent runs: git commit → uv run python compress.py → grep run.log → log TSV

# Morning: select best
python handoff.py --phase 2

# Phase 3: overnight
# ...

# Pi benchmark (run on Pi):
python benchmark_pi.py --model checkpoints/phase3_champion.onnx
```

## Design

- **evaluate_core.py** — fixed shared evaluation (mAP50 via pycocotools, ONNX CPU latency, MLflow)
- **initial_compress.py** — Phase 1 seed with `# EVOLVE-BLOCK-START/END` markers; ShinkaEvolve mutates the block
- **compress.py** — Phase 2-3 single-file autoresearch target; agent edits quantization + pruning config blocks
- **UIB blocks** — copied from MobileNetV4 into `blocks/` for in-repo reviewability
- **evaluate.py** — OpenEvolve-compatible interface; also works as Phase 2-3 CLI evaluator
- **handoff.py** — human-run phase transition: Pareto selection, checkpoint copy, branch creation

See `docs/architecture.md` for full system diagram.

## Score

`score = mAP50 / model_size_mb` where `mAP50 >= 0.15` (below floor → score = 0). Higher is better. This rewards models that are both accurate and compact.
