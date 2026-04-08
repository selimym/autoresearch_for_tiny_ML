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
# 1. Install
uv sync

# 2. API key (Gemini is the default LLM — free key at aistudio.google.com)
cp .env.example .env   # edit .env and fill in GEMINI_API_KEY

# 3. Hardware config — detects your RAM/VRAM and sets safe batch_size,
#    num_workers, pin_memory in .env automatically.
#    WSL2 users: first give WSL more RAM (see WSL2 Memory below).
uv run python hw_config.py           # print recommendations
uv run python hw_config.py --export  # write them into .env

# 4. One-time data setup (~20 min, downloads COCO 2017 val + annotations)
uv run prepare.py

# 5. Smoke test architecture (no COCO required)
uv run python initial_compress.py

# 6. Smoke test full evaluation loop
uv run python shinka_evaluate.py --program_path initial_compress.py --results_dir /tmp/smoke

# 7. Subset sanity check — trains the baseline on 5K/10K/20K subsets and
#    recommends which training set size to use for Phase 1 experiments.
#    Runs 3 sequential training+eval rounds; takes ~30-60 min on CPU.
uv run scripts/subset_sanity_check.py
# Or run one size at a time to reduce peak RAM:
uv run scripts/subset_sanity_check.py --sizes 5000

# 8. Phase 1: overnight architecture search
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
python scripts/benchmark_pi.py --model checkpoints/phase3_champion.onnx
```

## WSL2 Memory

By default WSL2 caps itself at ~50% of system RAM. If `hw_config.py` reports
less RAM than your machine has, create `C:\Users\<you>\.wslconfig` on Windows:

```ini
[wsl2]
memory=12GB    # adjust to ~75% of your physical RAM
swap=4GB
```

Then restart WSL: `wsl --shutdown` (from a Windows terminal), reopen WSL, and
re-run `hw_config.py --export` to update `.env` with the new limits.

## Config reference

`hw_config.py --export` writes these into `.env`; you can also set them manually:

| Variable | Default | Effect |
|---|---|---|
| `TRAINING_BATCH_SIZE` | `8` | Batch size for training loops in `compress.py` and `subset_sanity_check.py` |
| `DATALOADER_WORKERS` | `0` | DataLoader worker processes (each copies COCO JSON into RAM — 0 is safe on low-RAM machines) |
| `DATALOADER_PIN_MEMORY` | `0` | Page-locked memory transfers (only useful with real CUDA DMA, not WSL2) |
| `DATALOADER_PERSISTENT_WORKERS` | `0` | Keep workers alive between epochs (only when `WORKERS > 0`) |
| `TRAINING_DEVICE` | `cpu` | `cuda` or `cpu` |

> `initial_compress.py` is mutated by ShinkaEvolve agents and has its own
> hardcoded `BATCH_SIZE = 8`. Agents may change this value as part of the search.

## Project layout

```
tinydet/              # shared library (backbone, data, ONNX I/O, hw detection)
blocks/               # UIB blocks copied from MobileNetV4 (reviewable in-repo)
scripts/              # one-off tools: subset_sanity_check.py, benchmark_pi.py
docs/                 # architecture and phase guides

# Pipeline entry points (root — required by ShinkaEvolve path resolution)
initial_compress.py   # Phase 1 seed: EVOLVE-BLOCK mutated by ShinkaEvolve
shinka_evaluate.py    # ShinkaEvolve evaluator adapter
compress.py           # Phase 2-3: agent edits quantization + pruning blocks
evaluate_core.py      # fixed shared evaluation (mAP50, latency, MLflow)
evaluate.py           # CLI evaluator / OpenEvolve-compatible interface
run_phase1.py         # Phase 1 launcher
prepare.py            # one-time COCO data download + subset index builder
handoff.py            # phase transition: Pareto selection, checkpoint copy
hw_config.py          # hardware config advisor CLI
train_utils.py        # re-export shim for tinydet.* (backwards compat)
program_phase2.md     # Phase 2 agent instructions
program_phase3.md     # Phase 3 agent instructions
```

See `docs/architecture.md` for the full system diagram.

## Score

`score = mAP50 / model_size_mb` where `mAP50 >= 0.15` (below floor → score = 0). Higher is better. This rewards models that are both accurate and compact.
