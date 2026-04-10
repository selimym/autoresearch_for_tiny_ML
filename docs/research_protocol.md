# Research Protocol: TinyDet Architecture Search

## 1. Overview

The pipeline is a four-stage process: Phase 1 (LLM-driven architecture search on a cheap fixed budget) feeds survivors to Phase 1.5 (controlled reranking of top-K finalists under a stronger training recipe), whose winner proceeds to Phase 2 (quantization and compression search), which concludes at the Phase 2→3 handoff where a final compressed model is selected and optionally compared against an external baseline. Each stage produces a ranked candidate set; the next stage only proceeds on validated survivors — candidates that fail artifact resolution or gate checks are excluded, never silently substituted.


## 2. Required Validation Sequence

### Phase 0 (optional but recommended) — Subset proxy validation

- **Script:** `uv run scripts/validate_subset_corr.py --top-k 8 --seeds 2`
- **Purpose:** verify that ranking on the 5K/4-epoch cheap budget correlates (Spearman ρ ≥ 0.8) with ranking on larger subsets (10K, 20K), confirming the Phase 1 proxy is trustworthy before running an overnight search.
- **Required outputs:** `results/subset_corr.tsv` + printed correlation summary with per-subset ρ and p-value.
- **Gate:** if ρ < 0.8 for any comparison subset, increase subset size or epochs in `shinka_phase1.yaml` before proceeding to Phase 1. Running Phase 1 with an unreliable proxy wastes compute and may produce a misleading ranking.

### Phase 1 — Architecture search

- **Script:** `uv run run_phase1.py --config shinka_phase1.yaml`
- **Purpose:** broad LLM-driven exploration of neck/head architecture variants under a cheap, fixed training budget (4 epochs, 5K subset, frozen backbone).
- **Constraints:** only architecture parameters inside the EVOLVE-BLOCK may mutate; `LR`, `EPOCHS`, and `BATCH_SIZE` are locked for the duration of Phase 1 to ensure that composite score differences reflect architecture quality, not training recipe variance.
- **Required outputs:** `results/phase1/` (one generation directory per evolved program) + `results/phase1.tsv` (ranked candidates with commit hash, score, mAP50, size, latency).
- **Gate:** Phase 1 completes when the configured generation budget is exhausted. Top-K candidates by composite score are promoted to Phase 1.5.

### Phase 1.5 (required) — Finalist reranking

- **Script:** `uv run scripts/rerank_finalists.py --top-k 5 --seeds 2 --epochs 10 --frozen-stages 2`
- **Purpose:** remove undertraining bias introduced by the Phase 1 cheap budget. Each top-K candidate is retrained from scratch under a longer, partially-unfrozen recipe with multiple seeds, producing mean ± std mAP and AP_small estimates that are comparable across architectures.
- **Primary selection metric:** mean mAP @ [.50:.95] across seeds (not mAP50, not composite score).
- **Secondary metrics consulted:** AP_small, cpu_latency_ms, model_size_mb.
- **Required outputs:** `results/phase15.tsv` (one row per candidate × seed) + printed multi-metric summary with means and standard deviations.
- **Gate:** promote only the candidate whose mean mAP margin over the runner-up exceeds the combined standard deviation (the promotion gate is printed explicitly by the script). If the margin is within noise, run additional seeds before deciding. Do not promote based on mAP50 alone.

### Phase 2 — Compression search

- **Script:** `uv run compress.py` (via ShinkaEvolve or standalone)
- **Purpose:** search quantization and pruning configurations for the Phase 1.5 winner, targeting the smallest model that retains acceptable mAP.
- **Constraints:** only `QUANT_MODE` values `none` and `ptq_int8_static` are permitted. Any run where quantized ONNX export fails is automatically rejected with status `INVALID` and excluded from ranking.
- **Required outputs:** `results/phase2.tsv` with columns including quant_mode, mAP50, mAP, model_size_mb, cpu_latency_ms, status.
- **Gate:** at least one valid (non-INVALID) candidate must exist before proceeding to handoff.

### Phase 2→3 handoff — Final selection

- **Script:** `python handoff.py --phase 2 [--baseline path/to/reference.onnx]`
- **Purpose:** select the final compressed model from Phase 2 candidates; optionally compare against an external reference ONNX to validate competitiveness.
- **Selection basis:** Pareto tradeoff across mAP50, mAP, AP_small, model_size_mb, and cpu_latency_ms. Do not rely on the composite score alone for the final champion choice — composite score is a Phase 1 screening heuristic, not a deployment criterion.
- **Required outputs:** `handoff.json` (selected model path, config, metrics) + optional baseline comparison table printed to stdout.
- **Gate:** final champion must have a resolved ONNX artifact and passing evaluation metrics before any downstream deployment use.


## 3. Selection Policy

The following rules define which candidates advance between stages:

- **Phase 1 → Phase 1.5:** top-K by composite score: `mAP50 / (0.5 * size_mb + 0.5 * latency_ms / 100)`. K defaults to 5.
- **Phase 1.5 → Phase 2:** winner by mean mAP@[.50:.95] across seeds. The candidate must clear the promotion gate: the margin between the winner's mean mAP and the runner-up's mean mAP must exceed the sum of their standard deviations. If the gate is not cleared, run more seeds.
- **Phase 2 → Phase 3:** Pareto selection across mAP, model size, and latency. Do not use the composite score for this decision.
- **Artifact resolution is mandatory:** both `rerank_finalists.py` and `validate_subset_corr.py` will print an ERROR and skip any candidate whose artifact directory or program file cannot be resolved. They will never silently fall back to the seed architecture. A candidate that cannot be resolved is treated as absent from the ranking.


## 4. Reproducibility

- All experiment functions accept a `seed` argument.
- Use `tinydet.repro.set_seed(seed)` at the start of every training run for consistent seeding of Python `random`, NumPy, and PyTorch (CPU and CUDA). Pass `deterministic=True` for final validation runs where exact reproducibility matters more than throughput.
- Final claims must be supported by at least 2 independent seeds on the finalist(s). Single-seed results are preliminary.
- For each reported result, record: architecture config (from the evolved program file), training config (epochs, frozen stages, LR, batch size), seed, subset size, path to the exported ONNX, and all evaluation metrics from `evaluate_core.evaluate_model`.


## 5. Common Failure Modes

- **Fast-learner bias:** an architecture that converges quickly under the 4-epoch Phase 1 budget may not be the best long-term learner. Phase 1.5 (`rerank_finalists.py`) is the mandatory mitigation — do not skip it.
- **Recipe confound:** if the training recipe (LR, epochs, warmup) varies between Phase 1 candidates, score differences reflect recipe sensitivity rather than architecture quality. The EVOLVE-BLOCK constraint in Phase 1 and the fixed recipe in Phase 1.5 guard against this.
- **Proxy latency mismatch:** `cpu_latency_ms` measured on the development machine may not reflect the target deployment hardware. Treat latency rankings as relative indicators and validate on representative hardware before final model selection.
- **mAP50 overstatement:** mAP50 is easier to inflate than mAP@[.50:.95] and systematically overstates detection quality for overlapping or small objects. Phase 1.5 uses mAP@[.50:.95] as the primary selection metric; consult AP_small separately via the printed summary.
- **Noise-driven winner selection:** with small candidate sets and few seeds, rank differences can be dominated by noise. The promotion gate in `rerank_finalists.py` quantifies this: if the gate is not cleared, the ranking is unreliable and additional seeds are required before promotion.
