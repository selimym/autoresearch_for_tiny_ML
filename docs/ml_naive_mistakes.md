# ML Naive Mistakes: Lessons from Three Rounds of External Feedback

This document distills the methodological problems found in three rounds of external review of this NAS-driven tiny-ML detection pipeline. Each mistake is presented as it appeared in this specific codebase, why it matters scientifically, and what was done to fix it. The audience is an ML practitioner comfortable with deep learning who is new to object detection and NAS pipelines.

The problems are ordered from most to least severe: confounds that corrupt the central scientific claim come first; noise and robustness issues that weaken but do not invalidate conclusions come last.

---

## 1. Experimental Validity: Search Confounds

### Training recipe mutation mixed into architecture search

**What went wrong.**
`shinka_phase1.yaml` originally invited the LLM proposer to mutate not only architecture parameters (`NECK_CHANNELS`, `UIB_CONFIGS`, `HEAD_CHANNELS`, `HEAD_STACKS`) but also training hyperparameters: learning rate, number of epochs, and batch size. Those constants lived inside the `EVOLVE-BLOCK` alongside the architecture, so they were fair game for any proposed mutation.

**Why it matters.**
The entire premise of Phase 1 is that the search finds a better *architecture*. When training budget and learning rate are also free to vary, you cannot tell whether a higher score came from a genuinely better neck topology or from a candidate that simply received more epochs or a more forgiving learning rate. A fast-learning recipe can beat a slow-learning architecture in a short run even when the latter would win under proper training. Every score comparison loses its meaning: you are no longer doing architecture search, you are doing joint architecture-plus-optimisation search with an unclear objective.

**The fix.**
`LR`, `EPOCHS`, `BATCH_SIZE`, and `WARMUP_EPOCHS` were moved outside the `EVOLVE-BLOCK` in `initial_compress.py` and are now declared as fixed constants in the file header. The `shinka_phase1.yaml` prompt was updated to state explicitly that only architectural decisions may be mutated and that training hyperparameters must not be changed. All Phase 1 candidates now train under the same controlled budget, so score differences are attributable to architecture alone.

> **Rule of thumb:** In any NAS or ablation experiment, hold the training recipe fixed. If you want to jointly search over recipes, treat that as a separate, explicitly acknowledged search dimension and report it honestly.

---

## 2. Proxy Validity: Trusting Cheap Metrics Too Much

### Short undertrained runs do not reliably rank architectures

**What went wrong.**
Phase 1 trains each candidate for 4 epochs on a 5,000-image subset with all backbone stages frozen (`FROZEN_STAGES = 4`). Neck and head weights are initialised from scratch. The resulting score is used to decide which architecture advances.

**Why it matters.**
Ranking quality after 4 epochs of frozen-backbone training is a noisy proxy for final deployed quality. Some architectures learn faster in the early phase but plateau lower; others are slow starters that outperform once the backbone begins to co-adapt. A model that happens to be compatible with the frozen backbone's feature distribution at this particular learning rate can win Phase 1 while a fundamentally better architecture loses. The selected finalist may therefore be "best at fast frozen learning" rather than "best for deployment."

**The fix.**
`scripts/rerank_finalists.py` implements a Phase 1.5 reranking stage. After overnight search, it takes the top-K Phase 1 candidates, loads each candidate's evolved program file from the Phase 1 artifact directory, and retrains each one under a controlled longer schedule (default 10 epochs, `FROZEN_STAGES=2`, 2 seeds). Per-seed metrics are written to `results/phase15.tsv` and a final summary sorted by mean mAP at [.50:.95] is printed along with a promotion gate: if the winner's mAP margin over the runner-up is smaller than the combined standard deviation across seeds, the script warns that promotion is statistically uncertain.

> **Rule of thumb:** Treat cheap search as a proposal generator, not a final ranking. Always rerank shortlisted candidates under a more realistic training budget before promoting any winner.

### mAP50-only evaluation is too permissive

**What went wrong.**
`evaluate_core.py` originally reported only `mAP50` (AP at IoU threshold 0.50). The composite score used `mAP50 / model_size_mb`.

**Why it matters.**
mAP50 is a generous metric. A detector that predicts bounding boxes with substantial positional error can still score well if the boxes overlap the ground-truth by more than 50%. This is especially misleading for an anchor-free detector like FCOS, where regression quality is critical, and for small persons at 320×320 resolution, where localization matters most. Reporting mAP50 alone can overestimate quality and hide architectural weaknesses that only manifest at stricter IoU thresholds.

**The fix.**
`evaluate_core.py` now computes three COCO metrics from `pycocotools`: `mAP50` (AP@.50, `evaluator.stats[1]`), `mAP` (AP@[.50:.95], `evaluator.stats[0]`), and `AP_small` (AP for objects with area < 32², `evaluator.stats[3]`). All three are returned in the metrics dictionary and logged. The reranking script in Phase 1.5 sorts the final summary by `mean_mAP` as the primary criterion, with `mean_mAP50` as a secondary tiebreaker, so stronger localization is rewarded at the selection stage.

> **Rule of thumb:** mAP50 is fine for early screening. For any model you intend to trust or promote, always report AP@[.50:.95] and AP_small. They reveal weaknesses that mAP50 conceals.

### Optimising model size as a proxy for latency

**What went wrong.**
The original score formula was `mAP50 / model_size_mb`. Latency was measured and logged but played no role in architecture selection.

**Why it matters.**
Two ONNX models of identical file size can have very different inference latency on ARM Cortex-A class hardware. Latency depends on operator type mix (depthwise vs pointwise convolutions), memory access patterns, feature map sizes, and neck topology. A model with a compact file size but expensive operators (e.g., a neck with many large feature maps) may be slower than a slightly larger model with a simpler operator graph. Selecting on size alone can therefore produce a suboptimal deployment choice even when the size metric looks attractive.

**The fix.**
The score formula in `evaluate_core.py` now incorporates both size and latency:

```
latency_norm = latency_ms / 100.0   # 100 ms ARM Cortex-A53 reference
denom = 0.5 * size_mb + 0.5 * latency_norm
score = mAP50 / denom   (if mAP50 >= 0.15, else 0)
```

The 50/50 weighting between size and latency is explicit and documented in the source. Note that the latency measured here is local ONNX Runtime CPU latency, not actual Raspberry Pi latency. It serves as a deployment proxy during search; final champion selection should still use real target-hardware measurements.

> **Rule of thumb:** Size is a fast-to-compute but imperfect proxy for latency. Include measured inference time in your search objective, and validate final candidates on the actual target hardware.

### 5K subset ranking may not preserve ordering from larger subsets

**What went wrong.**
Phase 1 trains on a fixed stratified 5,000-image subset. The subset selection was validated for absolute mAP level (i.e., 5K gives approximately the same mAP as the full set), but not for whether it preserves the *ranking* of different architectures.

**Why it matters.**
Even if the absolute mAP difference between a 5K run and a 50K run is small, the *ordering* of two architectures on 5K might be reversed on a larger set. An architecture that is slightly overfitted to the particular composition of the 5K subset could rank higher than a more generalisable design. In NAS, ranking fidelity is what matters, not absolute mAP closeness.

**The fix.**
`scripts/validate_subset_corr.py` computes Spearman rank correlation between the 5K ranking and rankings produced on 10K and 20K subsets for the top-K Phase 1 candidates. It emits a categorical conclusion (trustworthy / uncertain / unreliable) based on the correlation magnitude. This script does not automatically fix a low correlation; it surfaces the problem so the practitioner can either increase the search subset or treat Phase 1 results with lower confidence.

> **Rule of thumb:** Validate that your cheap proxy *preserves ordering*, not just that it gives similar absolute values. These are different properties.

---

## 3. Data Pipeline: Geometry and Augmentation

### Anisotropic square resize distorts person shapes and bounding box geometry

**What went wrong.**
The original `tinydet/data.py` resized every image directly to a square 320×320 canvas using a naive resize call, ignoring the original aspect ratio. A tall portrait-oriented image of a person would be horizontally stretched; a wide landscape image would be vertically squashed.

**Why it matters.**
Person bounding boxes are predominantly taller than they are wide. Anisotropic resize systematically distorts this geometry, making the training distribution inconsistent with any real-world deployment where images would be pre-processed differently. More concretely: the ground-truth `boxes` transformed under anisotropic resize no longer reflect real object proportions, regression targets become inaccurate for elongated objects, and an architecture that tolerates distorted aspect ratios better than another may rank higher despite being worse under realistic conditions. This is a form of data leakage into architecture comparison.

**The fix.**
`tinydet/data.py` now implements letterboxing: the image is scaled by `min(target_size / orig_w, target_size / orig_h)` to preserve the aspect ratio, then padded symmetrically with mid-gray (128) to reach the target canvas size. All bounding box coordinates are adjusted to account for both the scale factor and the padding offset. The augmentation stack was also extended beyond the original flip-only baseline to include scale jitter (±20% resize then crop/pad) and brightness/contrast jitter during training.

> **Rule of thumb:** Always preserve aspect ratio in detection preprocessing. Direct square resize is a silent performance cap that also contaminates architecture comparisons.

### Minimal augmentation leads to brittle baselines

**What went wrong.**
The original data pipeline applied only random horizontal flip during training. No scale jitter, no color augmentation.

**Why it matters.**
A detection model trained with minimal augmentation is more likely to overfit to the specific scale and lighting distribution of the training subset. With only 5,000 training images and 4 epochs, overfitting is a real concern. More importantly, a stronger augmentation baseline separates architecture quality from training-data sensitivity: a model that performs well under weak augmentation but fails under typical deployment variation is a poor architecture choice.

**The fix.**
`tinydet/data.py` now adds scale jitter (random resize to 80–120% then crop or pad back to `img_size`, with consistent box rescaling) and brightness/contrast jitter (±0.2) for the training split. The fix is intentionally modest — no mosaic, copy-paste, or large augmentation stacks — to keep the training recipe simple and reproducible while eliminating the most brittle aspects of the original setup.

> **Rule of thumb:** A detection baseline needs at least scale jitter and color jitter. Flip-only augmentation is not a defensible baseline for evaluating architecture quality.

---

## 4. Evaluation and Reporting Integrity

### Fake quantization: PyTorch dynamic quantization does not touch Conv2d weights

**What went wrong.**
An earlier version of `compress.py` offered multiple quantization modes including `ptq_int4` (implemented by reducing the observer range to approximate 4-bit) and `dynamic` (PyTorch's `torch.quantization.quantize_dynamic`). The `ptq_int4` mode was not real INT4 deployment — it was a float model with a narrowed observer range. Dynamic quantization on a Conv2d-heavy detector is largely a no-op for the convolutional weights; PyTorch's dynamic quantization targets Linear layers by default and leaves Conv2d untouched.

**Why it matters.**
Reporting a metric alongside the label `ptq_int4` implies that the model was quantized to 4 bits. If the underlying artifact is still a float32 model, the reported size, latency, and accuracy do not represent what an INT4 deployment would actually look like. Any conclusion about INT4 viability drawn from these numbers is invalid.

**The fix.**
`compress.py` was simplified to support only two valid modes: `none` (float32 baseline) and `ptq_int8_static` (genuine ONNX Runtime static INT8 quantization with calibration data). The `ptq_int4` and dynamic modes were removed. The static INT8 path uses `onnxruntime.quantization.quantize_static` with a real calibration reader (`_ValCalibrationDataReader`) that feeds 50 validation images through the float ONNX model to compute activation statistics before quantizing.

> **Rule of thumb:** Only claim a model is quantized if the exported artifact is genuinely quantized. "Approximate" quantization emulations produce misleading benchmarks.

### Silent fallback: failed quantization evaluated as float and reported as compressed

**What went wrong.**
The original `compress.py` wrapped both the ONNX export and the quantization step in try/except blocks that, on failure, printed a warning and then continued to evaluate the float ONNX file. The result was logged as the quantization experiment's outcome. If INT8 quantization of a particular architecture failed (a not-uncommon occurrence for non-standard operator graphs), the pipeline silently reported the float model's metrics as the compressed model's metrics.

**Why it matters.**
This is one of the most consequential integrity failures in the codebase. A researcher reading the output logs would see a mAP and latency number attributed to a quantized model, when in fact they were measuring the uncompressed float baseline. Any conclusion about which architecture quantizes better, or whether quantization degrades accuracy, would be drawn from corrupted data. The failure mode is especially dangerous because it is silent: there is no indication in the output that something went wrong.

**The fix.**
`compress.py` now treats ONNX export failure and quantization failure as hard stops. If `export_onnx` raises an exception, the script prints `status:INVALID` and exits with code 1. If `quantize_onnx_static` fails, the same hard stop applies. There is no fallback path to evaluating a float model and labeling it as a compressed result. A comment in the source states the intent explicitly: "Never silently fall back to evaluating an unquantized model — that would produce misleading metrics for a 'quantized' experiment."

> **Rule of thumb:** Any result that could have been produced by a fallback instead of the intended pipeline is not a valid result. Fail loudly or not at all.

### Silent artifact fallback in reranking: wrong program reranked when candidate file not found

**What went wrong.**
An earlier draft of `scripts/rerank_finalists.py` handled the case where `_find_candidate_dir` could not locate a Phase 1 candidate's artifact directory by falling back to the root `initial_compress.py` (the seed program). The reranking would then proceed — silently training and evaluating the baseline seed architecture — and the result would appear in the output as if it came from the intended finalist.

**Why it matters.**
The reranking stage exists to compare the actual evolved architectures under a stronger training regime. If a candidate's file is unresolvable and the script silently falls back to the seed, the summary table will contain at least one row that does not represent an evolved architecture. The user cannot tell which row is corrupted. This can cause a genuinely better evolved architecture to be passed over in favour of the seed, or inflate the seed's apparent ranking, or simply corrupt the comparison table.

**The fix.**
The current `scripts/rerank_finalists.py` raises an explicit error message when `_find_candidate_dir` returns `None` and skips that candidate entirely (`continue`) rather than substituting the seed program. The error message names the commit hash and explains that the `results_dir` should contain `gen_*` directories with the evolved program files. No fallback training happens; the missing candidate simply does not appear in the output.

> **Rule of thumb:** When a result cannot be produced from the correct artifact, skip it and flag it — never substitute a proxy. A missing data point is less harmful than a wrong one.

---

## 5. FCOS Training Quality

### All points inside a GT box as positives degrades level specialisation and biases architecture comparisons

**What went wrong.**
The original FCOS loss in `initial_compress.py` assigned as positive every spatial grid point that fell inside any ground-truth bounding box at any FPN level. There was no scale-of-interest constraint (no rule that P3 handles small objects and P5 handles large ones) and no center-sampling heuristic (no constraint that only points near the box center are eligible).

**Why it matters.**
Without scale-of-interest filtering, a large person bbox spanning 200 pixels is simultaneously a positive training target on P3 (stride 8, fine detail), P4 (stride 16), and P5 (stride 32, coarse context). The three FPN levels cannot specialise: P3 learns to regress large objects it was not designed for, P5 tries to handle tiny overlapping persons. This produces noisy regression targets, weak level specialisation, and worse detection performance — independently of which architecture is being evaluated. When architecture comparison is done on top of a lossy training recipe, measured differences between architectures partly reflect how each architecture interacts with the noisy assignment rather than their true relative quality. The search can therefore develop a preference for architectures that happen to be more robust to noisy positives rather than architectures that are genuinely better detectors.

**The fix.**
`initial_compress.py` now implements two standard FCOS heuristics. First, scale-of-interest filtering: each FPN level defines a minimum and maximum ground-truth area it will accept as a positive (`_SOI_MIN = [0, 32², 64²]`, `_SOI_MAX = [96², 192², inf]`). A GT box whose area falls outside a level's range does not produce positives at that level. Second, center sampling: a grid point is only eligible if it lies within `1.5 * stride` pixels of the GT box's geometric center (`_CENTER_RADIUS = 1.5`). Points at the edges of boxes, which produce high-variance regression targets, are excluded. Together these reduce noisy positives and improve level specialisation, making architecture score differences more attributable to architecture rather than to assignment artifacts.

> **Rule of thumb:** Use a real FCOS assignment recipe (scale-of-interest + center sampling) or a well-tested detection head. A simplified assignment strategy introduces systematic noise that corrupts architecture comparisons.

---

## 6. Robustness and Noise

### Single-seed finalist selection: one lucky or unlucky initialisation can determine the winner

**What went wrong.**
Every Phase 1 candidate was trained and evaluated once, with a fixed random seed. The candidate with the highest composite score on that single run was promoted to Phase 2.

**Why it matters.**
Neural network training on small subsets (5,000 images) with random weight initialisation has meaningful variance. The mAP difference between two architectures after 4 epochs on a frozen backbone can easily be dominated by a lucky or unlucky initialisation rather than by architecture quality. If two candidates differ by 0.5 points of mAP50 and the initialisation variance is ±0.3 points, the ranking is essentially a coin flip. Promoting the wrong architecture propagates the error through all subsequent phases: Phase 2 compresses the wrong model, Phase 3 prunes it, and the final evaluation reflects the choices made at this noisy Stage 1 decision.

**The fix.**
`scripts/rerank_finalists.py` runs each Phase 1.5 candidate with multiple random seeds (default 2, configurable via `--seeds`). It reports mean ± std for both mAP50 and mAP across seeds. The promotion gate explicitly checks whether the winner's mAP advantage over the runner-up is larger than the combined standard deviation of both models. If not, it emits a warning: "PROMOTION UNCERTAIN: winner mAP margin (X) is smaller than combined std (Y). Consider running more seeds before promoting."

> **Rule of thumb:** Never select a NAS winner from a single training run. At minimum run 2–3 seeds on the finalists and report uncertainty. If the winner's margin is within noise, run more seeds before deciding.

### No reranking stage between cheap search and promotion: search score is a weak proxy for final quality

**What went wrong.**
The original pipeline promoted the Phase 1 winner directly to Phase 2 compression, using only the cheap 4-epoch frozen-backbone score as the selection criterion. There was no intermediate validation stage.

**Why it matters.**
The Phase 1 search is designed to be cheap: it uses a small subset, few epochs, and a frozen backbone to explore many candidates quickly. This is entirely appropriate as a *proposal mechanism*. The problem arises when the cheap search score is also used as the *final selection criterion*. The search proxy has several known weaknesses (short training, frozen backbone, 5K subset, mAP50-only gate) and conflating it with final quality leads to the selection of architectures that are "good at being found cheaply" rather than architectures that are "good for deployment." Any systematic gap between cheap-search quality and real quality becomes a source of compounding error: later phases invest compute in a suboptimal starting point.

**The fix.**
The pipeline now has an explicit Phase 1.5 reranking step (`scripts/rerank_finalists.py`) that sits between cheap search and Phase 2 promotion. The reranking trains under a longer schedule with partial backbone unfreezing and reports multi-metric results (mAP50, mAP, AP_small, latency, size) averaged across seeds. Only after this stage does the practitioner decide which architecture to hand off to Phase 2. The Phase 1 score is treated as a ranking signal for filtering, not as a measure of absolute quality.

> **Rule of thumb:** Cheap search finds candidates; a separate rerank stage selects the champion. Never promote directly from the cheapest proxy to the most expensive phase.

---

## Summary Table

| Mistake | Severity | File(s) | Status |
|---|---|---|---|
| Recipe mutation in search space | Critical | `initial_compress.py`, `shinka_phase1.yaml` | Fixed |
| Silent float fallback in quantization | Critical | `compress.py` | Fixed |
| Silent seed fallback in reranking | Critical | `scripts/rerank_finalists.py` | Fixed |
| Anisotropic square resize | High | `tinydet/data.py` | Fixed |
| Fake quantization modes | High | `compress.py` | Fixed |
| mAP50-only evaluation | High | `evaluate_core.py` | Fixed |
| Size-only score proxy | Medium | `evaluate_core.py` | Fixed (partial) |
| Minimal augmentation | Medium | `tinydet/data.py` | Fixed |
| FCOS assignment without scale/center | Medium | `initial_compress.py` | Fixed |
| Single-seed finalist selection | Medium | `scripts/rerank_finalists.py` | Fixed |
| No reranking before promotion | Medium | `scripts/rerank_finalists.py` | Fixed |
| Subset ranking correlation unvalidated | Low | `scripts/validate_subset_corr.py` | Added diagnostic |

The open items are: deployment-faithful latency (current measurement is host CPU ONNX Runtime, not Raspberry Pi ARM), and a formal codified promotion policy that makes reranking mandatory rather than optional. Both are documented in `docs/research_protocol.md`.
