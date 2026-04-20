# ML Techniques Reference

A bottom-up guide to every significant technique used in this project.
Written for someone who knows classification and GANs well, but is new to detection and NAS.

---

## Part 1 — Foundations (familiar ground)

### Transfer learning and frozen backbones

You already know this from classification: pretrain on ImageNet, fine-tune on your task. The same idea applies here, but with a twist.

In classification you typically unfreeze everything and fine-tune the whole network. In detection, especially with a tiny dataset or a short search budget, you often **freeze most of the backbone** and only train the detection head from scratch. This is because:

- backbone features (edges, textures, object parts) are general and expensive to re-learn
- training from scratch on 5K images would severely overfit the backbone
- the neck and head have far fewer parameters and converge in a few epochs

In `initial_compress.py`, `FROZEN_STAGES = 4` means all four backbone stages are frozen; only the FPN neck and FCOS head train. The Phase 1.5 reranking step uses `FROZEN_STAGES = 2` to let the last two stages co-adapt — a more realistic estimate of final quality.

### Feature maps vs global pooling

In classification, you collapse the spatial dimensions with global average pooling before the linear classifier — the spatial location of a feature doesn't matter, only its presence.

Detection breaks this completely. You need to know **where** each object is, so spatial resolution must be preserved all the way to the output. A `(B, C, H, W)` feature map where each position `(h, w)` predicts something about the image region it corresponds to is the core abstraction. Every detection head operates on these spatial feature maps.

---

## Part 2 — Object Detection

### From classification to detection

Classification asks: "what is in this image?" Detection asks three things simultaneously:

1. **What** is it? (class label)
2. **Where** is it? (bounding box)
3. **How many** are there? (variable number per image)

The variable-count problem is what makes detection architecturally different. You can't just add a fixed output head — you need a mechanism that scans the image and proposes detections at arbitrary locations.

### Anchor-based vs anchor-free detection

**Anchor-based** methods (SSD, Faster R-CNN, YOLOv3) tile the image with predefined boxes of various sizes and aspect ratios ("anchors"). For each anchor, the network predicts an offset to refine it into a detection. This works well but requires careful anchor design and matching heuristics.

**Anchor-free** methods (FCOS, CenterNet, CornerNet) skip the anchor catalog entirely. Instead, each spatial location on the feature map predicts a detection directly. This project uses FCOS — simpler, no anchor hyperparameters to tune.

### Feature Pyramid Network (FPN)

The central problem for multi-scale detection: a person close-up occupies most of the frame (large object), while a person in the background is tiny (small object). No single feature map scale handles both well.

FPN solves this by building a pyramid of feature maps at different resolutions:

```
Input image
    ↓
Backbone
    C3 (stride 8,  64ch)   ← fine details, small objects
    C4 (stride 16, 96ch)   ← medium objects
    C5 (stride 32, 960ch)  ← coarse semantics, large objects
    ↓
FPN lateral connections + top-down pathway
    P3 ← C3 + upsampled(P4)   stride 8  → predicts small objects
    P4 ← C4 + upsampled(P5)   stride 16 → predicts medium objects
    P5 ← C5                   stride 32 → predicts large objects
```

The **lateral connections** project each `Ci` to a common channel width (`NECK_CHANNELS`, e.g. 96) before the top-down fusion. This lets the network combine high-resolution spatial information (from C3) with high-level semantic information (from C5).

The **top-down pathway** propagates strong semantic features from deep layers back to shallow, high-resolution layers. Without it, P3 would have good resolution but weak semantics and would struggle to recognize what it sees.

In `initial_compress.py`, after the lateral projections, each FPN level also passes through a **UIB refinement block** (see Part 3) before being handed to the detection head.

### PAN (Path Aggregation Network)

FPN only flows top-down (C5→C4→C3). PAN adds a **bottom-up path** (P3→P4→P5) after the top-down path. This gives deeper layers access to strong low-level features (sharp edges, fine textures) that get diluted in the top-down pass alone.

In this project's architecture search, `FPN vs PAN` is one of the mutable choices — the LLM can try bidirectional fusion.

### FCOS (Fully Convolutional One-Stage Detector)

FCOS is the detection formulation used in Phase 1. Instead of anchors, every spatial location `(h, w)` on each FPN level is a candidate detection point.

**What each location predicts:**

For a point at pixel position `(cx, cy)` (computed from grid position and stride):
- **Classification** — logit for each class (here, just person: 1 logit)
- **Regression** — 4 distances: `(l, t, r, b)` = left, top, right, bottom distance to the GT box edges
- **Centerness** — a scalar in [0,1] measuring how close the point is to the center of its assigned GT box

The regression encodes a box as `[cx-l, cy-t, cx+r, cy+b]` in pixel coordinates. During training the model outputs raw distances; `exp()` is applied to keep them positive.

**Centerness:**
```
centerness = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
```
It is 1.0 at the exact box center and decays to 0 at the edges. During inference, the final score is `sqrt(cls_score * centerness)` — this suppresses detections from off-center points that happen to have high classification confidence. Without centerness, FCOS tends to produce many low-quality boxes near box boundaries.

**Target assignment — which points are positive:**

A point is a positive for GT box `n` if it satisfies all three:
1. It lies **inside** the GT box (its `(cx, cy)` is between the box edges)
2. It lies within `center_radius × stride` of the box's **geometric center** (center sampling)
3. The GT box area falls within the **scale-of-interest** range for this FPN level (SOI)

Scale-of-interest per level (from `initial_compress.py`):
```
P3 (stride 8):  area ∈ [0,      96²)   → small objects
P4 (stride 16): area ∈ [32²,   192²)   → medium objects
P5 (stride 32): area ∈ [64²,   ∞  )    → large objects
```

Without SOI, large GT boxes would generate positives on P3 (fine-grained level), overwhelming the small-object signal. Without center sampling, points near box edges become positives but produce high-error regression targets, adding noise to training.

**Loss components:**

- **Focal loss** on classification logits — down-weights easy negatives, focuses on hard examples. Critical when positives are rare (most spatial locations are background).
- **IoU loss** on regression — directly optimizes box overlap quality rather than L1 distance on coordinates.
- **BCE** on centerness — binary cross-entropy against the centerness target.

### COCO evaluation metrics

COCO uses a specific evaluation protocol:

**AP@0.50 (mAP50):** A detection is a true positive if its IoU with a GT box ≥ 0.50. Easy threshold — a rough box that overlaps most of the GT passes. Good for screening but permissive on localization.

**AP@[.50:.95] (mAP):** Average of AP at IoU thresholds 0.50, 0.55, 0.60, ..., 0.95. Forces the detector to be precise. This is the primary COCO metric and is much harder to game with sloppy boxes.

**AP_small:** AP restricted to objects with area < 32² pixels. For 320×320 person detection this is important — many background persons are tiny, and small-object AP exposes whether the detector actually learns to find them.

The project now reports all three. The search score uses mAP50 for speed; Phase 1.5 finalist ranking uses mAP as the primary metric.

---

## Part 3 — Efficient Network Design

### MobileNet family

Standard convolutions are expensive: a `3×3` conv on a `C_in → C_out` feature map costs `O(H × W × C_in × C_out × k²)`. MobileNets cut this with **depthwise separable convolutions**:

- **Depthwise conv**: one `3×3` filter per input channel (no cross-channel mixing) — `O(H × W × C_in × k²)`
- **Pointwise conv**: `1×1` conv to mix channels — `O(H × W × C_in × C_out)`

Total: roughly `8–9×` cheaper than a standard `3×3` conv for typical channel counts.

The **Inverted Bottleneck** (from MobileNetV2) expands channels, applies depthwise conv, then contracts:
```
input (C) → expand (C×exp_ratio) → depthwise 3×3 → project (C)
```
This gives the depthwise conv a richer feature space to operate in.

### MobileNetV4 and Universal Inverted Bottleneck (UIB)

MobileNetV4 generalizes the inverted bottleneck into a **Universal Inverted Bottleneck** that selects its structure via three depthwise kernel positions:

```
input → [optional DW_start] → expand PW → [optional DW_mid] → project PW → [optional DW_end] → output
```

Each `DW_*` is either absent (kernel size 0) or present (kernel size 3 or 5). This gives four named variants:

| Variant       | dw_start | dw_mid | dw_end | Effect |
|---------------|----------|--------|--------|--------|
| Inverted Bottleneck (IB) | 0 | 3 | 0 | Classic MBConv |
| ConvNext-like | 3 | 0 | 0 | Large-kernel at start, like ConvNeXt |
| ExtraDW       | 3 | 3 | 0 | Two depthwise ops, richer mixing |
| FFN / Pointwise | 0 | 0 | 0 | Pure 1×1, no spatial mixing |

The architecture search mutates which UIB variant appears at each FPN level and what `exp_ratio` (expansion factor, 2.0–8.0) is used. This controls the capacity/efficiency tradeoff of each refinement block.

**Backbone channels in this project:**
```
C3: 64 channels  (stride 8)
C4: 96 channels  (stride 16)
C5: 960 channels (stride 32)
```
The jump from 96 to 960 at C5 is characteristic of MobileNetV4-small — C5 uses a wide "last stage" for strong semantics. The FPN neck must project C5 down to `NECK_CHANNELS` (e.g. 96) via a lateral 1×1 conv, so the neck carries significant responsibility for reconciling these mismatched scales.

---

## Part 4 — Neural Architecture Search (NAS)

### What NAS is

Manual network design is expensive: an engineer proposes an architecture, trains it, evaluates, and iterates. NAS automates the proposal step, using some search algorithm to explore the space of possible architectures.

This project uses **program-level evolutionary search** (ShinkaEvolve): each candidate is a Python file that defines the detector. The search algorithm mutates the file (via LLM-generated diffs), trains it, scores it, and keeps the best in an archive.

### How ShinkaEvolve works

1. **Archive** — a population of (program, score) pairs, initialized with `initial_compress.py`
2. **Proposal** — LLM reads elite archive members, proposes a mutation (diff or full rewrite) guided by the `task_sys_msg` prompt
3. **Evaluation** — the mutated program runs `run_experiment()`, which trains the detector and calls `evaluate_model()` to get a score
4. **Selection** — if the score is competitive, the new program joins the archive (crowding/Pareto-based selection to maintain diversity)
5. **Repeat** — for `num_generations` iterations

### The search space

Only code inside the `# EVOLVE-BLOCK-START / END` markers is mutable. This currently covers:
- `NECK_CHANNELS` — FPN output width per level
- `UIB_CONFIGS` — per-level UIB variant and expansion ratio
- `HEAD_CHANNELS`, `HEAD_STACKS` — FCOS head width and depth
- FPN topology (top-down only vs PAN bidirectional)

Explicitly locked outside the block: `LR`, `EPOCHS`, `BATCH_SIZE`, `WARMUP_EPOCHS`. If these were mutable, a candidate could score better simply by training longer — making it impossible to tell whether the architecture or the budget won.

### Composite search score

```
score = mAP50 / (0.5 × model_size_mb + 0.5 × cpu_latency_ms / 100)
```

This balances three deployment concerns:
- **mAP50** — detection quality (numerator)
- **model_size_mb** — ONNX file size (proxy for memory footprint)
- **cpu_latency_ms / 100** — normalized CPU inference time (100ms = 1.0 baseline for ARM Cortex-A53)

A model that is half the size but twice as slow scores the same as the original — the search is indifferent between size and latency improvements.

### Why cheap search needs a reranking stage

Phase 1 uses 4 epochs, frozen backbone, 5K images. This is fast (cheap to evaluate many candidates) but biased: some architectures learn faster under frozen-backbone short-horizon training but aren't actually better after proper training. The search finds "what converges fastest under the proxy" not "what is best for deployment."

`scripts/rerank_finalists.py` addresses this by retraining the top-K Phase 1 candidates with 10 epochs, `FROZEN_STAGES=2`, and 2 seeds. This removes the fast-learner bias and gives a more reliable ranking before committing to Phase 2 compression.

---

## Part 5 — Model Compression

### Why compress

MobileNetV4-small is already lightweight, but "lightweight for ImageNet" is not the same as "fast on a Raspberry Pi." A typical forward pass at 320×320 on an ARM Cortex-A53 takes tens to hundreds of milliseconds depending on the model. Quantization and pruning can cut this by 2–4×.

### Static post-training quantization (PTQ)

Quantization maps floating-point weights and activations to integers (typically INT8):
```
x_int = round(x_float / scale) + zero_point
```
where `scale` and `zero_point` are calibrated per-tensor or per-channel.

**Why calibration data is needed:** the scale/zero-point must cover the actual range of activations for the model's inputs. You pass a small representative dataset (50–100 images) through the model and record activation statistics, then compute the quantization parameters. Without calibration, scale values are guesses and accuracy drops.

**Why dynamic quantization doesn't work for CNNs:** dynamic quantization quantizes weights statically but determines activation scale at runtime, which avoids calibration — but it only applies to `nn.Linear` and `nn.LSTM`. `nn.Conv2d` is untouched. For a CNN-heavy detector, dynamic quantization has effectively no effect on inference speed.

This project uses `onnxruntime.quantization.quantize_static` which actually quantizes Conv2d operations and produces a genuinely INT8 ONNX model.

### ONNX export

ONNX (Open Neural Network Exchange) is a hardware-independent model format. The workflow:
1. Train in PyTorch
2. Export to ONNX: `torch.onnx.export(model, dummy_input, path)`
3. Run with ONNX Runtime: works on CPU, GPU, ARM, without PyTorch installed

ONNX Runtime with `CPUExecutionProvider` is used for latency measurement and evaluation in this project. The quantized ONNX is the final deployment artifact.

### Latency vs model size

File size (MB) is only a rough proxy for inference speed. Two models with identical file sizes can have very different latencies due to:

- **Operator mix**: depthwise convolutions are memory-bandwidth-bound; pointwise convolutions are compute-bound. Their relative speed differs between x86 and ARM.
- **Feature map sizes**: a wide but shallow network touches more memory than a narrow but deep one.
- **Memory access patterns**: concat operations (PAN) access non-contiguous memory; add operations (residual) are cheaper.
- **Depthwise conv on ARM**: ARM NEON has optimized depthwise conv kernels; the speedup over standard conv is more pronounced than on x86.

This is why the score formula includes `cpu_latency_ms` directly rather than relying on `model_size_mb` alone.

---

## Part 6 — Experimental Methodology

### Proxy validity and rank correlation

A "proxy" is any cheap stand-in for an expensive evaluation. In this pipeline:
- 5K subset proxies for the full 118K training set
- 4-epoch training proxies for a full training run
- host CPU latency proxies for Pi latency

The key question is not "is the proxy accurate in absolute terms" but "does it preserve the **rank order** of candidates?" A proxy can be systematically pessimistic (everyone scores low) and still be valid for search, as long as the best candidates consistently rank higher.

`scripts/validate_subset_corr.py` tests this with Spearman rank correlation (ρ): it retrains top-K candidates at 5K, 10K, and 20K subsets and checks whether the orderings agree. ρ ≥ 0.8 is the threshold for trusting the 5K proxy; below that, Phase 1 should use a larger subset.

Spearman ρ is the correlation between rank vectors (not raw values) — it is robust to monotone nonlinear relationships and outliers, making it appropriate here since mAP values are noisy.

### Multi-seed evaluation

Neural network training has random components: weight initialization, data ordering, augmentation sampling. A single training run produces one sample from a distribution of outcomes. Two candidates whose single-run scores differ by 0.002 mAP may have overlapping outcome distributions — the "winner" may just have gotten a lucky seed.

Multi-seed evaluation (N=2–3 seeds) gives:
- **mean ± std** instead of a point estimate
- a **promotion gate**: only promote a candidate if its mean advantage exceeds the combined std of winner and runner-up

In `scripts/rerank_finalists.py`, the promotion gate prints a warning if the margin is within noise, prompting the user to run more seeds before committing.

### Letterboxing vs square resize

In classification, resizing an image to 224×224 by squishing it is harmless — the class label doesn't change if a cat looks slightly wide. In detection, the bounding box coordinates must match the resized image geometry.

**Square resize** applies different scale factors to width and height:
```
scale_x = target / orig_w
scale_y = target / orig_h   (≠ scale_x if aspect ratio ≠ 1)
```
This distorts object shapes: tall narrow persons become squat, which harms detection of elongated objects and makes small-person detection harder.

**Letterboxing** uses a single uniform scale:
```
scale = min(target / orig_w, target / orig_h)
```
then pads the shorter dimension with gray (128) to reach the target size. Bounding boxes only need a single scale and a pad offset correction:
```
x1_new = x1 * scale + pad_left
y1_new = y1 * scale + pad_top
```
This is what `tinydet/data.py` implements.

### Augmentation for detection

Classification augmentation (random crop, color jitter, flip) needs only to preserve the class label, which is usually robust. Detection augmentation must also transform the bounding boxes consistently with the image transformation.

**Horizontal flip:** straightforward — boxes flip symmetrically. `x1_new = W - x2_old`, `x2_new = W - x1_old`.

**Scale jitter:** resize to `[0.8, 1.2] × target_size`, then crop (if larger) or pad (if smaller) back to `target_size`. Each box coordinate is scaled by `jitter` then shifted by the crop/pad offset. Filtered for validity (`x2 > x1`).

**Color jitter (brightness/contrast):** pixel-level — no box transformation needed. Helps the detector generalize across lighting conditions.

Mosaic and copy-paste (used in YOLOv5+) are much stronger augmentation strategies but weren't used here to keep the recipe simple. They mix multiple images into one, dramatically increasing training diversity at the cost of complexity.
