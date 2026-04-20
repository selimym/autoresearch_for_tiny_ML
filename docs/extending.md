# Extending the Pipeline

## Knowledge Distillation

To add distillation in Phase 2-3, add these constants to `compress.py`:

```python
DISTILL_ALPHA = 0.5   # weight of distillation loss vs. task loss
DISTILL_TEMP  = 4.0   # temperature for softening teacher logits
TEACHER_PATH  = "checkpoints/yolov8n_teacher.pt"  # float teacher
```

Then in the training loop:
```python
with torch.no_grad():
    teacher_cls, _, _ = teacher(imgs_t)
# soft targets
for student_cls, teacher_cls_level in zip(cls_preds, teacher_cls):
    soft_student = F.log_softmax(student_cls / DISTILL_TEMP, dim=1)
    soft_teacher = F.softmax(teacher_cls_level / DISTILL_TEMP, dim=1)
    kd_loss += F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (DISTILL_TEMP ** 2)
loss = (1 - DISTILL_ALPHA) * task_loss + DISTILL_ALPHA * kd_loss
```

A strong teacher (YOLOv8s pretrained on COCO person) can recover 2–4pp mAP50 after quantization or pruning.

## Wake Vision Supplementary Benchmark

Wake Vision is a person/no-person binary classification dataset optimized for MCU deployment. To run a linear probe evaluation:

```python
# After exporting ONNX, extract P3 feature embeddings on Wake Vision val set
# Then fit a LogisticRegression on the embeddings
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(train_embeddings, train_labels)
acc = clf.score(val_embeddings, val_labels)
print(f"Wake Vision linear probe accuracy: {acc:.4f}")
```

Add this as a secondary metric in `evaluate_core.py` if Wake Vision becomes a target deployment benchmark.

## OpenEvolve Integration

`evaluate.py` already exposes an OpenEvolve-compatible interface:
```python
from evaluate import evaluate
metrics = evaluate("checkpoints/model.onnx")  # -> dict[str, float]
```

To switch Phase 1 from ShinkaEvolve to OpenEvolve:
1. Replace `run_phase1.py` with an OpenEvolve runner
2. Use `evaluate.py` as the evaluator (already compatible)
3. Keep `initial_compress.py` unchanged (EVOLVE-BLOCK markers are compatible)

## MotionFlow Integration

MotionFlow is a separate fork (not in this repo). After Phase 3 produces a champion model, integrate by:
1. Export Phase 3 ONNX to `checkpoints/phase3_champion.onnx`
2. In the MotionFlow fork, load this ONNX as the person detector backbone
3. MotionFlow adds temporal tracking on top of the per-frame detections

See the MotionFlow repo documentation for the expected ONNX input/output format.
