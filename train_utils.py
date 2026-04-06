"""
train_utils.py — Fixed utility module for autoresearch-tinydet.

Provides:
  - CACHE_DIR / SUBSET_INDEX_PATH constants
  - load_backbone()       – MobileNetV4-Conv-S feature extractor
  - CocoPersonDataset     – COCO 2017, person-only, bbox-rescaled
  - make_dataloader()     – train (5 K subset) / val (full)
  - export_onnx()
  - save_checkpoint() / load_checkpoint()
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import CocoDetection

import timm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path(
    os.environ.get("AUTORESEARCH_CACHE", Path.home() / ".cache" / "autoresearch-tinydet")
)
SUBSET_INDEX_PATH: Path = CACHE_DIR / "subset_5k_index.json"

# COCO dataset root – override via env var if needed
COCO_ROOT: Path = Path(os.environ.get("COCO_ROOT", CACHE_DIR / "coco"))

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def load_backbone(frozen_stages: int = 4) -> Tuple[nn.Module, List[int]]:
    """Return (backbone, feature_channels) for MobileNetV4-Conv-S.

    feature_channels are the C3/C4/C5 output channel counts:
      - C3 (stride 8):  64 channels
      - C4 (stride 16): 96 channels
      - C5 (stride 32): 960 channels

    Args:
        frozen_stages: Number of backbone stages (children) to freeze.
                       Counted from the top-level children:
                         0 = conv_stem, 1 = bn1, 2 = act1, 3 = blocks[0],
                         4 = blocks[0..1], …
                       In practice, frozen_stages=4 freezes conv_stem +
                       bn1 + act1 + blocks[0] (the first block group).
                       Counted as the first N *top-level named children*.
    """
    backbone = timm.create_model(
        "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4),
    )

    feature_channels: List[int] = backbone.feature_info.channels()

    # Freeze the first `frozen_stages` top-level children
    named_children = list(backbone.named_children())
    for i, (name, module) in enumerate(named_children):
        if i >= frozen_stages:
            break
        for param in module.parameters():
            param.requires_grad = False

    # If frozen_stages covers all of conv_stem/bn1/act1 plus some block
    # sub-stages inside `blocks`, handle that gracefully: the loop above
    # already freezes whole top-level children; callers that want finer
    # granularity should pass a larger frozen_stages value covering the
    # desired number of top-level children (conv_stem, bn1, act1, blocks).
    # For the common case of frozen_stages=4, we freeze:
    #   child 0 = conv_stem, child 1 = bn1, child 2 = act1, child 3 = blocks
    # i.e. the entire backbone is frozen except nothing when stages < 4.

    return backbone, feature_channels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_COCO_PERSON_CATEGORY_ID = 1

# ImageNet normalization
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class CocoPersonDataset(CocoDetection):
    """COCO 2017 – person annotations only, with bbox rescaling.

    Bounding boxes are rescaled from the original image dimensions to
    match the resized ``img_size × img_size`` canvas.  All boxes are
    clipped to ``[0, img_size]``.

    Returns ``(img_tensor, target_dict)`` where::

        target_dict = {
            "boxes":    FloatTensor[N, 4]   # xyxy, rescaled
            "labels":   LongTensor[N]       # always 1 (person)
            "image_id": LongTensor[1]
        }
    """

    def __init__(self, root: str | Path, annFile: str | Path, img_size: int, split: str = "val"):
        super().__init__(str(root), str(annFile))
        self.img_size = img_size
        self.split = split  # 'train' | 'val'

    def __getitem__(self, index: int):  # type: ignore[override]
        img, annotations = super().__getitem__(index)

        orig_w, orig_h = img.size  # PIL image: (W, H)
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        # Resize image
        img = TF.resize(img, [self.img_size, self.img_size])

        # Filter to person annotations with valid bboxes
        boxes = []
        labels = []
        for ann in annotations:
            if ann.get("category_id") != _COCO_PERSON_CATEGORY_ID:
                continue
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]  # COCO xywh
            if w <= 0 or h <= 0:
                continue
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            # Clip
            x1 = max(0.0, min(x1, self.img_size))
            y1 = max(0.0, min(y1, self.img_size))
            x2 = max(0.0, min(x2, self.img_size))
            y2 = max(0.0, min(y2, self.img_size))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(_COCO_PERSON_CATEGORY_ID)

        # Random horizontal flip (train only)
        if self.split == "train" and random.random() < 0.5:
            img = TF.hflip(img)
            W = float(self.img_size)
            boxes = [
                [W - b[2], b[1], W - b[0], b[3]]
                for b in boxes
            ]

        # Convert image to tensor + normalize
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=_MEAN, std=_STD)

        # Build target tensors
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        image_id = self.ids[index]
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.long),
        }

        return img_tensor, target


def _collate_fn(batch):
    """Returns (list_of_tensors, list_of_target_dicts)."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def make_dataloader(split: str, batch_size: int, img_size: int) -> DataLoader:
    """Create a DataLoader for COCO 2017 person detection.

    Args:
        split:      'train' or 'val'.
        batch_size: Images per batch.
        img_size:   Square canvas size (e.g. 320).

    Train:  stratified 5 K subset defined by SUBSET_INDEX_PATH.
    Val:    full COCO 2017 val set (person annotations).
    """
    if split == "train":
        img_dir = COCO_ROOT / "images" / "train2017"
        ann_file = COCO_ROOT / "annotations" / "instances_train2017.json"
    elif split == "val":
        img_dir = COCO_ROOT / "images" / "val2017"
        ann_file = COCO_ROOT / "annotations" / "instances_val2017.json"
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    dataset = CocoPersonDataset(
        root=img_dir,
        annFile=ann_file,
        img_size=img_size,
        split=split,
    )

    if split == "train":
        with open(SUBSET_INDEX_PATH) as f:
            subset_indices = json.load(f)
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    return loader


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, path: str, img_size: int = 320) -> str:
    """Export ``model`` to ONNX (opset 17) and return the file path.

    Args:
        model:    PyTorch module to export.
        path:     Destination file path (e.g. "model.onnx").
        img_size: Spatial resolution for the dummy input.

    Returns:
        The resolved path string.
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=17,
        input_names=["images"],
        dynamic_axes={"images": {0: "batch"}},
        do_constant_folding=True,
    )
    return str(Path(path).resolve())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    """Save a state dict to *path* (creates parent dirs if needed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(out))


def load_checkpoint(path: str) -> dict:
    """Load and return a checkpoint saved with :func:`save_checkpoint`."""
    return torch.load(path, map_location="cpu")
