"""COCO dataset and dataloader utilities for autoresearch-tinydet."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import CocoDetection

from tinydet.io import CACHE_DIR, COCO_ROOT, SUBSET_INDEX_PATH


_COCO_PERSON_CATEGORY_ID = 1
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class CocoPersonDataset(CocoDetection):
    """COCO 2017 – person annotations only, with bbox rescaling.

    Returns (img_tensor, target_dict) where target_dict has keys:
      boxes:    FloatTensor[N, 4]  — xyxy, rescaled to img_size
      labels:   LongTensor[N]      — always 1 (person)
      image_id: LongTensor[1]
    """

    def __init__(self, root: str | Path, annFile: str | Path, img_size: int, split: str = "val"):
        super().__init__(str(root), str(annFile))
        self.img_size = img_size
        self.split = split

    def __getitem__(self, index: int):  # type: ignore[override]
        img, annotations = super().__getitem__(index)

        orig_w, orig_h = img.size
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        img = TF.resize(img, [self.img_size, self.img_size])

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in annotations:
            if ann.get("category_id") != _COCO_PERSON_CATEGORY_ID:
                continue
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, min(x * scale_x, self.img_size))
            y1 = max(0.0, min(y * scale_y, self.img_size))
            x2 = max(0.0, min((x + w) * scale_x, self.img_size))
            y2 = max(0.0, min((y + h) * scale_y, self.img_size))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(_COCO_PERSON_CATEGORY_ID)

        if self.split == "train" and random.random() < 0.5:
            img = TF.hflip(img)
            W = float(self.img_size)
            boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]

        img_tensor = TF.normalize(TF.to_tensor(img), mean=_MEAN, std=_STD)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([self.ids[index]], dtype=torch.long),
        }
        return img_tensor, target


def collate_fn(batch: List[Tuple]) -> Tuple[List, List]:
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def make_dataloader(
    split: str,
    batch_size: int,
    img_size: int,
    subset_size: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader for COCO 2017 person detection.

    split:       'train' or 'val'
    batch_size:  images per batch
    img_size:    square canvas size (e.g. 320)
    subset_size: number of training images; None uses the default 5K index.
                 Any other value selects or builds subset_<N>_index.json.

    Train uses a stratified subset; val uses the full COCO 2017 val set.
    DataLoader settings (num_workers, pin_memory, persistent_workers) are
    read from env vars set by hw_config.py.
    """
    if split == "train":
        img_dir = COCO_ROOT / "train2017"
        ann_file = COCO_ROOT / "annotations" / "instances_train2017.json"
    elif split == "val":
        img_dir = COCO_ROOT / "val2017"
        ann_file = COCO_ROOT / "annotations" / "instances_val2017.json"
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    dataset = CocoPersonDataset(root=img_dir, annFile=ann_file, img_size=img_size, split=split)

    if split == "train":
        if subset_size is None:
            index_path = SUBSET_INDEX_PATH
        else:
            index_path = CACHE_DIR / f"subset_{subset_size}_index.json"

        if not index_path.exists():
            if subset_size is None:
                raise FileNotFoundError(
                    f"Subset index not found at {index_path}. Run prepare.py first."
                )
            import prepare  # late import — prepare.py imports train_utils
            print(f"Building subset index for {subset_size} images → {index_path}")
            prepare.build_subset_index(CACHE_DIR, index_path, subset_size=subset_size)

        with open(index_path) as f:
            data = json.load(f)
        subset_indices = data["indices"] if isinstance(data, dict) else data
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    num_workers = int(os.environ.get("DATALOADER_WORKERS", "0"))
    pin_mem = os.environ.get("DATALOADER_PIN_MEMORY", "0") == "1"
    persistent = os.environ.get("DATALOADER_PERSISTENT_WORKERS", "0") == "1" and num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )
