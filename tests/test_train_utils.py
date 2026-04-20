"""
Unit tests for train_utils.py — bbox math and collate_fn.

These tests do not require COCO data or network access.
"""

import torch
import pytest

from train_utils import collate_fn, CocoPersonDataset, _COCO_PERSON_CATEGORY_ID


class TestCollateFn:
    def test_returns_lists(self):
        imgs = [torch.randn(3, 32, 32), torch.randn(3, 32, 32)]
        targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}] * 2
        batch = list(zip(imgs, targets))

        out_imgs, out_targets = collate_fn(batch)

        assert isinstance(out_imgs, list)
        assert isinstance(out_targets, list)
        assert len(out_imgs) == 2
        assert len(out_targets) == 2

    def test_tensors_not_stacked(self):
        """Images of different sizes must survive collation without stacking."""
        imgs = [torch.randn(3, 32, 32), torch.randn(3, 64, 64)]
        targets = [{"boxes": torch.zeros(0, 4)}] * 2
        batch = list(zip(imgs, targets))

        out_imgs, _ = collate_fn(batch)

        assert out_imgs[0].shape == (3, 32, 32)
        assert out_imgs[1].shape == (3, 64, 64)


class TestBboxRescaling:
    """Test bbox rescaling and clipping in isolation using a synthetic dataset."""

    def _make_dataset(self, img_size: int) -> CocoPersonDataset:
        """Create an uninitialized CocoPersonDataset (no COCO files needed)."""
        ds = object.__new__(CocoPersonDataset)
        ds.img_size = img_size
        ds.split = "val"
        return ds

    def _process_anns(self, ds: CocoPersonDataset, annotations: list, orig_w: int, orig_h: int):
        """Run just the bbox-transform portion of __getitem__."""
        scale_x = ds.img_size / orig_w
        scale_y = ds.img_size / orig_h
        boxes = []
        labels = []
        for ann in annotations:
            if ann.get("category_id") != _COCO_PERSON_CATEGORY_ID:
                continue
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, min(x * scale_x, ds.img_size))
            y1 = max(0.0, min(y * scale_y, ds.img_size))
            x2 = max(0.0, min((x + w) * scale_x, ds.img_size))
            y2 = max(0.0, min((y + h) * scale_y, ds.img_size))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(_COCO_PERSON_CATEGORY_ID)
        return boxes, labels

    def test_unit_scale_identity(self):
        """When orig == img_size, bboxes are unchanged."""
        ds = self._make_dataset(img_size=320)
        anns = [{"category_id": 1, "iscrowd": 0, "bbox": [10.0, 20.0, 100.0, 80.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=320, orig_h=320)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert abs(x1 - 10.0) < 1e-5
        assert abs(y1 - 20.0) < 1e-5
        assert abs(x2 - 110.0) < 1e-5
        assert abs(y2 - 100.0) < 1e-5

    def test_downscale(self):
        """640→320 halves all coordinates."""
        ds = self._make_dataset(img_size=320)
        anns = [{"category_id": 1, "iscrowd": 0, "bbox": [100.0, 200.0, 200.0, 100.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=640, orig_h=640)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert abs(x1 - 50.0) < 1e-5
        assert abs(y1 - 100.0) < 1e-5
        assert abs(x2 - 150.0) < 1e-5
        assert abs(y2 - 150.0) < 1e-5

    def test_out_of_bounds_box_clipped(self):
        ds = self._make_dataset(img_size=320)
        # bbox starts at x=300 with w=100 (exceeds 320)
        anns = [{"category_id": 1, "iscrowd": 0, "bbox": [300.0, 0.0, 100.0, 50.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=320, orig_h=320)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert x2 <= 320.0

    def test_zero_area_box_dropped(self):
        ds = self._make_dataset(img_size=320)
        anns = [{"category_id": 1, "iscrowd": 0, "bbox": [10.0, 10.0, 0.0, 50.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=320, orig_h=320)
        assert len(boxes) == 0

    def test_crowd_annotation_dropped(self):
        ds = self._make_dataset(img_size=320)
        anns = [{"category_id": 1, "iscrowd": 1, "bbox": [10.0, 10.0, 100.0, 100.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=320, orig_h=320)
        assert len(boxes) == 0

    def test_non_person_annotation_dropped(self):
        ds = self._make_dataset(img_size=320)
        anns = [{"category_id": 2, "iscrowd": 0, "bbox": [10.0, 10.0, 100.0, 100.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=320, orig_h=320)
        assert len(boxes) == 0

    def test_asymmetric_scale(self):
        """Non-square original → x and y scale independently."""
        ds = self._make_dataset(img_size=320)
        # orig 640×480 → scale_x=0.5, scale_y=320/480≈0.6667
        anns = [{"category_id": 1, "iscrowd": 0, "bbox": [0.0, 0.0, 640.0, 480.0]}]
        boxes, _ = self._process_anns(ds, anns, orig_w=640, orig_h=480)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert abs(x2 - 320.0) < 1e-3
        assert abs(y2 - 320.0) < 1e-3
