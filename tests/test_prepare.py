"""
Unit tests for prepare.py — stratified subset building.

These tests use a synthetic COCO-format annotation file so no real COCO
download is required.
"""

import json
from pathlib import Path

import pytest

from prepare import build_subset_index


def _make_coco_ann(tmp_path: Path, images: list[dict], annotations: list[dict]) -> Path:
    """Write a minimal COCO instances JSON to a temp annotations dir."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    payload = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "images": images,
        "annotations": annotations,
    }
    ann_file = ann_dir / "instances_train2017.json"
    ann_file.write_text(json.dumps(payload))
    return tmp_path


def _image(img_id: int) -> dict:
    return {"id": img_id, "file_name": f"{img_id:012d}.jpg", "width": 640, "height": 480}


def _ann(ann_id: int, img_id: int, bbox: list[float], iscrowd: int = 0) -> dict:
    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }


class TestBuildSubsetIndex:
    def test_returns_exact_subset_size(self, tmp_path):
        """subset_size images are returned (or all available if fewer)."""
        images = [_image(i) for i in range(1, 101)]
        anns = [_ann(i, i, [10.0, 10.0, 100.0, 100.0]) for i in range(1, 101)]
        cache_dir = _make_coco_ann(tmp_path, images, anns)
        out = tmp_path / "subset.json"

        result = build_subset_index(cache_dir, out, subset_size=50)

        assert len(result["indices"]) == 50
        assert len(result["img_ids"]) == 50

    def test_output_file_written(self, tmp_path):
        images = [_image(i) for i in range(1, 21)]
        anns = [_ann(i, i, [5.0, 5.0, 50.0, 50.0]) for i in range(1, 21)]
        cache_dir = _make_coco_ann(tmp_path, images, anns)
        out = tmp_path / "subset.json"

        build_subset_index(cache_dir, out, subset_size=10)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "indices" in data
        assert "img_ids" in data
        assert "stratum_counts" in data

    def test_reproducible_with_same_seed(self, tmp_path):
        """Same input always produces the same sample (seed=42 fixed in build_subset_index)."""
        images = [_image(i) for i in range(1, 201)]
        anns = [_ann(i, i, [10.0, 10.0, float(30 + i % 50), float(30 + i % 50)]) for i in range(1, 201)]
        cache_dir = _make_coco_ann(tmp_path, images, anns)

        out1 = tmp_path / "subset1.json"
        out2 = tmp_path / "subset2.json"
        r1 = build_subset_index(cache_dir, out1, subset_size=50)
        r2 = build_subset_index(cache_dir, out2, subset_size=50)

        assert r1["indices"] == r2["indices"]

    def test_indices_sorted(self, tmp_path):
        images = [_image(i) for i in range(1, 51)]
        anns = [_ann(i, i, [1.0, 1.0, 200.0, 200.0]) for i in range(1, 51)]
        cache_dir = _make_coco_ann(tmp_path, images, anns)
        out = tmp_path / "subset.json"

        result = build_subset_index(cache_dir, out, subset_size=30)

        assert result["indices"] == sorted(result["indices"])

    def test_missing_annotation_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_subset_index(tmp_path, tmp_path / "out.json")

    def test_stratum_coverage(self, tmp_path):
        """Small/medium/large area buckets all appear when data covers all sizes."""
        images = [_image(i) for i in range(1, 61)]
        anns = []
        ann_id = 1
        # small: area < 1024 → e.g. 30×30=900
        for i in range(1, 21):
            anns.append(_ann(ann_id, i, [0.0, 0.0, 30.0, 30.0]))
            ann_id += 1
        # medium: 1024 ≤ area < 9216 → e.g. 50×50=2500
        for i in range(21, 41):
            anns.append(_ann(ann_id, i, [0.0, 0.0, 50.0, 50.0]))
            ann_id += 1
        # large: area ≥ 9216 → e.g. 100×100=10000
        for i in range(41, 61):
            anns.append(_ann(ann_id, i, [0.0, 0.0, 100.0, 100.0]))
            ann_id += 1

        cache_dir = _make_coco_ann(tmp_path, images, anns)
        out = tmp_path / "subset.json"
        result = build_subset_index(cache_dir, out, subset_size=30)

        strata = result["stratum_counts"]
        area_prefixes = {k.split("/")[0] for k in strata}
        assert "small" in area_prefixes
        assert "medium" in area_prefixes
        assert "large" in area_prefixes

    def test_subset_size_larger_than_data(self, tmp_path):
        """When subset_size > available images, all images are returned."""
        images = [_image(i) for i in range(1, 11)]
        anns = [_ann(i, i, [10.0, 10.0, 50.0, 50.0]) for i in range(1, 11)]
        cache_dir = _make_coco_ann(tmp_path, images, anns)
        out = tmp_path / "subset.json"

        result = build_subset_index(cache_dir, out, subset_size=1000)

        assert len(result["indices"]) == 10
