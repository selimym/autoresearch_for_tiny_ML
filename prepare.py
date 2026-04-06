"""
One-time data preparation for autoresearch-tinydet.

Downloads COCO 2017 detection data and builds a stratified 5K training subset.

Usage:
    uv run prepare.py                  # full setup

Data and subset index are stored in ~/.cache/autoresearch-tinydet/.
"""

import json
import os
import random
import urllib.request
from pathlib import Path

from train_utils import CACHE_DIR, SUBSET_INDEX_PATH

try:
    import pycocotools.coco
    HAS_COCO_TOOLS = True
except ImportError:
    HAS_COCO_TOOLS = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ---------------------------------------------------------------------------
# COCO URLs
# ---------------------------------------------------------------------------

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def download_coco(cache_dir: Path) -> None:
    """Download COCO 2017 val images and annotations.

    Idempotent: skips download if files already exist.
    """
    zips_dir = cache_dir / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    # Download val2017.zip
    val_zip = zips_dir / "val2017.zip"
    val_extract = cache_dir / "val2017"
    if not val_extract.exists():
        if not val_zip.exists():
            print(f"Downloading {COCO_VAL_URL}...")
            try:
                urllib.request.urlretrieve(COCO_VAL_URL, str(val_zip))
                print(f"  Downloaded to {val_zip}")
            except Exception as e:
                print(f"  Error downloading val2017: {e}")
                return

        print(f"Extracting {val_zip}...")
        import zipfile
        with zipfile.ZipFile(val_zip, "r") as z:
            z.extractall(cache_dir)
        print(f"  Extracted to {val_extract}")
    else:
        print(f"Val images already exist at {val_extract}")

    # Download annotations_trainval2017.zip
    ann_zip = zips_dir / "annotations_trainval2017.zip"
    ann_extract = cache_dir / "annotations"
    if not ann_extract.exists():
        if not ann_zip.exists():
            print(f"Downloading {COCO_ANN_URL}...")
            try:
                urllib.request.urlretrieve(COCO_ANN_URL, str(ann_zip))
                print(f"  Downloaded to {ann_zip}")
            except Exception as e:
                print(f"  Error downloading annotations: {e}")
                return

        print(f"Extracting {ann_zip}...")
        import zipfile
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(cache_dir)
        print(f"  Extracted to {ann_extract}")
    else:
        print(f"Annotations already exist at {ann_extract}")


# ---------------------------------------------------------------------------
# Subset building
# ---------------------------------------------------------------------------

def build_subset_index(cache_dir: Path, output_path: Path, subset_size: int = 5000) -> dict:
    """Build a stratified 5K subset index from COCO 2017 train annotations.

    Stratification key: (area_bucket, count_bucket)
    - area_bucket: "small" if max_bbox_area < 1024, "medium" if < 9216, "large" otherwise
    - count_bucket: "1" if count==1, "2-4" if 2-4, "5+" if 5+

    Returns a dict:
    {
        "indices": [list of 5000 integer indices],
        "img_ids": [list of 5000 COCO image IDs],
        "stratum_counts": {stratum_key: count, ...}
    }

    The indices are into the sorted list of person image IDs.
    """
    ann_file = cache_dir / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    print(f"Loading COCO annotations from {ann_file}...")
    if not HAS_COCO_TOOLS:
        raise ImportError(
            "pycocotools not available. Install with: pip install pycocotools"
        )

    from pycocotools.coco import COCO

    coco = COCO(str(ann_file))

    # Get person-only images (category 1)
    person_cat_id = 1
    person_img_ids = sorted(coco.getImgIds(catIds=[person_cat_id]))
    print(f"Found {len(person_img_ids)} images with person annotations")

    # Build stratified sampler
    strata = {}  # stratum_key -> list of indices

    for idx, img_id in enumerate(person_img_ids):
        anns = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id])
        if len(anns) == 0:
            continue

        ann_dicts = coco.loadAnns(anns)

        # Compute max bbox area
        max_area = 0.0
        for ann in ann_dicts:
            x, y, w, h = ann["bbox"]
            area = w * h
            max_area = max(max_area, area)

        # Area bucket
        if max_area < 1024:
            area_bucket = "small"
        elif max_area < 9216:
            area_bucket = "medium"
        else:
            area_bucket = "large"

        # Count bucket
        count = len(ann_dicts)
        if count == 1:
            count_bucket = "1"
        elif count <= 4:
            count_bucket = "2-4"
        else:
            count_bucket = "5+"

        stratum_key = f"{area_bucket}/{count_bucket}"
        if stratum_key not in strata:
            strata[stratum_key] = []
        strata[stratum_key].append(idx)

    # Stratified sample
    rng = random.Random(42)
    sampled_indices = []
    stratum_counts = {}

    # Total needed per stratum proportional to stratum size
    total_indices = sum(len(v) for v in strata.values())

    for stratum_key in sorted(strata.keys()):
        indices_in_stratum = strata[stratum_key]
        proportion = len(indices_in_stratum) / total_indices
        target_count = int(round(subset_size * proportion))

        # Sample up to target_count without replacement
        sampled = rng.sample(indices_in_stratum, min(target_count, len(indices_in_stratum)))
        sampled_indices.extend(sampled)
        stratum_counts[stratum_key] = len(sampled)

    # Ensure exactly subset_size
    if len(sampled_indices) < subset_size:
        all_indices = list(range(len(person_img_ids)))
        remaining = [i for i in all_indices if i not in sampled_indices]
        needed = subset_size - len(sampled_indices)
        sampled_indices.extend(rng.sample(remaining, min(needed, len(remaining))))
    elif len(sampled_indices) > subset_size:
        sampled_indices = rng.sample(sampled_indices, subset_size)

    # Sort for reproducibility
    sampled_indices.sort()
    sampled_img_ids = [person_img_ids[idx] for idx in sampled_indices]

    print(f"Stratified sample: {len(sampled_indices)} images")
    for stratum_key in sorted(stratum_counts.keys()):
        print(f"  {stratum_key}: {stratum_counts[stratum_key]}")

    result = {
        "indices": sampled_indices,
        "img_ids": sampled_img_ids,
        "stratum_counts": stratum_counts,
    }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved subset index to {output_path}")

    return result


# ---------------------------------------------------------------------------
# Backbone verification
# ---------------------------------------------------------------------------

def verify_backbone() -> list:
    """Load backbone and verify feature channels.

    Returns feature_channels list (e.g. [64, 96, 960]).
    """
    if not HAS_TIMM:
        raise ImportError("timm not available. Install with: pip install timm")

    print("Loading backbone: mobilenetv4_conv_small.e2400_r224_in1k...")
    backbone = timm.create_model(
        "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4),
    )

    feature_channels = backbone.feature_info.channels()
    print(f"  Feature channels: {feature_channels}")

    return feature_channels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run full data preparation."""
    print("=" * 70)
    print("autoresearch-tinydet data preparation")
    print("=" * 70)
    print()

    # Step 1: Download COCO
    print("Step 1: Download COCO 2017 data")
    print("-" * 70)
    download_coco(CACHE_DIR)
    print()

    # Step 2: Build subset index
    print("Step 2: Build stratified 5K subset index")
    print("-" * 70)
    if not SUBSET_INDEX_PATH.exists():
        subset_info = build_subset_index(CACHE_DIR, SUBSET_INDEX_PATH)
    else:
        print(f"Subset index already exists at {SUBSET_INDEX_PATH}")
        with open(SUBSET_INDEX_PATH) as f:
            subset_info = json.load(f)
    print()

    # Step 3: Verify backbone
    print("Step 3: Verify backbone")
    print("-" * 70)
    feature_channels = verify_backbone()
    print()

    # Print summary
    print("=" * 70)
    print("=== autoresearch-tinydet data ready ===")
    print("=" * 70)
    print(f"CACHE_DIR: {CACHE_DIR}")

    val_dir = CACHE_DIR / "val2017"
    if val_dir.exists():
        n_val = len(list(val_dir.glob("*.jpg")))
        print(f"Val images: {n_val} images in val2017/")

    print(f"Subset: {len(subset_info['indices'])} training image indices")
    for stratum_key in sorted(subset_info['stratum_counts'].keys()):
        count = subset_info['stratum_counts'][stratum_key]
        print(f"  {stratum_key}: ~{count}")

    print(f"Backbone: mobilenetv4_conv_small.e2400_r224_in1k — feature channels: {feature_channels}")
    print()


if __name__ == "__main__":
    main()
