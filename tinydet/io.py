"""Path constants and checkpoint/ONNX helpers for autoresearch-tinydet."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path(
    os.environ.get("AUTORESEARCH_CACHE", Path.home() / ".cache" / "autoresearch-tinydet")
)
SUBSET_INDEX_PATH: Path = CACHE_DIR / "subset_5k_index.json"
COCO_ROOT: Path = Path(os.environ.get("COCO_ROOT", CACHE_DIR))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, path: str, img_size: int = 320) -> str:
    """Export model to ONNX (opset 18) and return the resolved path."""
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=18,
        input_names=["images"],
        dynamic_axes={"images": {0: "batch"}},
        do_constant_folding=True,
    )
    return str(Path(path).resolve())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    """Save state dict to path, creating parent dirs if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(out))


def load_checkpoint(path: str) -> dict:
    """Load and return a checkpoint saved with save_checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)
