"""tinydet/repro.py — Shared reproducibility helpers."""
from __future__ import annotations
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable torch deterministic algorithms
            (may reduce performance, use for final validation runs only).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
