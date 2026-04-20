"""Backbone loader for autoresearch-tinydet."""

from __future__ import annotations

from typing import List, Tuple

import timm
import torch.nn as nn


def load_backbone(frozen_stages: int = 4) -> Tuple[nn.Module, List[int]]:
    """Load MNv4-Conv-S with features_only=True, out_indices=(2,3,4).

    Returns (backbone, feature_channels) for C3/C4/C5.
    feature_channels: e.g. [64, 96, 960] for mobilenetv4_conv_small.

    frozen_stages controls how many top-level backbone children are frozen:
      4 = ALL frozen (Phase 1 default — fastest experiments)
      2 = conv_stem + bn1 frozen (Phase 2-3 default — co-adaptation)
      0 = fully trainable
    """
    backbone = timm.create_model(
        "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4),
    )

    feature_channels: List[int] = backbone.feature_info.channels()

    for i, (_, module) in enumerate(backbone.named_children()):
        if i >= frozen_stages:
            break
        for param in module.parameters():
            param.requires_grad = False

    return backbone, feature_channels
