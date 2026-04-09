"""
initial_compress.py — Phase 1 seed program for ShinkaEvolve.

Everything inside the EVOLVE-BLOCK-START / END markers is mutable by
ShinkaEvolve.  Everything outside is fixed scaffolding.

Architecture: MobileNetV4-Conv-S backbone (frozen) → FPN neck with UIB
refinement blocks → FCOS detection head.  Single class (person).
"""

import os
os.makedirs("checkpoints", exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed training recipe — NOT mutable by ShinkaEvolve.
# Only neck/head architecture parameters inside the EVOLVE-BLOCK may change.
# ---------------------------------------------------------------------------
LR = 1e-3
EPOCHS = 4
BATCH_SIZE = 8
WARMUP_EPOCHS = 0.5

# EVOLVE-BLOCK-START
"""
FPN + FCOS detector seeded for ShinkaEvolve Phase 1.

Backbone: mobilenetv4_conv_small.e2400_r224_in1k
  - C3 (stride 8):  64 ch
  - C4 (stride 16): 96 ch
  - C5 (stride 32): 960 ch

Neck: lateral projections + top-down FPN + UIB refinement per level.
Head: shared FCOS classification / regression towers (single class).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config constants — ShinkaEvolve mutates these values
# ---------------------------------------------------------------------------

NECK_CHANNELS = [96, 96, 96]   # output channels per FPN level P3/P4/P5

# One UIB config dict per FPN level (P3, P4, P5).
# Keys map to UniversalInvertedResidual __init__ kwargs.
UIB_CONFIGS = [
    # P3 — finest scale, lightweight
    {
        "dw_kernel_size_start": 0,
        "dw_kernel_size_mid": 3,
        "dw_kernel_size_end": 0,
        "exp_ratio": 2.0,
    },
    # P4 — mid scale
    {
        "dw_kernel_size_start": 3,
        "dw_kernel_size_mid": 3,
        "dw_kernel_size_end": 0,
        "exp_ratio": 2.0,
    },
    # P5 — coarsest scale
    {
        "dw_kernel_size_start": 3,
        "dw_kernel_size_mid": 0,
        "dw_kernel_size_end": 0,
        "exp_ratio": 2.0,
    },
]

HEAD_CHANNELS = 64
HEAD_STACKS = 3
FROZEN_STAGES = 4
IMG_SIZE = 320  # fixed — do not change


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(backbone_channels):
    """Build and return (FPNNeck class, FCOSHead class, TinyDetector class).

    Args:
        backbone_channels: list of ints [C3_ch, C4_ch, C5_ch] from backbone.

    Returns:
        Tuple of three *classes* (not instances): FPNNeck, FCOSHead, TinyDetector.
    """
    from blocks.uib import UniversalInvertedResidual

    class ConvBNReLU(nn.Sequential):
        """1×1 or k×k Conv + BN + ReLU."""

        def __init__(self, in_ch, out_ch, kernel_size=1, padding=0):
            super().__init__(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    # ------------------------------------------------------------------
    class FPNNeck(nn.Module):
        """Feature Pyramid Network neck with UIB refinement blocks.

        Args:
            in_channels:  [C3_ch, C4_ch, C5_ch] from backbone.
            neck_channels: [P3_ch, P4_ch, P5_ch] output channels.
            uib_configs:  list of 3 dicts passed to UniversalInvertedResidual.
        """

        def __init__(self, in_channels, neck_channels, uib_configs):
            super().__init__()
            assert len(in_channels) == 3
            assert len(neck_channels) == 3
            assert len(uib_configs) == 3

            c3_ch, c4_ch, c5_ch = in_channels
            p3_ch, p4_ch, p5_ch = neck_channels

            # Lateral projections: backbone_ch → neck_ch
            self.lat_p5 = ConvBNReLU(c5_ch, p5_ch)
            self.lat_p4 = ConvBNReLU(c4_ch, p4_ch)
            self.lat_p3 = ConvBNReLU(c3_ch, p3_ch)

            # Top-down merging: after upsampling we merge with same-channel lateral
            # so neck_channels must agree across levels for simple addition.
            # We use a 1×1 conv after merge to keep channels consistent.
            self.merge_p4 = ConvBNReLU(p4_ch, p4_ch)
            self.merge_p3 = ConvBNReLU(p3_ch, p3_ch)

            # UIB refinement at each level
            self.refine_p3 = UniversalInvertedResidual(p3_ch, p3_ch, **uib_configs[0])
            self.refine_p4 = UniversalInvertedResidual(p4_ch, p4_ch, **uib_configs[1])
            self.refine_p5 = UniversalInvertedResidual(p5_ch, p5_ch, **uib_configs[2])

        def forward(self, features):
            c3, c4, c5 = features

            # Lateral
            p5 = self.lat_p5(c5)
            p4 = self.lat_p4(c4)
            p3 = self.lat_p3(c3)

            # Top-down pathway
            p4 = self.merge_p4(
                p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
            )
            p3 = self.merge_p3(
                p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
            )

            # UIB refinement
            p3 = self.refine_p3(p3)
            p4 = self.refine_p4(p4)
            p5 = self.refine_p5(p5)

            return [p3, p4, p5]

    # ------------------------------------------------------------------
    class FCOSHead(nn.Module):
        """Per-level FCOS head (single class).

        Each FPN level gets its own cls/reg tower to avoid the dynamo ONNX
        exporter bug that occurs when the same nn.Module is invoked in a
        Python for-loop (duplicate initializer names in the exported graph).
        Logically equivalent to a shared tower with tied weights; weights
        can be manually tied after construction if desired.

        Args:
            neck_channels: list of 3 ints [P3_ch, P4_ch, P5_ch].
                           All must be equal for this implementation.
            head_channels: inner channel width of the towers.
            num_stacks:    number of Conv-BN-ReLU blocks per tower.
        """

        def __init__(self, neck_channels, head_channels, num_stacks):
            super().__init__()
            in_ch = neck_channels[0]
            num_levels = len(neck_channels)

            def _tower():
                layers = []
                ch_in = in_ch
                for _ in range(num_stacks):
                    layers += [
                        nn.Conv2d(ch_in, head_channels, 3, padding=1, bias=False),
                        nn.BatchNorm2d(head_channels),
                        nn.ReLU(inplace=True),
                    ]
                    ch_in = head_channels
                return nn.Sequential(*layers)

            # One tower per level — avoids dynamo ONNX shared-module bug
            self.cls_towers = nn.ModuleList([_tower() for _ in range(num_levels)])
            self.reg_towers = nn.ModuleList([_tower() for _ in range(num_levels)])

            # Prediction heads (one set, called per-level in forward)
            self.cls_pred_p3 = nn.Conv2d(head_channels, 1, 1)
            self.cls_pred_p4 = nn.Conv2d(head_channels, 1, 1)
            self.cls_pred_p5 = nn.Conv2d(head_channels, 1, 1)
            self.reg_pred_p3 = nn.Conv2d(head_channels, 4, 1)
            self.reg_pred_p4 = nn.Conv2d(head_channels, 4, 1)
            self.reg_pred_p5 = nn.Conv2d(head_channels, 4, 1)
            self.ctr_pred_p3 = nn.Conv2d(head_channels, 1, 1)
            self.ctr_pred_p4 = nn.Conv2d(head_channels, 1, 1)
            self.ctr_pred_p5 = nn.Conv2d(head_channels, 1, 1)

        def forward(self, features):
            """
            Args:
                features: list of [P3, P4, P5] tensors.

            Returns:
                (cls_preds, reg_preds, ctr_preds) — each a list of len 3.
                Shapes per level: cls (B,1,H,W), reg (B,4,H,W), ctr (B,1,H,W).
                reg values are passed through exp() to enforce positivity.
            """
            p3, p4, p5 = features

            cls_feat_p3 = self.cls_towers[0](p3)
            reg_feat_p3 = self.reg_towers[0](p3)
            cls_feat_p4 = self.cls_towers[1](p4)
            reg_feat_p4 = self.reg_towers[1](p4)
            cls_feat_p5 = self.cls_towers[2](p5)
            reg_feat_p5 = self.reg_towers[2](p5)

            cls_preds = [
                self.cls_pred_p3(cls_feat_p3),
                self.cls_pred_p4(cls_feat_p4),
                self.cls_pred_p5(cls_feat_p5),
            ]
            reg_preds = [
                self.reg_pred_p3(reg_feat_p3).exp(),
                self.reg_pred_p4(reg_feat_p4).exp(),
                self.reg_pred_p5(reg_feat_p5).exp(),
            ]
            ctr_preds = [
                self.ctr_pred_p3(cls_feat_p3),
                self.ctr_pred_p4(cls_feat_p4),
                self.ctr_pred_p5(cls_feat_p5),
            ]
            return cls_preds, reg_preds, ctr_preds

    # ------------------------------------------------------------------
    class TinyDetector(nn.Module):
        """Full detector: backbone + FPN neck + FCOS head."""

        def __init__(self, backbone, neck, head):
            super().__init__()
            self.backbone = backbone
            self.neck = neck
            self.head = head

        def forward(self, x):
            features = self.backbone(x)       # [C3, C4, C5]
            fpn_feats = self.neck(features)   # [P3, P4, P5]
            return self.head(fpn_feats)       # (cls_preds, reg_preds, ctr_preds)

    return FPNNeck, FCOSHead, TinyDetector


# ---------------------------------------------------------------------------
# FCOS loss
# ---------------------------------------------------------------------------

def fcos_loss(cls_preds, reg_preds, ctr_preds, targets, strides):
    """Compute FCOS training loss.

    Args:
        cls_preds: list of (B, 1, H, W) logit tensors, one per FPN level.
        reg_preds: list of (B, 4, H, W) positive distance tensors (after exp).
        ctr_preds: list of (B, 1, H, W) centerness logit tensors.
        targets:   list of dicts per image with keys:
                     'boxes'    FloatTensor[N, 4] xyxy at IMG_SIZE scale
                     'labels'   LongTensor[N]
                     'image_id' LongTensor[1]
        strides:   list of ints [8, 16, 32].

    Returns:
        Scalar loss tensor (focal + iou + bce centerness).
    """
    from torchvision.ops import sigmoid_focal_loss, box_iou

    device = cls_preds[0].device
    B = cls_preds[0].shape[0]

    total_cls_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    total_ctr_loss = torch.tensor(0.0, device=device)
    total_pos = 0

    # Scale-of-interest thresholds per FPN level (area in pixels²).
    # P3 (stride 8): small objects; P4 (stride 16): medium; P5 (stride 32): large.
    # Overlapping boundaries reduce missed assignments at scale transitions.
    _SOI_MIN = [0,      32**2,  64**2]   # inclusive lower bound per level
    _SOI_MAX = [96**2,  192**2, float("inf")]  # exclusive upper bound per level
    # Center-sampling radius: only points within this many strides of the
    # GT box center are eligible (reduces noisy positives at box boundaries).
    _CENTER_RADIUS = 1.5

    for lvl_idx, stride in enumerate(strides):
        cls_pred = cls_preds[lvl_idx]   # (B, 1, H, W)
        reg_pred = reg_preds[lvl_idx]   # (B, 4, H, W)
        ctr_pred = ctr_preds[lvl_idx]   # (B, 1, H, W)

        H, W = cls_pred.shape[2], cls_pred.shape[3]

        # Build center grid [H*W] each
        cols = torch.arange(W, device=device, dtype=torch.float32)
        rows = torch.arange(H, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # (H, W)
        cx = (grid_x + 0.5) * stride  # (H, W)
        cy = (grid_y + 0.5) * stride  # (H, W)
        # Flatten to (H*W,)
        cx_flat = cx.reshape(-1)
        cy_flat = cy.reshape(-1)
        num_pts = H * W

        soi_min = _SOI_MIN[lvl_idx]
        soi_max = _SOI_MAX[lvl_idx]
        center_r = _CENTER_RADIUS * stride

        # Per-image processing
        for img_idx in range(B):
            gt_boxes = targets[img_idx]["boxes"]  # (N, 4) xyxy
            N = gt_boxes.shape[0]

            # Flatten predictions for this image at this level
            cls_i = cls_pred[img_idx, 0].reshape(-1)   # (H*W,)
            reg_i = reg_pred[img_idx].reshape(4, -1).T  # (H*W, 4): l,t,r,b
            ctr_i = ctr_pred[img_idx, 0].reshape(-1)   # (H*W,)

            # Classification target (all background initially)
            cls_target = torch.zeros(num_pts, device=device)

            if N == 0:
                # All negative — only focal loss on background
                total_cls_loss = total_cls_loss + sigmoid_focal_loss(
                    cls_i.unsqueeze(1),
                    cls_target.unsqueeze(1),
                    alpha=0.25, gamma=2.0, reduction="sum",
                )
                continue

            # gt_boxes: (N, 4) — x1, y1, x2, y2
            x1 = gt_boxes[:, 0]  # (N,)
            y1 = gt_boxes[:, 1]
            x2 = gt_boxes[:, 2]
            y2 = gt_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)  # (N,)

            # Scale-of-interest filter: only assign GT boxes whose area falls
            # within this level's range. This gives each level a specialised
            # scale so small objects don't leak into P5 and vice versa.
            soi_valid = (areas >= soi_min) & (areas < soi_max)  # (N,)

            # Center-sampling: a point is eligible for GT box n only if it
            # lies within center_r of the box's geometric centre.
            cx_gt = ((x1 + x2) / 2).unsqueeze(1)  # (N, 1)
            cy_gt = ((y1 + y2) / 2).unsqueeze(1)  # (N, 1)
            in_center = (
                (cx_flat.unsqueeze(0) - cx_gt).abs() < center_r
            ) & (
                (cy_flat.unsqueeze(0) - cy_gt).abs() < center_r
            )  # (N, H*W)

            # cx_flat: (H*W,) → broadcast with (N,) → (N, H*W)
            inside_x = (cx_flat.unsqueeze(0) > x1.unsqueeze(1)) & \
                       (cx_flat.unsqueeze(0) < x2.unsqueeze(1))
            inside_y = (cy_flat.unsqueeze(0) > y1.unsqueeze(1)) & \
                       (cy_flat.unsqueeze(0) < y2.unsqueeze(1))
            inside = inside_x & inside_y & in_center  # (N, H*W)

            # Apply scale-of-interest: mask out GT boxes outside this level's range
            inside = inside & soi_valid.unsqueeze(1)  # (N, H*W)

            # Encode positive points
            pos_mask = inside.any(dim=0)  # (H*W,) — at least one GT contains it
            pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)  # (P,)

            if pos_indices.numel() == 0:
                total_cls_loss = total_cls_loss + sigmoid_focal_loss(
                    cls_i.unsqueeze(1),
                    cls_target.unsqueeze(1),
                    alpha=0.25, gamma=2.0, reduction="sum",
                )
                continue

            # Assign each positive point to the GT with smallest area
            # inside[:, pos_indices]: (N, P)
            inside_pos = inside[:, pos_indices]  # (N, P)
            # Set area to inf where box doesn't contain the point
            areas_expanded = areas.unsqueeze(1).expand(N, pos_indices.numel())
            areas_masked = torch.where(inside_pos, areas_expanded,
                                       torch.full_like(areas_expanded, float("inf")))
            assigned_gt = areas_masked.argmin(dim=0)  # (P,)

            # Mark as positive in cls target
            cls_target[pos_indices] = 1.0

            # Compute (l, t, r, b) for positive points
            cx_pos = cx_flat[pos_indices]  # (P,)
            cy_pos = cy_flat[pos_indices]

            gt_x1 = x1[assigned_gt]  # (P,)
            gt_y1 = y1[assigned_gt]
            gt_x2 = x2[assigned_gt]
            gt_y2 = y2[assigned_gt]

            l_target = cx_pos - gt_x1
            t_target = cy_pos - gt_y1
            r_target = gt_x2 - cx_pos
            b_target = gt_y2 - cy_pos
            ltrb_target = torch.stack([l_target, t_target, r_target, b_target], dim=1)  # (P, 4)

            # Centerness target
            lr_min = torch.minimum(l_target, r_target)
            lr_max = torch.maximum(l_target, r_target)
            tb_min = torch.minimum(t_target, b_target)
            tb_max = torch.maximum(t_target, b_target)
            ctr_target = torch.sqrt(
                (lr_min / lr_max.clamp(min=1e-6)) * (tb_min / tb_max.clamp(min=1e-6))
            )  # (P,)

            # --- Classification loss (focal) ---
            total_cls_loss = total_cls_loss + sigmoid_focal_loss(
                cls_i.unsqueeze(1),
                cls_target.unsqueeze(1),
                alpha=0.25, gamma=2.0, reduction="sum",
            )

            # --- Regression loss (IoU) ---
            # Decode predicted boxes from positive points
            reg_pos = reg_i[pos_indices]  # (P, 4): l, t, r, b
            pred_x1 = cx_pos - reg_pos[:, 0]
            pred_y1 = cy_pos - reg_pos[:, 1]
            pred_x2 = cx_pos + reg_pos[:, 2]
            pred_y2 = cy_pos + reg_pos[:, 3]
            pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # (P, 4)
            gt_boxes_pos = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)        # (P, 4)

            # IoU loss = 1 - IoU (diagonal of box_iou gives matched-pair IoU)
            iou_mat = box_iou(pred_boxes, gt_boxes_pos)  # (P, P)
            iou_diag = iou_mat.diagonal()  # (P,)
            reg_loss = (1.0 - iou_diag).sum()
            total_reg_loss = total_reg_loss + reg_loss

            # --- Centerness loss (BCE) ---
            ctr_pos = ctr_i[pos_indices]  # (P,)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pos, ctr_target, reduction="sum"
            )
            total_ctr_loss = total_ctr_loss + ctr_loss

            total_pos += pos_indices.numel()

    denom = max(total_pos, 1)
    loss = (total_cls_loss + total_reg_loss + total_ctr_loss) / denom
    return loss


# ---------------------------------------------------------------------------
# Experiment runner — called by ShinkaEvolve
# ---------------------------------------------------------------------------

def run_experiment(seed: int = 1) -> dict:
    """Train a TinyDetector for EPOCHS epochs and return evaluation metrics.

    Returns:
        dict with keys: score, mAP50, model_size_mb, cpu_latency_ms.
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import OneCycleLR
    from train_utils import load_backbone, make_dataloader, export_onnx, save_checkpoint
    from evaluate_core import evaluate_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, backbone_channels = load_backbone(frozen_stages=FROZEN_STAGES)

    FPNNeck, FCOSHead, TinyDetector = build_model(backbone_channels)
    neck = FPNNeck(backbone_channels, NECK_CHANNELS, UIB_CONFIGS)
    head = FCOSHead(NECK_CHANNELS, HEAD_CHANNELS, HEAD_STACKS)
    model = TinyDetector(backbone, neck, head).to(device)

    strides = [8, 16, 32]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    train_loader = make_dataloader("train", BATCH_SIZE, IMG_SIZE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=EPOCHS * len(train_loader),
        pct_start=WARMUP_EPOCHS / EPOCHS,
    )

    from tqdm import tqdm

    model.train()
    total_steps = EPOCHS * len(train_loader)
    with tqdm(total=total_steps, unit="batch", desc="Training") as pbar:
        for epoch in range(EPOCHS):
            for imgs, targets in train_loader:
                imgs_t = torch.stack(imgs).to(device)
                tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                cls_preds, reg_preds, ctr_preds = model(imgs_t)
                loss = fcos_loss(cls_preds, reg_preds, ctr_preds, tgts, strides)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(epoch=f"{epoch + 1}/{EPOCHS}", loss=f"{loss.item():.4f}")
                pbar.update(1)

    model.eval()
    onnx_path = "checkpoints/phase1_candidate.onnx"
    export_onnx(model, onnx_path, IMG_SIZE)

    # evaluate_core.py computes mAP50, latency, and model size
    metrics = evaluate_model(onnx_path)

    save_checkpoint(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "config": {
                "NECK_CHANNELS": NECK_CHANNELS,
                "UIB_CONFIGS": UIB_CONFIGS,
                "HEAD_CHANNELS": HEAD_CHANNELS,
                "HEAD_STACKS": HEAD_STACKS,
            },
        },
        "checkpoints/phase1_candidate.pt",
    )

    return {
        "score": metrics["score"],
        "mAP50": metrics["mAP50"],
        "mAP": metrics["mAP"],
        "AP_small": metrics["AP_small"],
        "model_size_mb": metrics["model_size_mb"],
        "cpu_latency_ms": metrics["cpu_latency_ms"],
    }

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import torch
    print("=== Smoke test: architecture only (no COCO data required) ===")

    backbone_channels = [64, 96, 960]  # real MNv4-Conv-S channels
    FPNNeck, FCOSHead, TinyDetector = build_model(backbone_channels)
    neck = FPNNeck(backbone_channels, NECK_CHANNELS, UIB_CONFIGS)
    head = FCOSHead(NECK_CHANNELS, HEAD_CHANNELS, HEAD_STACKS)

    class _DummyBackbone(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return [
                torch.zeros(B, 64,  40, 40),
                torch.zeros(B, 96,  20, 20),
                torch.zeros(B, 960, 10, 10),
            ]

    model = TinyDetector(_DummyBackbone(), neck, head).eval()
    with torch.no_grad():
        cls_p, reg_p, ctr_p = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))

    assert len(cls_p) == 3, "Expected 3 FPN levels"
    assert cls_p[0].shape[1] == 1, f"cls should have 1 channel, got {cls_p[0].shape[1]}"
    assert reg_p[0].shape[1] == 4, f"reg should have 4 channels, got {reg_p[0].shape[1]}"
    print(f"  P3 cls: {cls_p[0].shape}, reg: {reg_p[0].shape}, ctr: {ctr_p[0].shape}")
    print(f"  P4 cls: {cls_p[1].shape}, reg: {reg_p[1].shape}")
    print(f"  P5 cls: {cls_p[2].shape}, reg: {reg_p[2].shape}")

    # Test fcos_loss with model-generated tensors (so they have grad_fn)
    model.train()
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    cls_p, reg_p, ctr_p = model(x)
    dummy_targets = [{
        "boxes": torch.tensor([[50., 50., 200., 200.]]),
        "labels": torch.tensor([1]),
        "image_id": torch.tensor([0]),
    }]
    loss = fcos_loss(cls_p, reg_p, ctr_p, dummy_targets, [8, 16, 32])
    assert loss.requires_grad, "loss must have grad_fn for backward to work"
    loss.backward()
    print(f"  fcos_loss = {loss.item():.4f}, backward OK")
    print("=== Smoke test PASSED ===")
    print("To run a full experiment (requires COCO data), call run_experiment() directly.")
