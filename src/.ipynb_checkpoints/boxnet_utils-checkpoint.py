from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def downsample_mask_to_256(msk: torch.Tensor) -> torch.Tensor:
    """
    msk: (B,H,W) or (B,1,H,W) binary/0-1
    returns: (B,1,256,256) float in {0,1}
    """
    if msk.dim() == 3:
        msk = msk.unsqueeze(1)  # (B,1,H,W)
    msk = (msk > 0).float()
    return F.interpolate(msk, size=(256, 256), mode="nearest")


def boxnet_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,1,256,256)
    target: (B,1,256,256) in {0,1}
    """
    return F.binary_cross_entropy_with_logits(logits, target)


def _mask_to_box_xyxy(mask01: torch.Tensor) -> Tuple[int, int, int, int] | None:
    """
    mask01: (H,W) bool/0-1
    returns xyxy box in that coordinate system, or None if empty
    """
    ys, xs = torch.where(mask01 > 0)
    if ys.numel() == 0:
        return None
    x0 = int(xs.min().item())
    x1 = int(xs.max().item())
    y0 = int(ys.min().item())
    y1 = int(ys.max().item())
    return (x0, y0, x1, y1)


def predict_box_from_boxnet(
    boxnet,
    images: torch.Tensor,          # (B,3,H,W)
    *,
    thresh: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predict one XYXY box per image using BoxNet.

    Returns:
      boxes: (B,4) float in original image pixel coords
      valid: (B,) bool; False means "no object predicted" (use empty mask)
    """
    device = images.device
    B, _, H, W = images.shape

    # BoxNet expects 256x256
    img_256 = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = boxnet(img_256)                 # (B,1,256,256)
        prob = torch.sigmoid(logits)             # (B,1,256,256)

    boxes = torch.zeros((B, 4), device=device, dtype=torch.float32)
    valid = torch.zeros((B,), device=device, dtype=torch.bool)

    # scale factors from 256-grid to original H,W
    sx = float(W) / 256.0
    sy = float(H) / 256.0

    for i in range(B):
        p = prob[i, 0]
        # if nothing confident, mark invalid
        if float(p.max().item()) < thresh:
            valid[i] = False
            boxes[i] = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.float32)
            continue

        m = (p >= thresh)
        b = _mask_to_box_xyxy(m)
        if b is None:
            valid[i] = False
            boxes[i] = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.float32)
            continue

        x0, y0, x1, y1 = b

        # map to original coords (clamp)
        X0 = max(0.0, min((x0 * sx), W - 1.0))
        X1 = max(0.0, min((x1 * sx), W - 1.0))
        Y0 = max(0.0, min((y0 * sy), H - 1.0))
        Y1 = max(0.0, min((y1 * sy), H - 1.0))

        # ensure sane ordering + at least 1px
        if X1 <= X0:
            X1 = min(W - 1.0, X0 + 1.0)
        if Y1 <= Y0:
            Y1 = min(H - 1.0, Y0 + 1.0)

        boxes[i] = torch.tensor([X0, Y0, X1, Y1], device=device, dtype=torch.float32)
        valid[i] = True

    return boxes, valid


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a,b: (B,4) float boxes in XYXY
    returns IoU: (B,)
    """
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix0 = torch.maximum(ax0, bx0)
    iy0 = torch.maximum(ay0, by0)
    ix1 = torch.minimum(ax1, bx1)
    iy1 = torch.minimum(ay1, by1)

    iw = (ix1 - ix0).clamp(min=0.0)
    ih = (iy1 - iy0).clamp(min=0.0)
    inter = iw * ih

    area_a = (ax1 - ax0).clamp(min=0.0) * (ay1 - ay0).clamp(min=0.0)
    area_b = (bx1 - bx0).clamp(min=0.0) * (by1 - by0).clamp(min=0.0)
    union = (area_a + area_b - inter).clamp(min=1e-6)

    return inter / union


def box_iou_from_logits(
    boxnet_logits: torch.Tensor,   # (B,1,256,256)
    gt_masks: torch.Tensor,        # (B,H,W) or (B,1,H,W)
    *,
    thresh: float = 0.30,
) -> torch.Tensor:
    """
    Compute IoU between predicted box (from logits) and GT box (from mask).
    Returns (B,) IoU. For empty GT masks, IoU is 0.
    """
    if gt_masks.dim() == 4:
        gt_masks = gt_masks[:, 0]
    B = gt_masks.shape[0]

    prob = torch.sigmoid(boxnet_logits)[:, 0]  # (B,256,256)
    gt_256 = downsample_mask_to_256(gt_masks)[:, 0]  # (B,256,256)

    pred_boxes = torch.zeros((B, 4), device=gt_masks.device, dtype=torch.float32)
    gt_boxes = torch.zeros((B, 4), device=gt_masks.device, dtype=torch.float32)

    for i in range(B):
        # pred box
        p = prob[i]
        if float(p.max().item()) < thresh:
            pred_boxes[i] = torch.tensor([0, 0, 1, 1], device=gt_masks.device)
        else:
            m = (p >= thresh)
            b = _mask_to_box_xyxy(m)
            if b is None:
                pred_boxes[i] = torch.tensor([0, 0, 1, 1], device=gt_masks.device)
            else:
                x0, y0, x1, y1 = b
                pred_boxes[i] = torch.tensor([x0, y0, x1, y1], device=gt_masks.device)

        # gt box
        g = (gt_256[i] > 0)
        gb = _mask_to_box_xyxy(g)
        if gb is None:
            gt_boxes[i] = torch.tensor([0, 0, 1, 1], device=gt_masks.device)
        else:
            x0, y0, x1, y1 = gb
            gt_boxes[i] = torch.tensor([x0, y0, x1, y1], device=gt_masks.device)

    # for truly empty GT, force IoU=0
    empty_gt = (gt_masks.sum(dim=(1, 2)) == 0)
    iou = box_iou_xyxy(pred_boxes, gt_boxes)
    iou = torch.where(empty_gt, torch.zeros_like(iou), iou)
    return iou
