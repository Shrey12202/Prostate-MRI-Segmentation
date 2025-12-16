from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.prompt import masks_to_boxes, jitter_boxes_xyxy
from src.boxnet_utils import predict_box_from_boxnet


def _center_small_box(H: int, W: int, frac: float = 0.20, device="cpu") -> torch.Tensor:
    """
    Returns (1,4) XYXY small box around center (used when GT mask is empty).
    """
    cx, cy = W * 0.5, H * 0.5
    bw, bh = W * frac, H * frac
    x0 = max(0.0, cx - bw * 0.5)
    x1 = min(W - 1.0, cx + bw * 0.5)
    y0 = max(0.0, cy - bh * 0.5)
    y1 = min(H - 1.0, cy + bh * 0.5)
    if x1 <= x0:
        x1 = min(W - 1.0, x0 + 1.0)
    if y1 <= y0:
        y1 = min(H - 1.0, y0 + 1.0)
    return torch.tensor([[x0, y0, x1, y1]], device=device, dtype=torch.float32)


def sam_forward_logits(
    sam,
    images: torch.Tensor,                 # (B,3,H,W)
    gt_masks: Optional[torch.Tensor],     # (B,H,W) or (B,1,H,W) OR None for prompt-less
    *,
    training: bool,
    box_jitter_frac: float = 0.10,
    boxnet=None,
    boxnet_thresh: float = 0.30,
) -> torch.Tensor:
    """
    Returns logits (B,1,H,W)

    - If gt_masks is not None: use GT-derived box prompts (with jitter during training)
      BUT for empty GT masks, we DO NOT use full-image box anymore.
    - If gt_masks is None: prompt-less; box comes from BoxNet.
      If BoxNet predicts "no object", logits are forced negative (empty pred).
    """
    B, _, H, W = images.shape
    device = images.device

    # normalize gt mask shape
    gt_masks_ = None
    if gt_masks is not None:
        if gt_masks.dim() == 4:
            gt_masks_ = gt_masks[:, 0]
        else:
            gt_masks_ = gt_masks

    logits_list = []

    for i in range(B):
        img_i = images[i : i + 1]  # (1,3,H,W)

        # Normalize for SAM
        img_i = (img_i - sam.pixel_mean) / sam.pixel_std

        # Encoder: trainable? then grad, else no_grad
        if any(p.requires_grad for p in sam.image_encoder.parameters()):
            image_embeddings = sam.image_encoder(img_i)
        else:
            with torch.no_grad():
                image_embeddings = sam.image_encoder(img_i)

        # ----------------------------
        # Choose prompt box
        # ----------------------------
        use_empty_output = False

        if gt_masks_ is not None:
            msk_i = gt_masks_[i : i + 1]  # (1,H,W)
            is_empty = (msk_i.sum() == 0)

            if is_empty:
                # CHANGE (1): do NOT prompt with full image box for empty masks
                box = _center_small_box(H, W, frac=0.20, device=device)  # (1,4)
            else:
                box = masks_to_boxes((msk_i > 0).float())  # (1,4)

            if training and box_jitter_frac > 0 and (msk_i.sum() > 0):
                box = jitter_boxes_xyxy(box, (H, W), jitter_frac=box_jitter_frac)

        else:
            # prompt-less: must have boxnet
            if boxnet is None:
                raise RuntimeError("gt_masks=None (prompt-less) requires boxnet != None")

            boxes, valid = predict_box_from_boxnet(boxnet, images[i : i + 1], thresh=boxnet_thresh)
            if not bool(valid[0].item()):
                use_empty_output = True
                box = _center_small_box(H, W, frac=0.20, device=device)
            else:
                box = boxes  # (1,4)

        # If BoxNet says "no object", force empty prediction (avoid huge false positives)
        if use_empty_output:
            logits_list.append(torch.full((1, 1, H, W), -10.0, device=device, dtype=torch.float32))
            continue

        box = box[:, None, :]  # (1,1,4)

        # prompt encoder (frozen)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )

        # mask decoder (trainable)
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # (1,1,256,256)

        logits_i = F.interpolate(
            low_res_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1,1,H,W)

        logits_list.append(logits_i)

    return torch.cat(logits_list, dim=0)  # (B,1,H,W)
