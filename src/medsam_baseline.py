from __future__ import annotations
import torch
from segment_anything import sam_model_registry


def build_sam_baseline(
    ckpt_path: str,
    model_type: str = "vit_b",
    device: str = "cuda",
    unfreeze_last_blocks: int = 0,
):
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device)

    # -------------------------------------------------
    # Freeze everything
    # -------------------------------------------------
    for p in sam.parameters():
        p.requires_grad = False

    # -------------------------------------------------
    # Always train mask decoder
    # -------------------------------------------------
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True

    # -------------------------------------------------
    # Optionally unfreeze last N encoder blocks
    # -------------------------------------------------
    if unfreeze_last_blocks > 0:
        blocks = sam.image_encoder.blocks  # list of ViT blocks
        for blk in blocks[-unfreeze_last_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True

    return sam
