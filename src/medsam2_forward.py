from __future__ import annotations
import torch
import torch.nn.functional as F


def masks_to_boxes_xyxy(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B,H,W) binary {0,1}
    returns: (B,4) XYXY pixel coords (x0,y0,x1,y1)
    empty mask -> full image box
    """
    B, H, W = mask.shape
    boxes = []
    for b in range(B):
        ys, xs = torch.where(mask[b] > 0)
        if ys.numel() == 0:
            boxes.append(torch.tensor([0, 0, W - 1, H - 1], device=mask.device))
        else:
            x0 = xs.min()
            x1 = xs.max()
            y0 = ys.min()
            y1 = ys.max()
            boxes.append(torch.stack([x0, y0, x1, y1]).to(mask.device))
    return torch.stack(boxes, dim=0).float()


def jitter_boxes_xyxy(boxes: torch.Tensor, H: int, W: int, frac: float) -> torch.Tensor:
    """
    boxes: (B,4) XYXY
    frac: jitter fraction relative to box size
    """
    if frac <= 0:
        return boxes

    b = boxes.clone()
    bw = (b[:, 2] - b[:, 0]).clamp(min=1.0)
    bh = (b[:, 3] - b[:, 1]).clamp(min=1.0)

    jx = (torch.rand_like(bw) * 2 - 1) * (bw * frac)
    jy = (torch.rand_like(bh) * 2 - 1) * (bh * frac)

    b[:, 0] = (b[:, 0] + jx).clamp(0, W - 1)
    b[:, 2] = (b[:, 2] + jx).clamp(0, W - 1)
    b[:, 1] = (b[:, 1] + jy).clamp(0, H - 1)
    b[:, 3] = (b[:, 3] + jy).clamp(0, H - 1)

    # ensure ordering
    x0 = torch.min(b[:, 0], b[:, 2])
    x1 = torch.max(b[:, 0], b[:, 2])
    y0 = torch.min(b[:, 1], b[:, 3])
    y1 = torch.max(b[:, 1], b[:, 3])
    return torch.stack([x0, y0, x1, y1], dim=1)


def _get_prompt_encoder(model):
    if hasattr(model, "prompt_encoder"):
        return model.prompt_encoder
    if hasattr(model, "sam_prompt_encoder"):
        return model.sam_prompt_encoder
    raise AttributeError("No prompt encoder found (prompt_encoder / sam_prompt_encoder).")


def _get_mask_decoder(model):
    if hasattr(model, "mask_decoder"):
        return model.mask_decoder
    if hasattr(model, "sam_mask_decoder"):
        return model.sam_mask_decoder
    raise AttributeError("No mask decoder found (mask_decoder / sam_mask_decoder).")


def _extract_image_embeddings(model, images: torch.Tensor):
    """
    SAM2 image_encoder usually returns a dict that contains 'image_embeddings'.
    We keep it opaque and pass it to the decoder exactly as expected.
    """
    out = model.image_encoder(images)
    if isinstance(out, dict):
        if "image_embeddings" in out:
            return out["image_embeddings"]
        # fallback: pick first tensor-like thing
        for v in out.values():
            if torch.is_tensor(v) or isinstance(v, (list, tuple, dict)):
                return v
        raise KeyError(f"image_encoder output dict has keys: {list(out.keys())}")
    return out


def forward_logits_from_boxes(
    sam_model,
    images: torch.Tensor,          # (B,3,H,W)
    boxes_xyxy: torch.Tensor,      # (B,4) XYXY in pixels
    multimask_output: bool = False,
) -> torch.Tensor:
    """
    Full MedSAM2 forward:
      image_encoder -> prompt_encoder (boxes) -> mask_decoder -> upsample to (H,W)

    Returns: logits (B,1,H,W)
    """
    device = images.device
    B, _, H, W = images.shape

    prompt_encoder = _get_prompt_encoder(sam_model)
    mask_decoder = _get_mask_decoder(sam_model)

    image_embeddings = _extract_image_embeddings(sam_model, images)

    boxes = boxes_xyxy.to(device).float()

    sparse_emb, dense_emb = prompt_encoder(
        points=None,
        boxes=boxes,
        masks=None,
    )

    dec_out = mask_decoder(
    image_embeddings=image_embeddings,
    image_pe=prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_emb,
    dense_prompt_embeddings=dense_emb,
    multimask_output=multimask_output,
    repeat_image=False,   # ✅ REQUIRED by this repo’s MaskDecoder
)


    low_res_masks = dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out

    if low_res_masks.ndim == 3:
        low_res_masks = low_res_masks.unsqueeze(1)
    if low_res_masks.shape[1] > 1:
        low_res_masks = low_res_masks[:, :1]

    logits = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)
    return logits
