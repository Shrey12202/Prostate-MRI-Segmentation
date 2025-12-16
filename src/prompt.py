from __future__ import annotations
import torch


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor | None:
    """
    masks: (B,H,W) or (B,1,H,W) binary
    returns: (B,4) boxes in XYXY format [x0,y0,x1,y1]
             OR None if any mask in the batch is empty.

    IMPORTANT CHANGE:
    - Previously, empty mask returned full-image box.
    - Now: empty mask returns None (so we don't prompt full image for negatives).
    """
    if masks.dim() == 4:
        masks = masks[:, 0]

    B, H, W = masks.shape
    boxes = torch.zeros((B, 4), device=masks.device, dtype=torch.float32)

    for i in range(B):
        ys, xs = torch.where(masks[i] > 0)
        if xs.numel() == 0:
            return None
        x0 = xs.min().float()
        x1 = xs.max().float()
        y0 = ys.min().float()
        y1 = ys.max().float()
        boxes[i] = torch.stack([x0, y0, x1, y1])

    return boxes


def jitter_boxes_xyxy(
    boxes: torch.Tensor,
    image_hw: tuple[int, int],
    jitter_frac: float = 0.10,
) -> torch.Tensor:
    """
    Jitter XYXY boxes by expanding/contracting each side by up to jitter_frac of box size.
    boxes: (B,4) float [x0,y0,x1,y1]
    image_hw: (H,W)
    """
    H, W = image_hw
    b = boxes.clone()

    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    bw = (x1 - x0).clamp(min=1.0)
    bh = (y1 - y0).clamp(min=1.0)

    dx0 = (torch.rand_like(bw) * 2 - 1.0) * (jitter_frac * bw)
    dx1 = (torch.rand_like(bw) * 2 - 1.0) * (jitter_frac * bw)
    dy0 = (torch.rand_like(bh) * 2 - 1.0) * (jitter_frac * bh)
    dy1 = (torch.rand_like(bh) * 2 - 1.0) * (jitter_frac * bh)

    x0n = (x0 + dx0).clamp(0, W - 1)
    y0n = (y0 + dy0).clamp(0, H - 1)
    x1n = (x1 + dx1).clamp(0, W - 1)
    y1n = (y1 + dy1).clamp(0, H - 1)

    x0f = torch.minimum(x0n, x1n - 1.0).clamp(0, W - 2)
    y0f = torch.minimum(y0n, y1n - 1.0).clamp(0, H - 2)
    x1f = torch.maximum(x1n, x0f + 1.0).clamp(1, W - 1)
    y1f = torch.maximum(y1n, y0f + 1.0).clamp(1, H - 1)

    return torch.stack([x0f, y0f, x1f, y1f], dim=1)
