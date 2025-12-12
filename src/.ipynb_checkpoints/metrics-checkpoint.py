# from __future__ import annotations
# import torch

# def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
#     # pred: (B,C,H,W) logits or probs; target: (B,H,W) long
#     if pred.shape[1] > 1:
#         pred_bin = pred.argmax(dim=1)
#     else:
#         pred_bin = (pred.sigmoid() > 0.5).long().squeeze(1)
#     tgt_bin = (target > 0).long()
#     inter = (pred_bin * tgt_bin).sum(dim=(1,2)).float()
#     sets = pred_bin.sum(dim=(1,2)).float() + tgt_bin.sum(dim=(1,2)).float()
#     dice = (2*inter + eps) / (sets + eps)
#     return dice.mean()

# class DiceLoss(torch.nn.Module):
#     def __init__(self, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#     def forward(self, logits: torch.Tensor, target: torch.Tensor):
#         if logits.shape[1] == 1:
#             tgt = torch.nn.functional.interpolate(tgt, size=probs.shape[2:], mode="nearest")
#             probs = logits.sigmoid()
#             tgt = (target > 0).float().unsqueeze(1)
#             inter = (probs * tgt).sum(dim=(1,2,3))
#             union = probs.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
#             dice = (2*inter + self.eps) / (union + self.eps)
#             return 1.0 - dice.mean()
#         else:
#             # multiclass soft dice (foreground vs background)
#             probs = torch.softmax(logits, dim=1)[:,1:2]
#             tgt = (target > 0).float().unsqueeze(1)
#             inter = (probs * tgt).sum(dim=(1,2,3))
#             union = probs.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
#             dice = (2*inter + self.eps) / (union + self.eps)
#             return 1.0 - dice.mean()
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_target(
    target: torch.Tensor,
    ref: torch.Tensor
) -> torch.Tensor:
    """
    Ensures target is binary float tensor of shape (B,1,H,W)
    and resized to match ref spatial dimensions.
    """
    # binarize
    tgt = (target > 0).float()

    # ensure channel dim
    if tgt.dim() == 3:              # (B,H,W)
        tgt = tgt.unsqueeze(1)      # → (B,1,H,W)

    # resize if needed
    if tgt.shape[2:] != ref.shape[2:]:
        tgt = F.interpolate(tgt, size=ref.shape[2:], mode="nearest")

    return tgt


@torch.no_grad()
def dice_coef(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """
    Binary Dice coefficient.

    logits: (B,1,H,W)
    target: (B,H,W) or (B,1,H,W) with {0,1}
    """
    probs = torch.sigmoid(logits)
    tgt = _prepare_target(target, probs)

    inter = (probs * tgt).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))

    dice = (2.0 * inter + eps) / (union + eps)
    return dice.mean().item()


class DiceBCELoss(nn.Module):
    """
    Binary Dice + BCEWithLogits loss.

    - Robust to empty masks
    - Shape safe
    - AMP compatible
    """
    def __init__(
        self,
        dice_w: float = 0.7,
        bce_w: float = 0.3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: (B,1,H,W)
        target: (B,H,W) or (B,1,H,W)
        """
        tgt = _prepare_target(target, logits)

        # BCE
        bce_loss = self.bce(logits, tgt)

        # Dice
        probs = torch.sigmoid(logits)
        inter = (probs * tgt).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice.mean()

        return self.dice_w * dice_loss + self.bce_w * bce_loss
