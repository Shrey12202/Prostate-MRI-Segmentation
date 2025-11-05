from __future__ import annotations
import torch

def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    # pred: (B,C,H,W) logits or probs; target: (B,H,W) long
    if pred.shape[1] > 1:
        pred_bin = pred.argmax(dim=1)
    else:
        pred_bin = (pred.sigmoid() > 0.5).long().squeeze(1)
    tgt_bin = (target > 0).long()
    inter = (pred_bin * tgt_bin).sum(dim=(1,2)).float()
    sets = pred_bin.sum(dim=(1,2)).float() + tgt_bin.sum(dim=(1,2)).float()
    dice = (2*inter + eps) / (sets + eps)
    return dice.mean()

class DiceLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if logits.shape[1] == 1:
            probs = logits.sigmoid()
            tgt = (target > 0).float().unsqueeze(1)
            inter = (probs * tgt).sum(dim=(1,2,3))
            union = probs.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
            dice = (2*inter + self.eps) / (union + self.eps)
            return 1.0 - dice.mean()
        else:
            # multiclass soft dice (foreground vs background)
            probs = torch.softmax(logits, dim=1)[:,1:2]
            tgt = (target > 0).float().unsqueeze(1)
            inter = (probs * tgt).sum(dim=(1,2,3))
            union = probs.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
            dice = (2*inter + self.eps) / (union + self.eps)
            return 1.0 - dice.mean()
