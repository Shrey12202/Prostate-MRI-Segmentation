from __future__ import annotations
import torch, os
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from metrics import DiceLoss, dice_coef
from config import cfg

def make_optim(model):
    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.num_epochs)
    return opt, sch

def save_ckpt(model, epoch: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch}, path)

def log_scalar(writer, name: str, value: float, step: int):
    if writer is not None:
        writer.add_scalar(name, value, step)

def step_metrics(logits, target):
    loss_fn = DiceLoss()
    loss = loss_fn(logits, target)
    dice = dice_coef(logits, target).item()
    return loss, dice
