from __future__ import annotations

import argparse, os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.config import cfg
from src.models.unet_decoder import UNetDecoder
from src.data.dataset import make_loader
from src.utils.train_utils import make_optim, save_ckpt, log_scalar, step_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(cfg.preprocessed_dir / "manifest.csv"))
    ap.add_argument("--train_split", type=str, default=str(cfg.splits_dir / "train.csv"))
    ap.add_argument("--val_split", type=str, default=str(cfg.splits_dir / "val.csv"))
    ap.add_argument("--embeddings_dir", type=str, default=str(cfg.embeddings_dir))
    ap.add_argument("--epochs", type=int, default=cfg.num_epochs)
    ap.add_argument("--batch_size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.learning_rate)
    ap.add_argument("--ckpt_dir", type=str, default=str(cfg.ckpt_dir))
    ap.add_argument("--log_dir", type=str, default=str(cfg.log_dir))
    ap.add_argument("--amp", action="store_true" if cfg.amp else "store_false")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(args.log_dir)

    model = UNetDecoder(in_channels=cfg.embedding_channels, num_classes=cfg.num_classes, stride=cfg.embedding_stride).to(device)
    opt, sch = make_optim(model)

    train_loader = make_loader(Path(args.manifest), Path(args.train_split), Path(args.embeddings_dir), args.batch_size, shuffle=True, augment=True, num_workers=cfg.num_workers)
    val_loader   = make_loader(Path(args.manifest), Path(args.val_split), Path(args.embeddings_dir), batch_size=1, shuffle=False, augment=False, num_workers=cfg.num_workers)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0
    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)         # (B,1,H,W) (not used by decoder but useful for viz)
            msk = batch["mask"].to(device)          # (B,H,W)
            emb = batch["embedding"].to(device)     # (B,C,h,w)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(emb)
                loss, dice = step_metrics(logits, msk)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            log_scalar(writer, "train/loss", loss.item(), global_step)
            log_scalar(writer, "train/dice", dice, global_step)
            global_step += 1

        sch.step()

        # ---- Validation ----
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                msk = batch["mask"].to(device)
                emb = batch["embedding"].to(device)
                logits = model(emb)
                _, dice = step_metrics(logits, msk)
                dices.append(dice)
        val_dice = sum(dices)/max(1,len(dices))
        log_scalar(writer, "val/dice", val_dice, epoch)

        if val_dice > best_val:
            best_val = val_dice
            save_ckpt(model, epoch, Path(args.ckpt_dir)/"best.pt")
        save_ckpt(model, epoch, Path(args.ckpt_dir)/"last.pt")
        print(f"Epoch {epoch}: val Dice={val_dice:.4f} (best={best_val:.4f})")

if __name__ == "__main__":
    main()
