from __future__ import annotations

import argparse, os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.config import cfg
from src.unet_decoder import UNetDecoder
from src.dataset import make_loader
from src.train_utils import make_optim, save_ckpt, log_scalar, step_metrics

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

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
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #writer = SummaryWriter(args.log_dir)

    # model = UNetDecoder(
    #     in_channels=cfg.embedding_channels,
    #     num_classes=cfg.num_classes,
    #     stride=cfg.embedding_stride,
    #     base=64 # or 32 if you change it later
    # ).to(device)
    # opt, sch = make_optim(model)

    # # ---------- NEW: resume from last checkpoint if it exists ----------
    # ckpt_last = Path(args.ckpt_dir) / "last_base64.pt"
    # start_epoch = 1
    # best_val = 0.0
    # global_step = 0

    # if ckpt_last.exists():
    #     print(f"🔄 Found checkpoint at {ckpt_last}, loading...")
    #     ckpt = torch.load(ckpt_last, map_location=device)
    #     model.load_state_dict(ckpt["model"])
    #     # resume from the next epoch after the one stored
    #     start_epoch = ckpt.get("epoch", 0) + 1
    #     print(f"✅ Resuming from epoch {start_epoch}")
    # else:
    #     print("🆕 No checkpoint found, starting from scratch.")
    # # -------------------------------------------------------------------
    
    model = UNetDecoder(
        in_channels=cfg.embedding_channels,
        num_classes=cfg.num_classes,
        stride=cfg.embedding_stride,
        base=64,  # or 32 if you change it later
    ).to(device)

    opt, sch = make_optim(model)

    ckpt_last = Path(args.ckpt_dir) / "last_base64.pt"
    start_epoch = 1
    best_val = 0.0
    global_step = 0

    if ckpt_last.exists():
        print(f"🔄 Found checkpoint at {ckpt_last}, loading...")
        ckpt = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ckpt["model"])

        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "sch" in ckpt:
            sch.load_state_dict(ckpt["sch"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", 0.0)
        global_step = ckpt.get("global_step", 0)

        print(f"✅ Resuming from epoch {start_epoch}, best_val={best_val:.4f}, global_step={global_step}")
    else:
        print("🆕 No checkpoint found, starting from scratch.")


    train_loader = make_loader(
        Path(args.manifest),
        Path(args.train_split),
        Path(args.embeddings_dir),
        args.batch_size,
        shuffle=True,
        augment=False,
        num_workers=cfg.num_workers,
    )
    val_loader = make_loader(
        Path(args.manifest),
        Path(args.val_split),
        Path(args.embeddings_dir),
        batch_size=1,
        shuffle=False,
        augment=False,
        num_workers=cfg.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # global_step, best_val already initialized above
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Debug: print current LR
        cur_lr = opt.param_groups[0]["lr"]
        print(f"[Epoch {epoch}] LR = {cur_lr:.6g}")

        for batch in train_loader:
            msk = batch["mask"].to(device)
            emb = batch["embedding"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(emb)
                loss, dice = step_metrics(logits, msk)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

           # log_scalar(writer, "train/loss", loss.item(), global_step)
           # log_scalar(writer, "train/dice", dice, global_step)
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
                _, d = step_metrics(logits, msk)
                dices.append(d)
        val_dice = sum(dices) / max(1, len(dices))
        # log_scalar(writer, "val/dice", val_dice, epoch)

        # if epoch == 1 or val_dice > best_val:
        #     best_val = val_dice
        #     save_ckpt(model, epoch, Path(args.ckpt_dir) / "best_base64.pt")
        # save_ckpt(model, epoch, Path(args.ckpt_dir) / "last_base64.pt")
        # print(f"Epoch {epoch}: val Dice={val_dice:.4f} (best={best_val:.4f})")
                # ---- Checkpoint saving ----
        ckpt_payload = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sch": sch.state_dict(),
            "epoch": epoch,
            "best_val": float(best_val),
            "global_step": int(global_step),
        }

        last_path = Path(args.ckpt_dir) / "last_base64.pt"
        torch.save(ckpt_payload, last_path)

        if epoch == 1 or val_dice > best_val:
            best_val = val_dice
            best_path = Path(args.ckpt_dir) / "best_base64.pt"
            torch.save(ckpt_payload, best_path)

        print(f"Epoch {epoch}: val Dice={val_dice:.4f} (best={best_val:.4f})")



if __name__ == "__main__":
    main()
