# # from __future__ import annotations

# # import argparse, os
# # import torch
# # from torch.utils.tensorboard import SummaryWriter
# # from pathlib import Path

# # from src.config import cfg
# # from src.unet_decoder import UNetDecoder
# # from src.dataset import make_loader
# # from src.train_utils import make_optim, save_ckpt, log_scalar, step_metrics

# # import torch.backends.cudnn as cudnn
# # cudnn.benchmark = True

# # def parse_args():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--manifest", type=str, default=str(cfg.preprocessed_dir / "manifest.csv"))
# #     ap.add_argument("--train_split", type=str, default=str(cfg.splits_dir / "train.csv"))
# #     ap.add_argument("--val_split", type=str, default=str(cfg.splits_dir / "val.csv"))
# #     ap.add_argument("--embeddings_dir", type=str, default=str(cfg.embeddings_dir))
# #     ap.add_argument("--epochs", type=int, default=cfg.num_epochs)
# #     ap.add_argument("--batch_size", type=int, default=cfg.batch_size)
# #     ap.add_argument("--lr", type=float, default=cfg.learning_rate)
# #     ap.add_argument("--ckpt_dir", type=str, default=str(cfg.ckpt_dir))
# #     ap.add_argument("--log_dir", type=str, default=str(cfg.log_dir))
# #     ap.add_argument("--amp", action="store_true")
# #     return ap.parse_args()

# # def main():
# #     args = parse_args()
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     #writer = SummaryWriter(args.log_dir)

# #     # model = UNetDecoder(
# #     #     in_channels=cfg.embedding_channels,
# #     #     num_classes=cfg.num_classes,
# #     #     stride=cfg.embedding_stride,
# #     #     base=64 # or 32 if you change it later
# #     # ).to(device)
# #     # opt, sch = make_optim(model)

# #     # # ---------- NEW: resume from last checkpoint if it exists ----------
# #     # ckpt_last = Path(args.ckpt_dir) / "last_base64.pt"
# #     # start_epoch = 1
# #     # best_val = 0.0
# #     # global_step = 0

# #     # if ckpt_last.exists():
# #     #     print(f"🔄 Found checkpoint at {ckpt_last}, loading...")
# #     #     ckpt = torch.load(ckpt_last, map_location=device)
# #     #     model.load_state_dict(ckpt["model"])
# #     #     # resume from the next epoch after the one stored
# #     #     start_epoch = ckpt.get("epoch", 0) + 1
# #     #     print(f"✅ Resuming from epoch {start_epoch}")
# #     # else:
# #     #     print("🆕 No checkpoint found, starting from scratch.")
# #     # # -------------------------------------------------------------------
    
# #     model = UNetDecoder(
# #         in_channels=cfg.embedding_channels,
# #         num_classes=cfg.num_classes,
# #         stride=cfg.embedding_stride,
# #         base=64,  # or 32 if you change it later
# #     ).to(device)

# #     opt, sch = make_optim(model)

# #     ckpt_last = Path(args.ckpt_dir) / "last_base64.pt"
# #     start_epoch = 1
# #     best_val = 0.0
# #     global_step = 0

# #     if ckpt_last.exists():
# #         print(f"🔄 Found checkpoint at {ckpt_last}, loading...")
# #         ckpt = torch.load(ckpt_last, map_location=device)
# #         model.load_state_dict(ckpt["model"])

# #         if "opt" in ckpt:
# #             opt.load_state_dict(ckpt["opt"])
# #         if "sch" in ckpt:
# #             sch.load_state_dict(ckpt["sch"])

# #         start_epoch = ckpt.get("epoch", 0) + 1
# #         best_val = ckpt.get("best_val", 0.0)
# #         global_step = ckpt.get("global_step", 0)

# #         print(f"✅ Resuming from epoch {start_epoch}, best_val={best_val:.4f}, global_step={global_step}")
# #     else:
# #         print("🆕 No checkpoint found, starting from scratch.")


# #     train_loader = make_loader(
# #         Path(args.manifest),
# #         Path(args.train_split),
# #         Path(args.embeddings_dir),
# #         args.batch_size,
# #         shuffle=True,
# #         augment=False,
# #         num_workers=cfg.num_workers,
# #     )
# #     val_loader = make_loader(
# #         Path(args.manifest),
# #         Path(args.val_split),
# #         Path(args.embeddings_dir),
# #         batch_size=1,
# #         shuffle=False,
# #         augment=False,
# #         num_workers=cfg.num_workers,
# #     )

# #     scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

# #     # global_step, best_val already initialized above
# #     for epoch in range(start_epoch, args.epochs + 1):
# #         model.train()
# #         # Debug: print current LR
# #         cur_lr = opt.param_groups[0]["lr"]
# #         print(f"[Epoch {epoch}] LR = {cur_lr:.6g}")

# #         for batch in train_loader:
# #             msk = batch["mask"].to(device)
# #             emb = batch["embedding"].to(device)

# #             opt.zero_grad(set_to_none=True)
# #             with torch.cuda.amp.autocast(enabled=args.amp):
# #                 logits = model(emb)
# #                 loss, dice = step_metrics(logits, msk)

# #             scaler.scale(loss).backward()
# #             scaler.step(opt)
# #             scaler.update()

# #            # log_scalar(writer, "train/loss", loss.item(), global_step)
# #            # log_scalar(writer, "train/dice", dice, global_step)
# #             global_step += 1

# #         sch.step()

# #         # ---- Validation ----
# #         model.eval()
# #         dices = []
# #         with torch.no_grad():
# #             for batch in val_loader:
# #                 msk = batch["mask"].to(device)
# #                 emb = batch["embedding"].to(device)
# #                 logits = model(emb)
# #                 _, d = step_metrics(logits, msk)
# #                 dices.append(d)
# #         val_dice = sum(dices) / max(1, len(dices))
# #         # log_scalar(writer, "val/dice", val_dice, epoch)

# #         # if epoch == 1 or val_dice > best_val:
# #         #     best_val = val_dice
# #         #     save_ckpt(model, epoch, Path(args.ckpt_dir) / "best_base64.pt")
# #         # save_ckpt(model, epoch, Path(args.ckpt_dir) / "last_base64.pt")
# #         # print(f"Epoch {epoch}: val Dice={val_dice:.4f} (best={best_val:.4f})")
# #                 # ---- Checkpoint saving ----
# #         ckpt_payload = {
# #             "model": model.state_dict(),
# #             "opt": opt.state_dict(),
# #             "sch": sch.state_dict(),
# #             "epoch": epoch,
# #             "best_val": float(best_val),
# #             "global_step": int(global_step),
# #         }

# #         last_path = Path(args.ckpt_dir) / "last_base64.pt"
# #         torch.save(ckpt_payload, last_path)

# #         if epoch == 1 or val_dice > best_val:
# #             best_val = val_dice
# #             best_path = Path(args.ckpt_dir) / "best_base64.pt"
# #             torch.save(ckpt_payload, best_path)

# #         print(f"Epoch {epoch}: val Dice={val_dice:.4f} (best={best_val:.4f})")



# # if __name__ == "__main__":
# #     main()

# from __future__ import annotations

# import argparse
# from pathlib import Path
# import torch
# import torch.backends.cudnn as cudnn

# from src.config import cfg
# from src.dataset import make_image_loader
# from src.metrics import DiceBCELoss, dice_coef
# from src.medsam_baseline import build_sam_baseline
# from src.medsam_forward import sam_forward_logits

# cudnn.benchmark = True


# # ---------------------------------------------------------
# # Args
# # ---------------------------------------------------------
# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--manifest", type=str, required=True)
#     ap.add_argument("--train_split", type=str, required=True)
#     ap.add_argument("--val_split", type=str, required=True)
#     ap.add_argument("--sam_ckpt", type=str, required=True)
#     ap.add_argument("--ckpt_dir", type=str, required=True)

#     ap.add_argument("--epochs", type=int, default=cfg.num_epochs)
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--amp", action="store_true")

#     # new / important
#     ap.add_argument("--box_jitter", type=float, default=0.10)
#     ap.add_argument("--unfreeze_last_blocks", type=int, default=1)

#     return ap.parse_args()


# # ---------------------------------------------------------
# # Train
# # ---------------------------------------------------------
# def main():
#     args = parse_args()
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     ckpt_dir = Path(args.ckpt_dir)
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     # -----------------------------------------------------
#     # Build SAM (decoder + last encoder blocks trainable)
#     # -----------------------------------------------------
#     sam = build_sam_baseline(
#         ckpt_path=args.sam_ckpt,
#         model_type="vit_b",
#         device=device,
#         unfreeze_last_blocks=args.unfreeze_last_blocks,
#     )

#     # -----------------------------------------------------
#     # Loss (Dice-only is most stable here)
#     # -----------------------------------------------------
#     loss_fn = DiceBCELoss(dice_w=1.0, bce_w=0.0)

#     # -----------------------------------------------------
#     # Optimizer with param groups
#     # -----------------------------------------------------
#     decoder_params = []
#     encoder_params = []

#     for name, p in sam.named_parameters():
#         if not p.requires_grad:
#             continue
#         if "image_encoder" in name:
#             encoder_params.append(p)
#         else:
#             decoder_params.append(p)

#     optimizer = torch.optim.AdamW(
#         [
#             {"params": decoder_params, "lr": 1e-4},
#             {"params": encoder_params, "lr": 1e-5},
#         ],
#         weight_decay=1e-4,
#     )

#     scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

#     # -----------------------------------------------------
#     # Data loaders
#     # -----------------------------------------------------
#     train_loader = make_image_loader(
#         Path(args.manifest),
#         Path(args.train_split),
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers,
#     )

#     val_loader = make_image_loader(
#         Path(args.manifest),
#         Path(args.val_split),
#         batch_size=1,
#         shuffle=False,
#         num_workers=cfg.num_workers,
#     )

#     best_val = 0.0

#     # -----------------------------------------------------
#     # Epoch loop
#     # -----------------------------------------------------
#     for epoch in range(1, args.epochs + 1):
#         sam.train()
#         print(f"\n[Epoch {epoch}]")

#         # -----------------------------
#         # Training
#         # -----------------------------
#         for batch in train_loader:
#             # img = batch["image"].to(device)   # (B,3,1024,1024)
#             # msk = batch["mask"].to(device)    # (B,H,W)
#             img = batch["image"].to(device)
#             msk = batch["mask"].to(device)
            
#             # ⛔ skip empty slices
#             valid = (msk.sum(dim=(1, 2)) > 0)
            
#             if valid.sum() == 0:
#                 continue
            
#             img = img[valid]
#             msk = msk[valid]


#             optimizer.zero_grad(set_to_none=True)

#             with torch.cuda.amp.autocast(enabled=args.amp):
#                 logits = sam_forward_logits(
#                     sam,
#                     img,
#                     msk,
#                     training=True,
#                     box_jitter_frac=args.box_jitter,
#                 )
#                 loss = loss_fn(logits, msk)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#         # -----------------------------
#         # Validation
#         # -----------------------------
#         sam.eval()
#         dices = []

#         with torch.no_grad():
#             for batch in val_loader:
#                 img = batch["image"].to(device)
#                 msk = batch["mask"].to(device)
            
#                 # ⛔ skip empty GT slices in validation
#                 if msk.sum() == 0:
#                     continue
            
#                 logits = sam_forward_logits(
#                     sam,
#                     img,
#                     msk,
#                     training=False,
#                     box_jitter_frac=0.0,
#                 )
#                 dices.append(dice_coef(logits, msk))

#         val_dice = sum(dices) / max(1, len(dices))

#         # -----------------------------
#         # Checkpoint
#         # -----------------------------
#         ckpt = {
#             "epoch": epoch,
#             "val_dice": float(val_dice),
#             "mask_decoder": sam.mask_decoder.state_dict(),
#             "image_encoder": sam.image_encoder.state_dict(),
#             "unfreeze_last_blocks": args.unfreeze_last_blocks,
#             "box_jitter": args.box_jitter,
#         }

#         torch.save(ckpt, ckpt_dir / "last_sam.pt")

#         if epoch == 1 or val_dice > best_val:
#             best_val = val_dice
#             torch.save(ckpt, ckpt_dir / "best_sam.pt")

#         print(f"Epoch {epoch}: val Dice = {val_dice:.4f} (best = {best_val:.4f})")


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from src.config import cfg
from src.dataset import make_image_loader
from src.metrics import DiceBCELoss, dice_coef
from src.medsam_baseline import build_sam_baseline
from src.medsam_forward import sam_forward_logits

from src.boxnet import BoxUNet
from src.boxnet_utils import downsample_mask_to_256, boxnet_bce_loss, box_iou_from_logits

cudnn.benchmark = True


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--train_split", type=str, required=True)
    ap.add_argument("--val_split", type=str, required=True)
    ap.add_argument("--sam_ckpt", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=cfg.num_epochs)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=cfg.num_workers)
    ap.add_argument("--amp", action="store_true")

    # MedSAM prompting
    ap.add_argument("--box_jitter", type=float, default=0.10)
    ap.add_argument("--unfreeze_last_blocks", type=int, default=1)

    # BoxNet + prompt-less training
    ap.add_argument("--train_boxnet", action="store_true")
    ap.add_argument("--boxnet_lr", type=float, default=1e-3)
    ap.add_argument("--boxnet_weight", type=float, default=1.0)
    ap.add_argument("--boxnet_thresh", type=float, default=0.30)

    # prompt-less probability after epoch 1
    ap.add_argument("--promptless_prob", type=float, default=0.30)

    # validation
    ap.add_argument("--val_promptless", action="store_true")
    ap.add_argument("--exclude_empty_val", action="store_true")

    # resume
    ap.add_argument("--resume", action="store_true")

    return ap.parse_args()


def _extract_state_dict(ckpt_obj):
    """
    Accepts a variety of checkpoint formats and returns a plain state_dict.
    Works for:
      - state_dict directly (OrderedDict / dict of tensors)
      - {"state_dict": ...}
      - {"model": ...}
      - {"net": ...} etc (common patterns)
    """
    if not isinstance(ckpt_obj, dict):
        # rare, but just return as-is
        return ckpt_obj

    for key in ["state_dict", "model", "model_state_dict", "net", "sam", "weights"]:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]

    # If it already looks like a state_dict (tensor values), use it
    # Heuristic: at least one tensor value
    if any(torch.is_tensor(v) for v in ckpt_obj.values()):
        return ckpt_obj

    return ckpt_obj


def _strip_prefix_if_present(state_dict, prefix="module."):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_pretrained_sam_weights(sam, ckpt_path: str):
    print(f"🧠 Loading pretrained SAM from {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")

    # Most MedSAM/SAM weights have keys like:
    # image_encoder.*, prompt_encoder.*, mask_decoder.*
    missing, unexpected = sam.load_state_dict(sd, strict=False)

    # Print a short diagnostic (not too spammy)
    print(f"✅ Loaded pretrained SAM (strict=False). Missing={len(missing)} Unexpected={len(unexpected)}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last_sam.pt"
    best_path = ckpt_dir / "best_sam.pt"

    # -----------------------------
    # Build MedSAM (NO weights here)
    # -----------------------------
    sam = build_sam_baseline(
        ckpt_path=None,  # ❗ do not load here; we control loading below
        model_type="vit_b",
        device=device,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    )
    
    # -----------------------------
    # Build BoxNet (optional)
    # -----------------------------
    boxnet = BoxUNet(in_ch=3, base=32).to(device) if args.train_boxnet else None

    # -----------------------------
    # Loss
    # -----------------------------
    seg_loss_fn = DiceBCELoss(dice_w=1.0, bce_w=0.0)

    # -----------------------------
    # Optimizers
    # -----------------------------
    decoder_params = []
    encoder_params = []
    for name, p in sam.named_parameters():
        if not p.requires_grad:
            continue
        if "image_encoder" in name:
            encoder_params.append(p)
        else:
            decoder_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_params, "lr": 1e-4},
            {"params": encoder_params, "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )

    boxnet_opt = None
    if boxnet is not None:
        boxnet_opt = torch.optim.AdamW(boxnet.parameters(), lr=args.boxnet_lr, weight_decay=1e-4)

    # Keep your AMP usage; warning is harmless, but you can modernize later.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # -----------------------------
    # Resume OR load pretrained
    # -----------------------------
    start_epoch = 1
    best_val = 0.0

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)

        if "mask_decoder" in ckpt:
            sam.mask_decoder.load_state_dict(ckpt["mask_decoder"])
        if "image_encoder" in ckpt:
            sam.image_encoder.load_state_dict(ckpt["image_encoder"])
        if boxnet is not None and ckpt.get("boxnet") is not None:
            boxnet.load_state_dict(ckpt["boxnet"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if boxnet_opt is not None and ckpt.get("boxnet_opt") is not None:
            boxnet_opt.load_state_dict(ckpt["boxnet_opt"])

        if args.amp and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", 0.0))
        print(f"🔄 Resuming from {last_path} @ epoch {start_epoch} (best_val={best_val:.4f})")

    else:
        # ✅ robust pretrained loader (fixes your KeyError)
        load_pretrained_sam_weights(sam, args.sam_ckpt)
        print("🆕 Starting from pretrained SAM (no resume).")

    # -----------------------------
    # Data loaders
    # -----------------------------
    train_loader = make_image_loader(
        Path(args.manifest),
        Path(args.train_split),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = make_image_loader(
        Path(args.manifest),
        Path(args.val_split),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        sam.train()
        if boxnet is not None:
            boxnet.train()

        print(f"\n[Epoch {epoch}]")

        for batch in train_loader:
            img = batch["image"].to(device)  # (B,3,H,W)
            msk = batch["mask"].to(device)   # (B,H,W) or (B,1,H,W)
            if msk.dim() == 4:
                msk = msk[:, 0]

            optimizer.zero_grad(set_to_none=True)
            if boxnet_opt is not None:
                boxnet_opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                total_loss = 0.0

                # (A) BoxNet loss (ALL slices, including empty)
                if boxnet is not None:
                    img_256 = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
                    tgt_256 = downsample_mask_to_256(msk)      # (B,1,256,256)
                    bn_logits = boxnet(img_256)                # (B,1,256,256)
                    bn_loss = boxnet_bce_loss(bn_logits, tgt_256)
                    total_loss = total_loss + (args.boxnet_weight * bn_loss)

                # (B) MedSAM seg loss (ONLY positive slices)
                valid = (msk.sum(dim=(1, 2)) > 0)
                if valid.any():
                    img_pos = img[valid]
                    msk_pos = msk[valid]

                    do_promptless = (
                        (epoch > 1)
                        and (boxnet is not None)
                        and (args.promptless_prob > 0.0)
                        and (float(torch.rand(()).item()) < args.promptless_prob)
                    )

                    if do_promptless:
                        logits = sam_forward_logits(
                            sam,
                            img_pos,
                            gt_masks=None,
                            training=True,
                            box_jitter_frac=args.box_jitter,
                            boxnet=boxnet,
                            boxnet_thresh=args.boxnet_thresh,
                        )
                    else:
                        logits = sam_forward_logits(
                            sam,
                            img_pos,
                            gt_masks=msk_pos,
                            training=True,
                            box_jitter_frac=args.box_jitter,
                            boxnet=None,
                        )

                    seg_loss = seg_loss_fn(logits, msk_pos)
                    total_loss = total_loss + seg_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)

            if boxnet_opt is not None:
                scaler.step(boxnet_opt)

            scaler.update()

        # -----------------------------
        # Validation Dice
        # -----------------------------
        sam.eval()
        if boxnet is not None:
            boxnet.eval()

        dices = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                if msk.dim() == 4:
                    msk = msk[:, 0]

                if args.exclude_empty_val and (msk.sum() == 0):
                    continue

                if args.val_promptless and (boxnet is not None):
                    logits = sam_forward_logits(
                        sam,
                        img,
                        gt_masks=None,
                        training=False,
                        box_jitter_frac=0.0,
                        boxnet=boxnet,
                        boxnet_thresh=args.boxnet_thresh,
                    )
                else:
                    logits = sam_forward_logits(
                        sam,
                        img,
                        gt_masks=msk,
                        training=False,
                        box_jitter_frac=0.0,
                        boxnet=None,
                    )

                dices.append(dice_coef(logits, msk))

        val_dice = float(sum(dices) / max(1, len(dices)))
        print(f"Epoch {epoch}: val Dice = {val_dice:.4f} (best = {best_val:.4f})")

        # -----------------------------
        # Save checkpoints (MedSAM + BoxNet together)
        # -----------------------------
        ckpt = {
            "epoch": epoch,
            "val_dice": val_dice,
            "best_val": float(best_val),
            "mask_decoder": sam.mask_decoder.state_dict(),
            "image_encoder": sam.image_encoder.state_dict(),
            "unfreeze_last_blocks": args.unfreeze_last_blocks,
            "box_jitter": args.box_jitter,
            "boxnet": None if boxnet is None else boxnet.state_dict(),
            "boxnet_thresh": args.boxnet_thresh,
            "optimizer": optimizer.state_dict(),
            "boxnet_opt": None if boxnet_opt is None else boxnet_opt.state_dict(),
            "scaler": scaler.state_dict() if args.amp else None,
        }

        torch.save(ckpt, last_path)

        if epoch == 1 or val_dice > best_val:
            best_val = val_dice
            ckpt["best_val"] = float(best_val)
            torch.save(ckpt, best_path)

    print(f"\n🎯 Done. last: {last_path} | best: {best_path}")


if __name__ == "__main__":
    main()
