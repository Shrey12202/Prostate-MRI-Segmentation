from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import cfg
from src.dataset import make_image_loader
from src.metrics import DiceBCELoss

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -----------------------------
# utils
# -----------------------------
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

    x0 = torch.min(b[:, 0], b[:, 2])
    x1 = torch.max(b[:, 0], b[:, 2])
    y0 = torch.min(b[:, 1], b[:, 3])
    y1 = torch.max(b[:, 1], b[:, 3])
    return torch.stack([x0, y0, x1, y1], dim=1)


def img_tensor_to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """
    img: (3,H,W) float tensor, typically normalized [0,1] or [0,255]
    returns uint8 (H,W,3)
    """
    x = img.detach().cpu()
    if x.max() <= 1.5:
        x = x * 255.0
    x = x.clamp(0, 255).byte()
    return x.permute(1, 2, 0).contiguous().numpy()


# -----------------------------
# args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--train_split", type=str, required=True)
    ap.add_argument("--val_split", type=str, required=True)

    ap.add_argument("--sam2_ckpt", type=str, required=True)
    ap.add_argument("--sam2_cfg", type=str, default="configs/sam2.1_hiera_t512")

    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=cfg.num_epochs)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--box_jitter", type=float, default=0.10)

    # IMPORTANT: predictor path runs per-image anyway; this just controls dataloader
    ap.add_argument("--num_workers", type=int, default=1)

    ap.add_argument("--lr", type=float, default=1e-5)
    return ap.parse_args()


# -----------------------------
# train
# -----------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build SAM2 / MedSAM2 model
    sam = build_sam2(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_ckpt,
        device=device,
        mode="train",
    )

    # Predictor wrapper (the correct way for MedSAM2)
    predictor = SAM2ImagePredictor(sam)

    # Freeze everything by default; train mask decoder + prompt encoder only
    for p in sam.parameters():
        p.requires_grad = False
    for name, p in sam.named_parameters():
        if ("mask_decoder" in name) or ("prompt_encoder" in name) or ("sam_mask_decoder" in name) or ("sam_prompt_encoder" in name):
            p.requires_grad = True

    trainable = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Loss (logits not available directly; we train on probs as a fallback)
    # If you want pure logits, we can adjust using internal predictor calls later.
    loss_fn = DiceBCELoss(dice_w=1.0, bce_w=0.0)

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

    best_val = 0.0
    start_epoch = 1
    last_path = ckpt_dir / "last_medsam2_predictor.pt"

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        sam.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["opt"])
        best_val = float(ckpt.get("best_val", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"🔄 Resumed from {last_path} @ epoch {start_epoch} (best={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        sam.train()
        print(f"\n[Epoch {epoch}]")

        # -------- TRAIN ----------
        for batch in train_loader:
            img = batch["image"].to(device)  # (B,3,1024,1024) usually
            msk = batch["mask"].to(device)   # (B,1024,1024)

            # Predictor config is 512 => resize to 512
            img = F.interpolate(img, size=(512, 512), mode="bilinear", align_corners=False)
            msk = F.interpolate(msk.unsqueeze(1).float(), size=(512, 512), mode="nearest").squeeze(1)

            B, H, W = msk.shape
            boxes = masks_to_boxes_xyxy((msk > 0).float())
            boxes = jitter_boxes_xyxy(boxes, H=H, W=W, frac=args.box_jitter)

            optimizer.zero_grad(set_to_none=True)

            # Predictor runs per-image; we accumulate loss over batch
            losses = []
            with torch.cuda.amp.autocast(enabled=args.amp):
                for b in range(B):
                    img_np = img_tensor_to_uint8_hwc(img[b])              # uint8 HWC
                    gt = msk[b]                                           # (H,W)
                    box = boxes[b].detach().cpu().numpy().astype(np.float32)

                    predictor.set_image(img_np)

                    masks, _, _ = predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
                    # masks: (1,H,W) numpy bool/float
                    pred = torch.from_numpy(masks[0]).to(device).float()  # (H,W)

                    # Loss expects logits; we feed probs (still works for Dice-only loss)
                    losses.append(loss_fn(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0)))

                loss = torch.stack(losses).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # -------- VAL ----------
        sam.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)

                img = F.interpolate(img, size=(512, 512), mode="bilinear", align_corners=False)
                msk = F.interpolate(msk.unsqueeze(1).float(), size=(512, 512), mode="nearest").squeeze(1)

                H, W = msk.shape[-2], msk.shape[-1]
                box = masks_to_boxes_xyxy((msk > 0).float())[0].cpu().numpy().astype(np.float32)

                img_np = img_tensor_to_uint8_hwc(img[0])
                predictor.set_image(img_np)

                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                pred = torch.from_numpy(masks[0]).to(device).float()

                gt = msk[0].float()
                inter = (pred * gt).sum()
                union = pred.sum() + gt.sum()
                dice = (2 * inter + 1e-6) / (union + 1e-6)
                dices.append(dice.item())

        val_dice = float(sum(dices) / max(1, len(dices)))

        payload = {
            "epoch": epoch,
            "best_val": float(best_val),
            "val_dice": float(val_dice),
            "model": sam.state_dict(),
            "opt": optimizer.state_dict(),
            "sam2_cfg": args.sam2_cfg,
            "sam2_ckpt": args.sam2_ckpt,
            "box_jitter": args.box_jitter,
        }

        torch.save(payload, last_path)
        if epoch == 1 or val_dice > best_val:
            best_val = val_dice
            torch.save(payload, ckpt_dir / "best_medsam2_predictor.pt")

        print(f"Epoch {epoch}: val Dice = {val_dice:.4f} (best = {best_val:.4f})")


if __name__ == "__main__":
    main()
