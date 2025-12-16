from pathlib import Path
import argparse
import torch
import numpy as np
import pandas as pd

from src.dataset import ImageMaskDataset
from src.metrics import dice_coef
from src.medsam2_forward_predictor import infer_mask_from_box


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val_split", required=True)
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--sam2_cfg", required=True)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageMaskDataset(
        manifest_csv=Path(args.manifest),
        split_csv=Path(args.val_split),
    )

    dices = []

    BATCH = 8
    for i in range(0, len(dataset), BATCH):
        batch = [dataset[j] for j in range(i, min(i+BATCH, len(dataset)))]
        
        imgs = torch.stack([b["image"] for b in batch]).to(device)
        gts  = torch.stack([b["mask"]  for b in batch]).to(device)
    
        logits = infer_mask_from_box_batch(
            imgs,
            gts,
            sam2_ckpt=args.sam2_ckpt,
            sam2_cfg=args.sam2_cfg,
            device=device,
        )

       # logits: force (1,1,H,W)
        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 3:
            logits = logits.unsqueeze(1)
        elif logits.ndim != 4:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        
        # gt: force (1,1,H,W)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0).unsqueeze(0)
        elif gt.ndim == 3:
            gt = gt.unsqueeze(1)
        elif gt.ndim != 4:
            raise RuntimeError(f"Unexpected GT shape: {gt.shape}")
        
        d = dice_coef(logits, gt)
        dices.append(float(d))



        # d = dice_coef(logits.unsqueeze(0), gt.unsqueeze(0))
        # dices.append(d)

    mean_dice = float(np.mean(dices))

    pd.DataFrame({"dice": dices}).to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Mean Dice: {mean_dice:.4f}\n")

    print(f"✅ MedSAM2 VAL Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
