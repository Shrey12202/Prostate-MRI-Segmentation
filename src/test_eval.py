import torch
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from src.metrics import DiceLoss
from src.unet_decoder import UNetDecoder
from src.config import cfg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--embeddings_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained decoder
    model = UNetDecoder(
        in_channels=cfg.embedding_channels,
        num_classes=cfg.num_classes,
        stride=cfg.embedding_stride
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    df = pd.read_csv(args.manifest)
    dice_fn = DiceLoss()

    dice_scores = []

    for _, row in df.iterrows():
        stem = Path(row["image"]).stem
        emb_path = Path(args.embeddings_dir) / f"{stem}.pt"
        mask_path = Path(row["mask"])

        if not emb_path.exists():
            print(f"[WARN] Missing embedding for {stem}")
            continue

        # Load embedding + mask
        emb = torch.load(emb_path, map_location=device).unsqueeze(0)
        msk = np.array(Image.open(mask_path))
        msk = torch.tensor(msk / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(emb.to(device))
            probs = torch.sigmoid(logits)

        dice = float(1 - dice_fn(probs, msk))
        dice_scores.append(dice)

    if dice_scores:
        print("=======================================")
        print(" TEST SET EVALUATION ")
        print("=======================================")
        print(f"Mean Dice: {sum(dice_scores)/len(dice_scores):.4f}")
        print(f"Slices evaluated: {len(dice_scores)}")
    else:
        print("❌ No test slices found.")


if __name__ == "__main__":
    main()
