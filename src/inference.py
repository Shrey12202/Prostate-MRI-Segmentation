from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
import pandas as pd
from src.config import cfg
from src.unet_decoder import UNetDecoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--embeddings_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load UNet decoder
    model = UNetDecoder(
        in_channels=cfg.embedding_channels,
        num_classes=cfg.num_classes,
        stride=cfg.embedding_stride
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # Read slice-level test manifest
    df = pd.read_csv(args.manifest)

    for _, row in df.iterrows():
        image_path = Path(row["image"])
        stem = image_path.stem
        emb_path = Path(args.embeddings_dir) / f"{stem}.pt"

        if not emb_path.exists():
            print(f"[WARN] Missing embedding: {emb_path}")
            continue

        emb = torch.load(emb_path, map_location=device).unsqueeze(0)  # (1,C,h,w)

        with torch.no_grad():
            logits = model(emb)
            pred = (logits.sigmoid() > 0.5).long().squeeze().cpu().numpy().astype("uint8") * 255

        Image.fromarray(pred).save(out_dir / f"{stem}.png")

    print(f"✅ Inference complete. Predictions saved to: {out_dir}")


if __name__ == "__main__":
    main()
