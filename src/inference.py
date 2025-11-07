from __future__ import annotations

import argparse, os
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from config import cfg
from unet_decoder import UNetDecoder

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", type=str, default=str(cfg.embeddings_dir))
    ap.add_argument("--preprocessed_dir", type=str, default=str(cfg.preprocessed_dir))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="experiments/preds")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = UNetDecoder(in_channels=cfg.embedding_channels, num_classes=cfg.num_classes, stride=cfg.embedding_stride).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    img_dir = Path(args.preprocessed_dir) / "images"
    for ip in sorted(img_dir.glob("*.png")):
        stem = ip.stem
        emb_p = Path(args.embeddings_dir) / f"{stem}.pt"
        if not emb_p.exists():
            print(f"Missing embedding for {stem}")
            continue
        emb = torch.load(emb_p, map_location=device).unsqueeze(0)  # (1,C,h,w)

        with torch.no_grad():
            logits = model(emb)
            if cfg.num_classes > 1:
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype("uint8")*255
            else:
                pred = (logits.sigmoid() > 0.5).long().squeeze(0).squeeze(0).cpu().numpy().astype("uint8")*255

        Image.fromarray(pred).save(out_dir/f"{stem}.png")

if __name__ == "__main__":
    main()
