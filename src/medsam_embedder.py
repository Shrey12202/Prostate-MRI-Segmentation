"""
Extract slice-level embeddings from the MedSAM image encoder.
Saves one .pt embedding per preprocessed slice:
    /.../images/068_s012.png -> /.../embeddings/068_s012.pt

Each saved .pt is a torch tensor of shape (256, 64, 64).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from segment_anything import sam_model_registry
from src.config import cfg


def load_medsam_model(ckpt_path: Path, device: str):
    """
    Load MedSAM ViT-B model and freeze it.
    IMPORTANT: we do NOT touch image_encoder.pos_embed at all.
    """
    print(f"Loading MedSAM weights from: {ckpt_path}")

    # Build SAM backbone (without loading weights yet)
    sam = sam_model_registry[cfg.medsam_backbone](checkpoint=None)

    # Load official MedSAM weights (we trust this file)
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sam.load_state_dict(state_dict, strict=True)

    sam.to(device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad_(False)

    # Optional sanity print
    pe = sam.image_encoder.pos_embed
    print("pos_embed shape:", tuple(pe.shape))  # expect (1, 64, 64, 768)

    return sam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preprocessed_dir",
        type=str,
        default=str(cfg.preprocessed_dir),
        help="Directory with preprocessed PNG slices (contains images/ and masks/).",
    )
    ap.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(cfg.embeddings_dir),
        help="Output directory for per-slice .pt embeddings.",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(cfg.medsam_ckpt),
        help="Path to medsam_vit_b.pth",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for MedSAM inference.",
    )

    args = ap.parse_args()

    pre_dir = Path(args.preprocessed_dir)
    emb_dir = Path(args.embeddings_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load MedSAM
    sam = load_medsam_model(Path(args.checkpoint), args.device)

    # Discover preprocessed slices
    img_dir = pre_dir / "images"
    imgs = sorted(img_dir.glob("*.png"))
    print(f"Found {len(imgs)} slices to embed in: {img_dir}")

    for ip in tqdm(imgs, desc="Extracting embeddings"):
        out_p = emb_dir / f"{ip.stem}.pt"

        # Skip if already embedded
        if out_p.exists():
            continue

        # Load slice as grayscale
        im = Image.open(ip).convert("L")
        g = TF.pil_to_tensor(im).float() / 255.0  # (1, H, W)

        # MedSAM expects 3-channel input; replicate the single channel
        img3 = torch.cat([g, g, g], dim=0)  # (3, H, W)

        # Resize to MedSAM's native resolution (1024x1024)
        img3 = TF.resize(img3, cfg.image_size)  # (3, 1024, 1024) if cfg.image_size=(1024,1024)

        x = img3.unsqueeze(0).to(args.device)   # (1, 3, 1024, 1024)

        # Extract encoder features
        with torch.no_grad():
            feat = sam.image_encoder(x)  # (1, 256, 64, 64)

        feat = feat.squeeze(0).cpu()    # (256, 64, 64)
        torch.save(feat, out_p)

    print(f"✅ Done. Embeddings saved to: {emb_dir}")


if __name__ == "__main__":
    main()
