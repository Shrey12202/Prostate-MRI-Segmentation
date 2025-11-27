"""
Extract slice-level embeddings from the MedSAM image encoder.
Saves one .pt embedding per preprocessed slice (t2_sXXX.png → t2_sXXX.pt).
"""

from __future__ import annotations
from pathlib import Path
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from segment_anything import sam_model_registry

from src.config import cfg


def load_medsam_model(checkpoint: str):
    """Load MedSAM encoder on CPU."""
    print(f"Loading MedSAM weights from: {checkpoint}")

    sam = sam_model_registry[cfg.medsam_backbone](checkpoint=None)
    state_dict = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    sam.load_state_dict(state_dict)
    sam.eval()

    for p in sam.parameters():
        p.requires_grad_(False)
    return sam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed_dir", type=str, default=str(cfg.preprocessed_dir))
    ap.add_argument("--embeddings_dir", type=str, default=str(cfg.embeddings_dir))
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to MedSAM checkpoint (.pth)")
    ap.add_argument("--device", type=str, default="cpu")  # CPU on NYU HPC frontend
    args = ap.parse_args()

    pre_dir = Path(args.preprocessed_dir)
    emb_dir = Path(args.embeddings_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load MedSAM encoder
    sam = load_medsam_model(args.checkpoint).to(args.device)

    # Preprocessed PNG slices
    img_dir = pre_dir / "images"
    imgs = sorted(img_dir.glob("*.png"))

    print(f"Found {len(imgs)} slices to embed.")

    for ip in tqdm(imgs, desc="Extracting embeddings"):
        out_p = emb_dir / f"{ip.stem}.pt"
    
        # ---- Skip if embedding already exists ----
        if out_p.exists():
            continue
    
        # ---- Only compute if needed ----
        im = Image.open(ip).convert("L")
        g = TF.pil_to_tensor(im).float() / 255.0
        img3 = torch.cat([g, g, g], dim=0)
        img3 = TF.resize(img3, cfg.image_size)
        x = img3.unsqueeze(0).to(args.device)
    
        with torch.no_grad():
            feat = sam.image_encoder(x)
        feat = feat.squeeze(0).cpu()
    
        torch.save(feat, out_p)


    print(f"Done. Embeddings saved to: {emb_dir}")


if __name__ == "__main__":
    main()
