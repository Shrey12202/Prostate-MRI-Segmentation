"""
Extract slice-level embeddings from the MedSAM image encoder.
Supports arbitrary image sizes by interpolating positional embeddings.
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


# -------------------------------------------------------------
# Positional Embedding Interpolation (Key Fix)
# -------------------------------------------------------------
def resize_pos_embed(pos_embed: torch.Tensor, new_h: int, new_w: int):
    """
    pos_embed: (1, C, H_old, W_old)
    returns resized pos_embed: (1, C, H_new, W_new)
    """
    C = pos_embed.shape[1]
    pos_embed = torch.nn.functional.interpolate(
        pos_embed,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    return pos_embed


def load_medsam_model(checkpoint: str, device="cuda"):
    """Load MedSAM encoder and interpolate positional embeddings."""
    print(f"Loading MedSAM weights from: {checkpoint}")

    sam = sam_model_registry[cfg.medsam_backbone](checkpoint=None).to(device)

    state_dict = torch.load(str(checkpoint), map_location=device, weights_only=False)
    sam.load_state_dict(state_dict)

    # ----------------------------
    # FIX: interpolate pos_embed
    # ----------------------------
    old_pos = sam.image_encoder.pos_embed  # shape (1, 256, 64, 64)
    _, C, H_old, W_old = old_pos.shape

    H_new = cfg.image_size[0] // cfg.embedding_stride
    W_new = cfg.image_size[1] // cfg.embedding_stride

    if (H_old, W_old) != (H_new, W_new):
        print(f"🔧 Resizing pos_embed: {H_old}x{W_old} → {H_new}x{W_new}")
        new_pos = resize_pos_embed(old_pos, H_new, W_new)
        sam.image_encoder.pos_embed = torch.nn.Parameter(new_pos)

    sam.eval()
    for p in sam.parameters():
        p.requires_grad_(False)

    return sam


# -------------------------------------------------------------
# Main embedding extraction
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed_dir", type=str, default=str(cfg.preprocessed_dir))
    ap.add_argument("--embeddings_dir", type=str, default=str(cfg.embeddings_dir))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device
    pre_dir = Path(args.preprocessed_dir)
    emb_dir = Path(args.embeddings_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load MedSAM encoder with pos_embed fix
    sam = load_medsam_model(args.checkpoint, device=device)

    # Preprocessed PNG slices
    img_dir = pre_dir / "images"
    imgs = sorted(img_dir.glob("*.png"))
    print(f"Found {len(imgs)} slices to embed.")

    for ip in tqdm(imgs, desc="Extracting embeddings"):
        out_p = emb_dir / f"{ip.stem}.pt"
        if out_p.exists():
            continue

        im = Image.open(ip).convert("L")
        g = TF.pil_to_tensor(im).float() / 255.0
        img3 = torch.cat([g, g, g], dim=0)
        img3 = TF.resize(img3, cfg.image_size)
        x = img3.unsqueeze(0).to(device)

        with torch.no_grad():
            feat = sam.image_encoder(x)

        feat = feat.squeeze(0).cpu()
        torch.save(feat, out_p)

    print(f"✅ Done. Embeddings saved to: {emb_dir}")


if __name__ == "__main__":
    main()
