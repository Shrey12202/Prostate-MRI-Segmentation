"""Extract slice-level embeddings from the MedSAM image encoder.
Requires the `segment_anything`-style model for MedSAM (ViT-based).
This script saves per-slice feature maps as .pt tensors with shape (C, H//stride, W//stride).

NOTE: You must install the MedSAM package/weights separately and set `--checkpoint`.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from segment_anything import sam_model_registry
from config import cfg

def load_medsam_model(checkpoint: str):
    """Load MedSAM encoder. Replace with the correct import for your MedSAM fork.
    Expected to return a module with a method `image_encoder(x)` -> feature map (B,C,h,w).
    """
    print(f"Loading MedSAM weights from: {cfg.medsam_ckpt}")
    # sam = sam_model_registry[cfg.medsam_backbone](checkpoint=str(cfg.medsam_ckpt))
    # sam.eval()
    # for p in sam.parameters():
    #     p.requires_grad_(False)
    # return sam

    #CPU
    sam = sam_model_registry[cfg.medsam_backbone](checkpoint=None)
    state_dict = torch.load(str(cfg.medsam_ckpt), map_location='cpu')
    sam.load_state_dict(state_dict)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    return sam

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--preprocessed_dir", type=str, default=str(cfg.preprocessed_dir))
#     ap.add_argument("--embeddings_dir", type=str, default=str(cfg.embeddings_dir))
#     ap.add_argument("--checkpoint", type=str, required=True, help="Path to MedSAM checkpoint (SAM-style)" )
#     ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     args = ap.parse_args()

#     pre_dir = Path(args.preprocessed_dir)
#     emb_dir = Path(args.embeddings_dir)
#     emb_dir.mkdir(parents=True, exist_ok=True)

#     sam = load_medsam_model(args.checkpoint).to(args.device)

#     img_dir = pre_dir / "images"
#     imgs = sorted(list(img_dir.glob("*.png")))
#     for ip in tqdm(imgs, desc="Extracting embeddings"):
#         im = Image.open(ip).convert("RGB")  # SAM expects 3-ch; replicate
#         # replicate gray to 3-ch
#         g = TF.pil_to_tensor(im.convert("L")).float() / 255.0
#         img3 = torch.cat([g,g,g], dim=0)  # (3,H,W)
#         img3 = TF.resize(img3, cfg.image_size)  # ensure size match
#         x = img3.unsqueeze(0).to(args.device)   # (1,3,H,W)

#         with torch.no_grad():
#             feat = sam.image_encoder(x)  # (1,C,h,w)
#         feat = feat.squeeze(0).cpu()
#         out_p = emb_dir / f"{ip.stem}.pt"
#         torch.save(feat, out_p)

# if __name__ == "__main__":
#     main()
