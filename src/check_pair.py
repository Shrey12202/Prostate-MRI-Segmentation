import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ---- CONFIG ----
emb_dir = Path("/scratch/sbv2019/mri/experiments/embeddings_full")
img_dir = Path("/scratch/sbv2019/mri/experiments/preprocessed_full/images")
mask_dir = Path("/scratch/sbv2019/mri/experiments/preprocessed_full/masks")

stem = sys.argv[1] if len(sys.argv) > 1 else None
if stem is None:
    print("Usage: python check_pair.py <stem>")
    sys.exit(1)

emb_path = emb_dir / f"{stem}.pt"
img_path = img_dir / f"{stem}.png"
mask_path = mask_dir / f"{stem}.png"

# ---- Exists? ----
print("\n=== FILE CHECK ===")
print("Embedding:", emb_path.exists(), emb_path)
print("Image:    ", img_path.exists(), img_path)
print("Mask:     ", mask_path.exists(), mask_path)

# ---- Load data ----
emb = torch.load(emb_path, map_location="cpu")
emb = emb.float().numpy()                       # (C, h, w)
emb_mean = emb.mean(axis=0)                     # (h, w)

img = np.array(Image.open(img_path))
mask = np.array(Image.open(mask_path))

print("\n=== SHAPES ===")
print("Embedding:", emb.shape)
print("Emb mean: ", emb_mean.shape)
print("Image:    ", img.shape)
print("Mask:     ", mask.shape)

# Normalize for ASCII print
def to_ascii(arr, label):
    arr = arr.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    H, W = arr.shape

    # downsample to smaller grid for terminal
    step_h = H // 16
    step_w = W // 32
    arr_small = arr[::step_h, ::step_w]

    chars = " .:-=+*#%@"
    ascii_img = "\n".join(
        "".join(chars[int(v * (len(chars)-1))] for v in row)
        for row in arr_small
    )
    print(f"\n--- {label} (ASCII downsample) ---")
    print(ascii_img)


# ---- ASCII visual summaries ----
to_ascii(img,  "Image")
to_ascii(mask, "Mask")
to_ascii(emb_mean, "Embedding mean")

print("\nDone.\n")
