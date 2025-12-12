"""Data cleaning & preprocessing for Prostate158 MRI (2D slice-wise).
Writes PNG slices and binary masks with consistent size/orientation.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage.io import imsave
from typing import Tuple, List, Dict

from config import cfg

def resize_slice(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D slice to (H, W).
    Uses bilinear for images, nearest-neighbor for masks (auto-detected).
    """
    arr = arr.astype(np.float32)

    # masks are binary or small integers → use nearest neighbor
    is_mask = arr.dtype != np.float32 or np.unique(arr).shape[0] <= 5

    if is_mask:
        return resize(
            arr,
            out_hw,
            order=0,             # nearest
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.float32)

    # images → use bilinear + anti-alias
    return resize(
        arr,
        out_hw,
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

def robust_min_max(x: np.ndarray, p_lo: float, p_hi: float) -> Tuple[float, float]:
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi <= lo:
        hi = x.max()
        lo = x.min()
    return lo, hi

def normalize_img(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    lo, hi = robust_min_max(x, p_lo, p_hi)
    x = np.clip((x - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return x

def bbox_from_mask(mask: np.ndarray) -> Tuple[int,int,int,int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # fallback: full image
        h,w = mask.shape
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2

def pad_and_crop(img: np.ndarray, mask: np.ndarray, pad: int, out_hw: Tuple[int,int]) -> Tuple[np.ndarray, np.ndarray]:
    x1,y1,x2,y2 = bbox_from_mask(mask)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1]-1, x2 + pad); y2 = min(mask.shape[0]-1, y2 + pad)
    crop_img = img[y1:y2+1, x1:x2+1]
    crop_msk = mask[y1:y2+1, x1:x2+1]
    # resize to out_hw
    out_h, out_w = out_hw
    img_r = resize(crop_img, (out_h, out_w), order=1, anti_aliasing=True, preserve_range=True)
    msk_r = resize(crop_msk, (out_h, out_w), order=0, anti_aliasing=False, preserve_range=True) > 0.5
    return img_r.astype(np.float32), msk_r.astype(np.uint8)

def resize_pair(img: np.ndarray, mask: np.ndarray, out_hw: Tuple[int,int]) -> Tuple[np.ndarray, np.ndarray]:
    out_h, out_w = out_hw
    img_r = resize(img, (out_h, out_w), order=1, anti_aliasing=True, preserve_range=True)
    msk_r = resize(mask, (out_h, out_w), order=0, anti_aliasing=False, preserve_range=True) > 0.5
    return img_r.astype(np.float32), msk_r.astype(np.uint8)

def clean_case(image_path: Path, mask_path: Path, out_dir: Path, case_id: str) -> List[Dict]:
    os.makedirs(out_dir / "images", exist_ok=True)
    os.makedirs(out_dir / "masks", exist_ok=True)

    img_nii = nib.load(str(image_path))
    msk_nii = nib.load(str(mask_path))

    img = np.asarray(img_nii.get_fdata(), dtype=np.float32)  # e.g. (H, W, S)
    msk = np.asarray(msk_nii.get_fdata(), dtype=np.float32)

    if img.ndim != 3 or msk.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got image={img.shape}, mask={msk.shape}")
    if img.shape != msk.shape:
        raise ValueError(f"Image and mask shapes differ: image={img.shape}, mask={msk.shape}")

    # Treat the SMALLEST dimension as the slice axis (e.g. (442,442,27) → axis=2)
    slice_axis = int(np.argmin(img.shape))

    # Reorder to (S, H, W)
    if slice_axis != 0:
        img = np.moveaxis(img, slice_axis, 0)
        msk = np.moveaxis(msk, slice_axis, 0)

    # Now img.shape == msk.shape == (S, H, W), e.g. (27, 442, 442)


    records = []
    for s in range(img.shape[0]):
        im = img[s]
        mk = (msk[s] > 0.5).astype(np.uint8)

        # normalize slice
        im = normalize_img(im, cfg.robust_norm[0], cfg.robust_norm[1])

        # bbox crop or resize
        if cfg.use_bbox_crop:
            im, mk = pad_and_crop(im, mk, cfg.bbox_pad_px, cfg.image_size)
        else:
            im, mk = resize_pair(im, mk, cfg.image_size)

        # to uint8 PNGs
        im_u8 = (im * 255.0).round().astype(np.uint8)
        mk_u8 = (mk * 255).astype(np.uint8)

        img_out = out_dir / "images" / f"{case_id}_s{s:03d}.png"
        msk_out = out_dir / "masks" / f"{case_id}_s{s:03d}.png"
        imsave(str(img_out), im_u8)
        imsave(str(msk_out), mk_u8)

        records.append({"image": str(img_out), "mask": str(msk_out), "case": case_id, "slice": s})

    return records

def run_cleaning(pairs: List[Tuple[Path, Path]], out_dir: Path | None = None) -> List[Dict]:
    """pairs: list of (image_nii_path, mask_nii_path) for the dataset cases."""
    if out_dir is None:
        out_dir = cfg.preprocessed_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for img_p, msk_p in pairs:
        case_id = Path(img_p).parent.name
        # case_id = Path(img_p).stem.split('.')[0]
        all_records += clean_case(Path(img_p), Path(msk_p), out_dir, case_id)

    # also save a CSV manifest
    import csv
    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","mask","case","slice"])
        writer.writeheader()
        writer.writerows(all_records)
    return all_records
    
def generate_test_manifest(raw_test_root: Path, out_dir: Path, image_size=(1024,1024)):
    """
    Preprocess the test set under raw_test_root and create:
      - test_images/
      - test_masks/
      - test_manifest.csv (slice-level)
    Assumes:
      t2.nii.gz
      t2_anatomy_reader1.nii.gz    # prostate mask
    """
    import nibabel as nib
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    from PIL import Image

    test_img_dir = out_dir / "test" / "images"
    test_msk_dir = out_dir / "test" / "masks"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_msk_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # Scan test patients
    for p in sorted(raw_test_root.iterdir()):
        if not p.is_dir():
            continue

        case_id = p.name  # e.g. "020"

        t2_path = p / "t2.nii.gz"
        msk_path = p / "t2_anatomy_reader1.nii.gz"

        if not t2_path.exists() or not msk_path.exists():
            print(f"[WARN] Skipping {case_id} (missing t2 or mask)")
            continue

        nii_img = nib.load(str(t2_path)).get_fdata()
        nii_msk = nib.load(str(msk_path)).get_fdata()

        for s in range(nii_img.shape[-1]):
            img = nii_img[..., s]
            msk = nii_msk[..., s]

            img = resize_slice(img, image_size)
            msk = resize_slice(msk, image_size)

            img_file = test_img_dir / f"{case_id}_s{s:03d}.png"
            msk_file = test_msk_dir / f"{case_id}_s{s:03d}.png"

            Image.fromarray((img * 255).astype(np.uint8)).save(img_file)
            Image.fromarray((msk > 0).astype(np.uint8) * 255).save(msk_file)

            records.append([str(img_file), str(msk_file), case_id, s])

    manifest_path = out_dir / "test_manifest.csv"
    pd.DataFrame(records, columns=["image", "mask", "case", "slice"]).to_csv(manifest_path, index=False)
    return manifest_path
