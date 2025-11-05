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

from src.config import cfg

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

    img = np.asarray(img_nii.get_fdata(), dtype=np.float32)  # (H, W, S) or (S, H, W) depending on file
    msk = np.asarray(msk_nii.get_fdata(), dtype=np.float32)

    # Standardize to (S, H, W) = slice-first
    if img.ndim == 3 and img.shape[0] != msk.shape[0] and img.shape[-1] == msk.shape[-1]:
        # assume (H, W, S) -> transpose to (S, H, W)
        img = np.moveaxis(img, -1, 0)
        msk = np.moveaxis(msk, -1, 0)
    elif img.ndim == 3 and img.shape[0] == msk.shape[0]:
        # already (S, H, W)
        pass
    else:
        raise ValueError(f"Unexpected shapes image={img.shape} mask={msk.shape}")

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
        case_id = Path(img_p).stem.split('.')[0]
        all_records += clean_case(Path(img_p), Path(msk_p), out_dir, case_id)

    # also save a CSV manifest
    import csv
    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","mask","case","slice"])
        writer.writeheader()
        writer.writerows(all_records)
    return all_records
