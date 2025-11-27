"""Torch Dataset that loads cleaned slices, masks, and precomputed MedSAM embeddings (.pt).
Embeddings are expected per-slice as shape (C, H//stride, W//stride).
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Tuple

from src.config import cfg
from src.augmentation import SliceAugmentor

class SliceDataset(Dataset):
    def __init__(self, manifest_csv: Path, split_csv: Optional[Path], embeddings_dir: Path, augment: bool=False):
        self.manifest = pd.read_csv(manifest_csv)
        if split_csv is not None:
            split_df = pd.read_csv(split_csv)
            keep = set(split_df['image'].tolist())
            self.manifest = self.manifest[self.manifest['image'].isin(keep)].reset_index(drop=True)
        self.embeddings_dir = Path(embeddings_dir)
        self.augment = augment
        self.aug = SliceAugmentor() if augment else None

    def __len__(self):
        return len(self.manifest)

    def _load_img(self, p: str) -> torch.Tensor:
        im = Image.open(p).convert("L")
        t = TF.pil_to_tensor(im).float() / 255.0  # (1,H,W)
        return t

    def _load_mask(self, p: str) -> torch.Tensor:
        mk = Image.open(p).convert("L")
        mk_t = (TF.pil_to_tensor(mk) > 127).long().squeeze(0)  # (H,W) long in {0,1}
        return mk_t

    def _load_embedding(self, image_path: str) -> torch.Tensor:
        # expects a .pt with same basename as image file
        stem = Path(image_path).stem
        emb_p = self.embeddings_dir / f"{stem}.pt"
        if not emb_p.exists():
            raise FileNotFoundError(f"Missing embedding for {stem} at {emb_p}")
        feat = torch.load(emb_p)  # (C,h,w)
        return feat.float()

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img = self._load_img(row['image'])         # (1,H,W)
        mask = self._load_mask(row['mask'])        # (H,W)
        emb = self._load_embedding(row['image'])   # (C,h,w)

        if self.augment:
            img, mask = self.aug(img, mask)

        return {
            "image": img,             # (1,H,W)
            "mask": mask,             # (H,W)
            "embedding": emb,         # (C,h,w)
            "image_path": row['image']
        }

def make_loader(manifest_csv: Path, split_csv: Optional[Path], embeddings_dir: Path, batch_size: int, shuffle: bool, augment: bool, num_workers: int):
    ds = SliceDataset(manifest_csv, split_csv, embeddings_dir, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
