"""Augmentation pipeline (2D) using torchvision transforms."""
from __future__ import annotations

import random
from typing import Tuple
import torch
import torchvision.transforms.functional as TF

from src.config import cfg

class SliceAugmentor:
    def __init__(self):
        self.rot_deg = cfg.rotation_deg
        self.hflip_p = cfg.hflip_p
        self.vflip_p = cfg.vflip_p
        self.brightness = cfg.brightness
        self.contrast = cfg.contrast
        self.translate = cfg.translate
        self.scale = cfg.scale

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # img: (1,H,W) float in [0,1], mask: (H,W) long
        # random flips
        if random.random() < self.hflip_p:
            img = TF.hflip(img)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        if random.random() < self.vflip_p:
            img = TF.vflip(img)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # affine
        angle = random.uniform(-self.rot_deg, self.rot_deg)
        translate = (int(cfg.image_size[1]*self.translate*random.uniform(-1,1)),
                     int(cfg.image_size[0]*self.translate*random.uniform(-1,1)))
        scale = random.uniform(self.scale[0], self.scale[1])
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[0.0,0.0])
        mask = TF.affine(mask.unsqueeze(0).float(), angle=angle, translate=translate, scale=scale, shear=[0.0,0.0], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

        # color jitter (brightness/contrast) on image only
        if self.brightness > 0:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = TF.adjust_brightness(img, b)
        if self.contrast > 0:
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = TF.adjust_contrast(img, c)

        img = torch.clamp(img, 0.0, 1.0)
        return img, mask
