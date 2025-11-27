"""U-Net style decoder that maps MedSAM encoder features -> segmentation mask.
Input: (B, C, H/stride, W/stride) where stride≈16 for 1024x1024 MedSAM embeddings.
Output: (B, num_classes, H, W) at full 1024x1024 resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg


def conv_block(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNetDecoder(nn.Module):
    """
    Decoder for MedSAM embeddings.

    Assumes:
      - Input embedding spatial size ~ H/16 x W/16 (e.g. 64x64 for 1024x1024 images).
      - We upsample 4 times: 64 -> 128 -> 256 -> 512 -> 1024.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        base: int = 64,
        up_stages: int = 4,
        stride: int = 16,  # kept for compatibility, not strictly used
    ):
        super().__init__()
        self.stride = stride

        # Project MedSAM channels to a wider feature space
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        # 64 -> 128
        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec1 = conv_block(base * 4, base * 4)

        # 128 -> 256
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base * 2, base * 2)

        # 256 -> 512
        self.up3 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec3 = conv_block(base, base)

        # 512 -> 1024
        self.up4 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.dec4 = conv_block(base, base)

        # Final prediction head
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, C, h, w) with h,w ~ 64 for 1024x1024 inputs.
        Returns:
            logits: (B, num_classes, 1024, 1024)
        """
        x = self.stem(emb)          # (B, 8B, 64, 64)

        x = self.up1(x)             # (B, 4B, 128, 128)
        x = self.dec1(x)

        x = self.up2(x)             # (B, 2B, 256, 256)
        x = self.dec2(x)

        x = self.up3(x)             # (B, B, 512, 512)
        x = self.dec3(x)

        x = self.up4(x)             # (B, B, 1024, 1024)
        x = self.dec4(x)

        logits = self.head(x)       # (B, num_classes, 1024, 1024)

        # If for any reason the size is off by 1–2 pixels, enforce exact cfg.image_size
        if logits.shape[-2:] != cfg.image_size:
            logits = F.interpolate(
                logits,
                size=cfg.image_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits
