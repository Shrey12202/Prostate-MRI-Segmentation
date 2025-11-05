"""U-Net style decoder that maps MedSAM encoder features -> segmentation mask.
Input: (B, C, H/stride, W/stride). Output: (B, num_classes, H, W).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import cfg

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, base: int = 64, up_stages: int = 2, stride: int = 4):
        super().__init__()
        self.stride = stride
        # project input to feature width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
        )
        # progressive upsampling stages (e.g., from H/4->H/2->H)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.dec1 = conv_block(base*2, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec2 = conv_block(base, base)

        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, emb: torch.Tensor):
        # emb: (B,C,h,w) where h=H/stride, w=W/stride
        x = self.stem(emb)             # (B,4B,h,w)
        x = self.dec1(self.up1(x))     # (B,2B,h*2,w*2)
        x = self.dec2(self.up2(x))     # (B,B,h*4,w*4) -> should match H,W if stride=4
        logits = self.head(x)          # (B,num_classes,H,W)
        return logits
