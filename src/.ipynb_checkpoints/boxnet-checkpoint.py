import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = conv_block(base * 4, base * 2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        x = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(x)

        return self.out(x)  # logits
