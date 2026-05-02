import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, 1, 0)
        self.fc2 = nn.Conv2d(hidden, channels, 1, 1, 0)

    def forward(self, x):
        w = torch.mean(x, dim=(2, 3), keepdim=True)
        w = torch.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w, w

class LightResBlock(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.body(x)

class PixelShuffleUpsampler(nn.Module):
    def __init__(self, scale, num_feat):
        super().__init__()
        layers = []
        if scale == 4:
            for _ in range(2):
                layers += [
                    nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True)
                ]
        elif scale == 2:
            layers += [
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True)
            ]
        else:
            raise ValueError(f"Unsupported scale={scale}")
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

@ARCH_REGISTRY.register()
class WaveletMultiBranchNet(nn.Module):
    def __init__(self, num_in_ch=4, num_feat=32, num_block=8, scale=4, ca_reduction=4):
        super().__init__()
        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[LightResBlock(num_feat) for _ in range(num_block)])
        self.ca = ChannelSE(num_feat, reduction=ca_reduction)
        self.upsampler = PixelShuffleUpsampler(scale, num_feat)

        # 4个输出头，每个子带1通道
        self.tail_ll = nn.Conv2d(num_feat, 1, 3, 1, 1)
        self.tail_lh = nn.Conv2d(num_feat, 1, 3, 1, 1)
        self.tail_hl = nn.Conv2d(num_feat, 1, 3, 1, 1)
        self.tail_hh = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat) + feat
        feat, _ = self.ca(feat)
        feat = self.upsampler(feat)
        ll = self.tail_ll(feat)
        lh = self.tail_lh(feat)
        hl = self.tail_hl(feat)
        hh = self.tail_hh(feat)
        return torch.cat([ll, lh, hl, hh], dim=1)