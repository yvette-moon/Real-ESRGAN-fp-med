import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ChannelSE(nn.Module):
    def __init__(self, channels=4, reduction=2):
        super(ChannelSE, self).__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, 1, 0)
        self.fc2 = nn.Conv2d(hidden, channels, 1, 1, 0)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w, w


@ARCH_REGISTRY.register()
class WaveletCAFuser(nn.Module):
    def __init__(self, channels=4, mid_channels=32):
        super(WaveletCAFuser, self).__init__()
        self.ca = ChannelSE(channels=channels, reduction=2)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        x_ca, w = self.ca(x)
        x_ref = self.refine(x_ca)
        out = x_ca + x_ref
        return out, w


def denorm_subbands(ll_n, lh_n, hl_n, hh_n):
    ll = ll_n * 2.0
    lh = lh_n * 2.0 - 1.0
    hl = hl_n * 2.0 - 1.0
    hh = hh_n * 2.0 - 1.0
    return ll, lh, hl, hh


def haar_idwt2_torch(ll, lh, hl, hh):
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5

    bsz, _, h, w = ll.shape
    out = torch.zeros((bsz, 1, h * 2, w * 2), dtype=ll.dtype, device=ll.device)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out
