import argparse
import os
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from realesrgan.archs.wavelet_ca_fuser_arch import WaveletCAFuser, haar_idwt2_torch, denorm_subbands


def load_npy_float(path):
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported shape: {arr.shape} @ {path}")
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]


class WaveletFusionDataset(Dataset):
    def __init__(self, ll_dir, lh_dir, hl_dir, hh_dir, gt_dir, ext="npy", crop_size=256):
        super().__init__()
        self.ll_paths = sorted(glob(os.path.join(ll_dir, f"*.{ext}")))
        self.lh_dir = lh_dir
        self.hl_dir = hl_dir
        self.hh_dir = hh_dir
        self.gt_dir = gt_dir
        self.names = [os.path.basename(p) for p in self.ll_paths]
        self.crop_size = crop_size  # 子带的裁剪尺寸，如 256

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        ll = load_npy_float(os.path.join(os.path.dirname(self.ll_paths[idx]), n))
        lh = load_npy_float(os.path.join(self.lh_dir, n))
        hl = load_npy_float(os.path.join(self.hl_dir, n))
        hh = load_npy_float(os.path.join(self.hh_dir, n))
        gt = load_npy_float(os.path.join(self.gt_dir, n))

        x = torch.cat([ll, lh, hl, hh], dim=0)  # [4, H, W]

        # --- 同步裁剪逻辑 ---
        c, h, w = x.shape
        th, tw = self.crop_size, self.crop_size

        if h < th or w < tw:
            # 如果图片比裁剪尺寸小，先插值放大到 crop_size
            x = F.interpolate(x.unsqueeze(0), size=(th, tw), mode='bilinear').squeeze(0)
            gt = F.interpolate(gt.unsqueeze(0), size=(th * 2, tw * 2), mode='bilinear').squeeze(0)
        else:
            # 随机选择裁剪起始点
            i = torch.randint(0, h - th + 1, (1,)).item()
            j = torch.randint(0, w - tw + 1, (1,)).item()

            x = x[:, i:i + th, j:j + tw]
            # GT 对应的裁剪位置必须是子带的 2 倍（对应 iDWT 逻辑）
            gt = gt[:, i * 2:(i + th) * 2, j * 2:(j + tw) * 2]

        return x, gt, n

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ll_dir", required=True)
    parser.add_argument("--lh_dir", required=True)
    parser.add_argument("--hl_dir", required=True)
    parser.add_argument("--hh_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--save_path", default="experiments/pretrained_models/wavelet_ca_fuser.pth")
    parser.add_argument("--ext", default="npy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256, help="Crop size for subbands")  # 新增
    args = parser.parse_args()

    # 修改 Dataset 的初始化，传入 crop_size
    ds = WaveletFusionDataset(
        args.ll_dir, args.lh_dir, args.hl_dir, args.hh_dir, args.gt_dir,
        ext=args.ext, crop_size=args.crop_size
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WaveletFusionDataset(args.ll_dir, args.lh_dir, args.hl_dir, args.hh_dir, args.gt_dir, args.ext)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = WaveletCAFuser(channels=4, mid_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    l1 = nn.L1Loss()

    model.train()
    for ep in range(args.epochs):
        loss_sum = 0.0
        for x, gt, _ in dl:
            x = x.to(device)      # [B,4,H,W]
            gt = gt.to(device)    # [B,1,2H,2W] or [B,1,H,W] depending on your prep

            # If gt has same size as subbands, that means your gt is band-sized.
            # For iDWT supervision, gt should be full-resolution.

            out_bands, _ = model(x)
            ll, lh, hl, hh = torch.chunk(out_bands, 4, dim=1)

            # [0,1] 子带 -> 原始小波系数域
            ll, lh, hl, hh = denorm_subbands(ll, lh, hl, hh)

            recon = haar_idwt2_torch(ll, lh, hl, hh)
            recon = torch.clamp(recon, 0.0, 1.0)

            # If dimensions mismatch, resize gt to recon for robust training.
            if gt.shape[-2:] != recon.shape[-2:]:
                gt = F.interpolate(gt, size=recon.shape[-2:], mode="bilinear", align_corners=False)

            loss = l1(recon, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        print(f"Epoch [{ep+1}/{args.epochs}] loss={loss_sum / max(len(dl),1):.6f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({"params": model.state_dict()}, args.save_path)
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
