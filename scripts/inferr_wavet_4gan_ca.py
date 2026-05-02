import argparse
import os
from glob import glob
import numpy as np
import torch

from realesrgan.archs.wavelet_ca_fuser_arch import haar_idwt2_torch, denorm_subbands
from realesrgan.archs.wavelet_multiband_arch import WaveletMultiBandNet


def read_npy_gray_float01(path):
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported shape: {arr.shape} @ {path}")
    return np.clip(arr, 0.0, 1.0)


def load_generator(ckpt_path, device, in_ch=4, out_ch=4, mid_ch=32):
    # TODO: 如果你实际的生成器构造参数不同，请对齐这里
    net_g = WaveletMultiBandNet(in_ch=in_ch, out_ch=out_ch, mid_ch=mid_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容常见格式：{"params": ...} 或直接 state_dict
    state = ckpt.get("params", ckpt)
    net_g.load_state_dict(state, strict=True)
    net_g.eval()
    return net_g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ll_dir", required=True)
    parser.add_argument("--lh_dir", required=True)
    parser.add_argument("--hl_dir", required=True)
    parser.add_argument("--hh_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--model_g", required=True)
    parser.add_argument("--ext", default="npy")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=None)

    # 可选：如果你的生成器需要指定通道数
    parser.add_argument("--mid_ch", type=int, default=32)

    args = parser.parse_args()

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    net_g = load_generator(
        args.model_g,
        device=device,
        in_ch=4,
        out_ch=4,
        mid_ch=args.mid_ch
    )

    ll_paths = sorted(glob(os.path.join(args.ll_dir, f"*.{args.ext}")))
    for ll_path in ll_paths:
        name = os.path.basename(ll_path)
        lh_path = os.path.join(args.lh_dir, name)
        hl_path = os.path.join(args.hl_dir, name)
        hh_path = os.path.join(args.hh_dir, name)
        if not (os.path.exists(lh_path) and os.path.exists(hl_path) and os.path.exists(hh_path)):
            continue

        ll = read_npy_gray_float01(ll_path)
        lh = read_npy_gray_float01(lh_path)
        hl = read_npy_gray_float01(hl_path)
        hh = read_npy_gray_float01(hh_path)

        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError(f"Shape mismatch: {ll.shape}, {lh.shape}, {hl.shape}, {hh.shape}")
        if ll.shape[0] % 2 != 0 or ll.shape[1] % 2 != 0:
            raise ValueError(f"Subbands must have even dimensions: {ll.shape}")

        x = np.stack([ll, lh, hl, hh], axis=0)  # [4,H,W]
        x = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,4,H,W]

        with torch.no_grad():
            y = net_g(x)
            # 如果你的生成器输出 (y, attn)，用：
            # y, _ = net_g(x)

            y = torch.clamp(y, 0.0, 1.0)

            ll_t, lh_t, hl_t, hh_t = torch.chunk(y, 4, dim=1)
            ll_t, lh_t, hl_t, hh_t = denorm_subbands(ll_t, lh_t, hl_t, hh_t)
            recon = haar_idwt2_torch(ll_t, lh_t, hl_t, hh_t)
            recon = torch.clamp(recon, 0.0, 1.0)

        out_path = os.path.join(args.out_dir, os.path.splitext(name)[0] + ".npy")
        np.save(out_path, recon[0, 0].cpu().numpy().astype(np.float32), allow_pickle=False)
        print(f"Saved {name}")


if __name__ == "__main__":
    main()