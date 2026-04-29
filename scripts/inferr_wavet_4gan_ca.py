import argparse
import os
from glob import glob
import cv2
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.wavelet_ca_fuser_arch import WaveletCAFuser, haar_idwt2_torch, denorm_subbands

def load_npy_float(path):
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported shape: {arr.shape} @ {path}")
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

def build_upsampler(model_path, scale=4, fp32=False, gpu_id=None):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=not fp32,
        gpu_id=gpu_id
    )


def read_npy_gray(path):
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported shape: {arr.shape} @ {path}")
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)



def to_3ch(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_band(upsampler, band_gray, outscale):
    inp = to_3ch(band_gray)
    out, _ = upsampler.enhance(inp, outscale=outscale)
    if out.ndim == 3:
        out = out[:, :, 0]
    return out.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ll_dir", required=True)
    parser.add_argument("--lh_dir", required=True)
    parser.add_argument("--hl_dir", required=True)
    parser.add_argument("--hh_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--model_ll", required=True)
    parser.add_argument("--model_lh", required=True)
    parser.add_argument("--model_hl", required=True)
    parser.add_argument("--model_hh", required=True)

    parser.add_argument("--fuser_ckpt", required=True)
    parser.add_argument("--ext", default="npy")
    parser.add_argument("--outscale", type=float, default=4.0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    up_ll = build_upsampler(args.model_ll, fp32=args.fp32, gpu_id=args.gpu_id)
    up_lh = build_upsampler(args.model_lh, fp32=args.fp32, gpu_id=args.gpu_id)
    up_hl = build_upsampler(args.model_hl, fp32=args.fp32, gpu_id=args.gpu_id)
    up_hh = build_upsampler(args.model_hh, fp32=args.fp32, gpu_id=args.gpu_id)

    fuser = WaveletCAFuser(channels=4, mid_channels=32).to(device)
    ckpt = torch.load(args.fuser_ckpt, map_location=device)
    fuser.load_state_dict(ckpt["params"], strict=True)
    fuser.eval()

    ll_paths = sorted(glob(os.path.join(args.ll_dir, f"*.{args.ext}")))
    for ll_path in ll_paths:
        name = os.path.basename(ll_path)
        lh_path = os.path.join(args.lh_dir, name)
        hl_path = os.path.join(args.hl_dir, name)
        hh_path = os.path.join(args.hh_dir, name)
        if not (os.path.exists(lh_path) and os.path.exists(hl_path) and os.path.exists(hh_path)):
            continue

        ll = enhance_band(up_ll, read_npy_gray(ll_path), args.outscale)
        lh = enhance_band(up_lh, read_npy_gray(lh_path), args.outscale)
        hl = enhance_band(up_hl, read_npy_gray(hl_path), args.outscale)
        hh = enhance_band(up_hh, read_npy_gray(hh_path), args.outscale)
        assert ll.shape == lh.shape == hl.shape == hh.shape, \
            f"Shape mismatch: {ll.shape}, {lh.shape}, {hl.shape}, {hh.shape}"
        assert ll.shape[0] % 2 == 0 and ll.shape[1] % 2 == 0, \
            f"Subbands must have even dimensions: {ll.shape}"
        x = np.stack([ll, lh, hl, hh], axis=0)  # [4,H,W]
        x = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,4,H,W]

        with torch.no_grad():
            y, _ = fuser(x)


            #
            y = torch.clamp(y, 0.0, 1.0)

            # [0,1] 子带 -> 原始小波系数域
            ll_t, lh_t, hl_t, hh_t = torch.chunk(y, 4, dim=1)  # ✅ 再切分
            ll_t, lh_t, hl_t, hh_t = denorm_subbands(ll_t, lh_t, hl_t, hh_t)
            recon = haar_idwt2_torch(ll_t, lh_t, hl_t, hh_t)
            recon = torch.clamp(recon, 0.0, 1.0)

        out_path = os.path.join(args.out_dir, os.path.splitext(name)[0] + ".npy")
        np.save(out_path, recon[0, 0].cpu().numpy().astype(np.float32), allow_pickle=False)
        print(f"Saved {name}")


if __name__ == "__main__":
    main()
