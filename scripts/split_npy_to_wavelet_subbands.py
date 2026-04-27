import argparse
from pathlib import Path
import numpy as np


def center_crop_even(x):
    """小波分解前强制裁到偶数尺寸，避免奇数陷阱"""
    h, w = x.shape
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    if new_h == h and new_w == w:
        return x
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return x[top:top + new_h, left:left + new_w]


def haar_dwt2(x):
    """x: HxW float32 in [0,1]"""
    x = center_crop_even(x)  # ✅ 防止奇数尺寸破坏配对
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]

    # Match the inverse convention used later
    ll = (a + b + c + d) * 0.5          # approx, range ~[0,2]
    lh = (-a - b + c + d) * 0.5         # high-freq, range ~[-1,1]
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def norm_subbands(ll, lh, hl, hh):
    # Normalize into [0,1] for RealESRGAN pipeline
    ll_n = np.clip(ll / 2.0, 0.0, 1.0).astype(np.float32)
    lh_n = np.clip((lh + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    hl_n = np.clip((hl + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    hh_n = np.clip((hh + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    return ll_n, lh_n, hl_n, hh_n


def iter_npy_files(root):
    root = Path(root)
    yield from root.rglob("*.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)   # contains LL/LH/HL/HH
    parser.add_argument("--meta_root", type=str, required=True)     # outputs meta_*.txt
    args = parser.parse_args()

    in_root = Path(args.input_npy_dir)
    out_root = Path(args.output_root)
    meta_root = Path(args.meta_root)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    band_dirs = {
        "LL": out_root / "ll",
        "LH": out_root / "lh",
        "HL": out_root / "hl",
        "HH": out_root / "hh",
    }
    for d in band_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rel_list = []
    count = 0
    for p in iter_npy_files(in_root):
        rel = p.relative_to(in_root)
        arr = np.load(str(p), allow_pickle=False).astype(np.float32)
        if arr.ndim == 3:
            # accept HxWx1 or HxWx3, use first channel
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Unsupported shape {arr.shape} in {p}")

        ll, lh, hl, hh = haar_dwt2(arr)
        ll, lh, hl, hh = norm_subbands(ll, lh, hl, hh)

        out_rel = rel
        for band_name, band_arr in [("LL", ll), ("LH", lh), ("HL", hl), ("HH", hh)]:
            save_path = band_dirs[band_name] / out_rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), band_arr, allow_pickle=False)

        rel_list.append(str(out_rel).replace("\\", "/"))
        count += 1

    # meta files for 4 trainings
    for band_name in ["LL", "LH", "HL", "HH"]:
        meta_path = meta_root / f"meta_{band_name.lower()}.txt"
        with meta_path.open("w", encoding="utf-8") as f:
            for rel in rel_list:
                f.write(rel + "\n")

    print(f"Done. Split {count} npy files into ll/lh/hl/hh.")
    print(f"Output root: {out_root}")
    print(f"Meta root:   {meta_root}")


if __name__ == "__main__":
    main()