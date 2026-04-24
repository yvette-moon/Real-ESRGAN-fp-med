import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pydicom


def dcm_to_uint8(ds, wl=None, ww=None):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept

    if wl is not None and ww is not None and ww > 0:
        low = wl - ww / 2.0
        high = wl + ww / 2.0
        hu = np.clip(hu, low, high)
        norm = (hu - low) / (high - low + 1e-8)
    else:
        # fallback: per-slice min-max
        norm = (hu - hu.min()) / (hu.max() - hu.min() + 1e-8)

    img8 = (norm * 255.0).round().astype(np.uint8)
    return img8


def iter_dcm_files(root, recursive):
    if recursive:
        for p in Path(root).rglob("*.dcm"):
            yield p
    else:
        for p in Path(root).glob("*.dcm"):
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dcm_dir", type=str, required=True)
    parser.add_argument("--output_gt_dir", type=str, required=True)
    parser.add_argument("--output_lq_dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--wl", type=float, default=None, help="Window level")
    parser.add_argument("--ww", type=float, default=None, help="Window width")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    in_root = Path(args.input_dcm_dir)
    out_gt = Path(args.output_gt_dir)
    out_lq = Path(args.output_lq_dir)
    out_gt.mkdir(parents=True, exist_ok=True)
    out_lq.mkdir(parents=True, exist_ok=True)

    count = 0
    for dcm_path in iter_dcm_files(in_root, args.recursive):
        rel = dcm_path.relative_to(in_root)
        save_rel = rel.with_suffix(".png")

        gt_path = out_gt / save_rel
        lq_path = out_lq / save_rel
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        lq_path.parent.mkdir(parents=True, exist_ok=True)

        ds = pydicom.dcmread(str(dcm_path))
        img8 = dcm_to_uint8(ds, wl=args.wl, ww=args.ww)

        # Convert to 3-channel to match RealESRGAN paired val pipeline
        gt_bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        h, w = gt_bgr.shape[:2]
        lq_w = max(1, w // args.scale)
        lq_h = max(1, h // args.scale)
        lq_bgr = cv2.resize(gt_bgr, (lq_w, lq_h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(str(gt_path), gt_bgr)
        cv2.imwrite(str(lq_path), lq_bgr)
        count += 1

    print(f"Done. Processed {count} DICOM files.")


if __name__ == "__main__":
    main()
