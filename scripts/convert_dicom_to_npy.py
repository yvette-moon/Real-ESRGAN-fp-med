import argparse
from pathlib import Path
import numpy as np
import pydicom


def _to_float(x, default):
    if x is None:
        return float(default)
    if isinstance(x, (list, tuple)):
        return float(x[0])
    try:
        return float(x[0])  # pydicom MultiValue
    except Exception:
        return float(x)


def dcm_to_normalized_float32(ds, default_wl=-450.0, default_ww=1300.0):
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric == "MONOCHROME1":
        hu = hu.max() + hu.min() - hu

    wl = _to_float(getattr(ds, "WindowCenter", None), default_wl)
    ww = _to_float(getattr(ds, "WindowWidth", None), default_ww)
    if ww <= 0:
        ww = default_ww

    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0

    hu = np.clip(hu, vmin, vmax)
    img = (hu - vmin) / (vmax - vmin + 1e-8)
    return img.astype(np.float32)


def iter_dcm_files(root, recursive):
    root = Path(root)
    if recursive:
        yield from root.rglob("*.dcm")
    else:
        yield from root.glob("*.dcm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dcm_dir", type=str, required=True)
    parser.add_argument("--output_npy_dir", type=str, required=True)
    parser.add_argument("--meta_info_out", type=str, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--default_wl", type=float, default=-450.0)
    parser.add_argument("--default_ww", type=float, default=1300.0)
    args = parser.parse_args()

    in_root = Path(args.input_dcm_dir)
    out_root = Path(args.output_npy_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_info_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    rel_paths = []
    count = 0
    for dcm_path in iter_dcm_files(in_root, args.recursive):
        rel = dcm_path.relative_to(in_root).with_suffix(".npy")
        save_path = out_root / rel
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ds = pydicom.dcmread(str(dcm_path))
        arr = dcm_to_normalized_float32(ds, default_wl=args.default_wl, default_ww=args.default_ww)
        np.save(str(save_path), arr, allow_pickle=False)

        rel_paths.append(str(rel).replace("\\", "/"))
        count += 1

    with meta_path.open("w", encoding="utf-8") as f:
        for rp in rel_paths:
            f.write(rp + "\n")

    print(f"Done. Converted {count} DICOM files to NPY.")
    print(f"Meta file: {meta_path}")


if __name__ == "__main__":
    main()
