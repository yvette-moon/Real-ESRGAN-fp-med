import os
import numpy as np
from torch.utils import data
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedNPYDataset(data.Dataset):
    """Paired NPY dataset for GT/LQ training (offline degraded)."""

    def __init__(self, opt):
        super(PairedNPYDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.io_backend = opt.get("io_backend", {"type": "disk"})
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)

        # npy_value_mode: 'normalized' or 'hu'
        self.npy_value_mode = opt.get("npy_value_mode", "normalized").lower()
        self.default_wl = float(opt.get("default_wl", -450.0))
        self.default_ww = float(opt.get("default_ww", 1300.0))

        # meta_info required
        meta_info = opt.get("meta_info", None)
        if not meta_info:
            raise ValueError("PairedNPYDataset requires meta_info file.")

        self.paths = []
        with open(meta_info, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if ", " in line:
                    gt_rel, lq_rel = line.split(", ")
                elif "," in line:
                    gt_rel, lq_rel = [s.strip() for s in line.split(",")]
                else:
                    gt_rel = lq_rel = line

                gt_path = os.path.join(self.gt_folder, gt_rel)
                lq_path = os.path.join(self.lq_folder, lq_rel)
                self.paths.append({"gt_path": gt_path, "lq_path": lq_path})

    def _window_hu(self, hu):
        ww = max(self.default_ww, 1e-6)
        vmin = self.default_wl - ww / 2.0
        vmax = self.default_wl + ww / 2.0
        hu = np.clip(hu, vmin, vmax)
        return (hu - vmin) / (vmax - vmin + 1e-8)

    def _read_npy(self, path):
        arr = np.load(path, allow_pickle=False).astype(np.float32)

        # Accept HxW, HxWx1/3, or 1/3xHxW
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            img = arr
            if self.npy_value_mode == "hu":
                img = self._window_hu(img)
            else:
                img = np.clip(img, 0.0, 1.0)
            img = np.stack([img, img, img], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                img = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] == 3:
                img = arr
            else:
                raise ValueError(f"Unsupported NPY shape: {arr.shape} at {path}")

            if self.npy_value_mode == "hu":
                img = self._window_hu(img)
            else:
                img = np.clip(img, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported NPY ndim: {arr.ndim} at {path}")

        return img.astype(np.float32)

    def __getitem__(self, index):
        gt_path = self.paths[index]["gt_path"]
        lq_path = self.paths[index]["lq_path"]

        img_gt = self._read_npy(gt_path)
        img_lq = self._read_npy(lq_path)

        if img_gt.shape[:2] != img_lq.shape[:2]:
            raise ValueError(f"GT/LQ shape mismatch: {img_gt.shape} vs {img_lq.shape} for {gt_path}")

        if self.opt.get("phase", "train") == "train":
            gt_size = self.opt["gt_size"]
            scale = self.opt["scale"]
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"])

        # keep channel order; grayscale expanded to 3 channels already
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        # optional normalize
        if self.mean is not None or self.std is not None:
            from torchvision.transforms.functional import normalize
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)
