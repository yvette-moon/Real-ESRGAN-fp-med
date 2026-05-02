import os
import numpy as np
import torch
from torch.utils import data
from basicsr.utils.registry import DATASET_REGISTRY

def _read_npy_1ch(path):
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported shape {arr.shape} at {path}")
    return np.clip(arr, 0.0, 1.0)

@DATASET_REGISTRY.register()
class PairedWaveletSubbandDataset(data.Dataset):
    """LQ subbands (LL/LH/HL/HH) + GT full-res image."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ll_dir = opt["dataroot_lq_ll"]
        self.lh_dir = opt["dataroot_lq_lh"]
        self.hl_dir = opt["dataroot_lq_hl"]
        self.hh_dir = opt["dataroot_lq_hh"]
        self.gt_dir = opt["dataroot_gt"]
        self.scale = int(opt.get("scale", 4))
        self.lq_crop = int(opt.get("lq_size", 64))
        self.use_hflip = bool(opt.get("use_hflip", True))
        self.use_rot = bool(opt.get("use_rot", False))

        meta_info = opt.get("meta_info", None)
        if not meta_info:
            raise ValueError("PairedWaveletSubbandDataset requires meta_info.")
        with open(meta_info, "r", encoding="utf-8") as fin:
            self.names = [line.strip().split(" ")[0] for line in fin if line.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ll = _read_npy_1ch(os.path.join(self.ll_dir, name))
        lh = _read_npy_1ch(os.path.join(self.lh_dir, name))
        hl = _read_npy_1ch(os.path.join(self.hl_dir, name))
        hh = _read_npy_1ch(os.path.join(self.hh_dir, name))
        gt = _read_npy_1ch(os.path.join(self.gt_dir, name))

        # stack LQ subbands: [4, H, W]
        lq = np.stack([ll, lh, hl, hh], axis=0)

        # crop: GT spatial size = LQ * scale * 2
        if self.opt.get("phase", "train") == "train":
            c, h, w = lq.shape
            th, tw = self.lq_crop, self.lq_crop
            if h < th or w < tw:
                # 简化处理：如果太小，直接中心裁剪/插值可加在这里
                pass
            i = np.random.randint(0, h - th + 1)
            j = np.random.randint(0, w - tw + 1)
            lq = lq[:, i:i + th, j:j + tw]

            gt_i = i * self.scale * 2
            gt_j = j * self.scale * 2
            gt_h = th * self.scale * 2
            gt_w = tw * self.scale * 2
            gt = gt[gt_i:gt_i + gt_h, gt_j:gt_j + gt_w]

            if self.use_hflip and np.random.rand() < 0.5:
                lq = lq[..., ::-1]
                gt = gt[..., ::-1]
            if self.use_rot and np.random.rand() < 0.5:
                lq = np.swapaxes(lq, -1, -2)
                gt = np.swapaxes(gt, -1, -2)

        lq = torch.from_numpy(lq.copy()).float()
        gt = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return {"lq": lq, "gt": gt, "lq_path": name, "gt_path": name}