import cv2
import math
import numpy as np
import os
import random
import torch
from torch.utils import data
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import img2tensor


@DATASET_REGISTRY.register()
class NPYRealESRGANDataset(data.Dataset):
    """Read preprocessed NPY slices and feed Real-ESRGAN training pipeline."""

    def __init__(self, opt):
        super(NPYRealESRGANDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt["dataroot_gt"]

        with open(self.opt["meta_info"], "r", encoding="utf-8") as fin:
            paths = [line.strip().split(" ")[0] for line in fin if line.strip()]
            self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # npy_value_mode: 'normalized' (already [0,1]) or 'hu' (need windowing here)
        self.npy_value_mode = opt.get("npy_value_mode", "normalized").lower()
        self.default_wl = float(opt.get("default_wl", -450.0))
        self.default_ww = float(opt.get("default_ww", 1300.0))

        # degradation kernel settings (same as original)
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.kernel_list = opt["kernel_list"]
        self.kernel_prob = opt["kernel_prob"]
        self.blur_sigma = opt["blur_sigma"]
        self.betag_range = opt["betag_range"]
        self.betap_range = opt["betap_range"]
        self.sinc_prob = opt["sinc_prob"]

        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.kernel_list2 = opt["kernel_list2"]
        self.kernel_prob2 = opt["kernel_prob2"]
        self.blur_sigma2 = opt["blur_sigma2"]
        self.betag_range2 = opt["betag_range2"]
        self.betap_range2 = opt["betap_range2"]
        self.sinc_prob2 = opt["sinc_prob2"]
        self.final_sinc_prob = opt["final_sinc_prob"]

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1.0

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
                # If user stored HU in 3 channels, window each channel identically
                img = self._window_hu(img)
            else:
                img = np.clip(img, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported NPY ndim: {arr.ndim} at {path}")

        return img.astype(np.float32)

    def __getitem__(self, index):
        gt_path = self.paths[index]
        img_gt = self._read_npy(gt_path)

        img_gt = augment(img_gt, self.opt["use_hflip"], self.opt["use_rot"])
        h, w = img_gt.shape[:2]
        # 让它读取配置文件，离线生成时我们配的是 gt_size: 0，就不会裁剪了
        crop_pad_size = self.opt.get("gt_size", 0)

        # 将后面的判断用 if crop_pad_size > 0 包裹起来
        if crop_pad_size > 0:
            if h < crop_pad_size or w < crop_pad_size:
                pad_h = max(0, crop_pad_size - h)
                pad_w = max(0, crop_pad_size - w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
                h, w = img_gt.shape[:2]
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # first kernel
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob"]:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list, self.kernel_prob, kernel_size,
                self.blur_sigma, self.blur_sigma, [-math.pi, math.pi],
                self.betag_range, self.betap_range, noise_range=None
            )
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # second kernel
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob2"]:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2, self.kernel_prob2, kernel_size,
                self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2, self.betap_range2, noise_range=None
            )
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # channel values are equal, no BGR<->RGB swap needed
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0]

        return {
            "gt": img_gt,
            "kernel1": torch.FloatTensor(kernel),
            "kernel2": torch.FloatTensor(kernel2),
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path
        }

    def __len__(self):
        return len(self.paths)
