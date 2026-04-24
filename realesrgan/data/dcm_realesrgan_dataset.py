import pydicom
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
class DICOMRealESRGANDataset(data.Dataset):
    """用于读取单通道 DICOM 并将其伪装为 3 通道供给 Real-ESRGAN 的数据集类"""

    def __init__(self, opt):
        super(DICOMRealESRGANDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']

        # 从 meta_info 文件中读取文件列表
        with open(self.opt['meta_info']) as fin:
            paths = [line.strip().split(' ')[0] for line in fin]
            self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # ---------------- 复制原版的退化核参数初始化 ---------------- #
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def _read_dcm(self, path):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)

        # 1) Convert to HU using DICOM rescale tags.
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        hu = arr * slope + intercept

        # 2) Optional: handle MONOCHROME1 (inverse grayscale).
        # Most CT is MONOCHROME2; this keeps behavior robust for mixed data.
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
        if photometric == "MONOCHROME1":
            hu = hu.max() + hu.min() - hu

        # 3) Resolve window center/width.
        # Priority: explicit fixed ww/wl -> DICOM tags -> fallback defaults.
        # Keep your requested defaults for chest CT.
        default_wl = -450.0
        default_ww = 1300.0

        def _to_float(x, default):
            if x is None:
                return float(default)
            # DICOM tag may be MultiValue/list/tuple
            if isinstance(x, (list, tuple)):
                return float(x[0])
            try:
                # pydicom MultiValue can be indexed
                return float(x[0])
            except Exception:
                return float(x)

        wl = _to_float(getattr(ds, "WindowCenter", None), default_wl)
        ww = _to_float(getattr(ds, "WindowWidth", None), default_ww)
        if ww <= 0:
            ww = default_ww

        vmin = wl - ww / 2.0
        vmax = wl + ww / 2.0

        # 4) Windowing + normalization to [0, 1].
        # This replaces the previous "img_array <= -900" hard threshold.
        hu = np.clip(hu, vmin, vmax)
        img = (hu - vmin) / (vmax - vmin + 1e-8)

        # 5) Keep current 3-channel pipeline compatibility.
        img_rgb = np.stack([img, img, img], axis=-1)
        return img_rgb

    def __getitem__(self, index):
        gt_path = self.paths[index]

        # 调用刚才写的方法加载并转换为 3 通道
        img_gt = self._read_dcm(gt_path)

        # ---------------- 维持原版的增强和裁剪逻辑 ---------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400  # 对应原版硬编码大小
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ---------------- 生成一阶/二阶/Sinc退化核（完全保留原版） ---------------- #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, kernel_size,
                                          self.blur_sigma, self.blur_sigma, [-math.pi, math.pi],
                                          self.betag_range, self.betap_range, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(self.kernel_list2, self.kernel_prob2, kernel_size,
                                           self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                                           self.betag_range2, self.betap_range2, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # 【修改点3】: bgr2rgb 设置为 False，因为我们是手搓的同值三通道，不需要倒换顺位
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0]

        return {'gt': img_gt, 'kernel1': torch.FloatTensor(kernel), 'kernel2': torch.FloatTensor(kernel2),
                'sinc_kernel': sinc_kernel, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)