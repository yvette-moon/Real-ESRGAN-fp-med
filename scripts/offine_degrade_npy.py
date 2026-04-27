import sys
import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import random
import math

# 强制将项目根目录加入到 Python 搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入基础的核生成工具
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels

try:
    from basicsr.utils.img_process_util import filter2D, USMSharp
except ImportError:
    from basicsr.utils import filter2D

    try:
        from basicsr.utils import USMSharp
    except ImportError:
        USMSharp = None


def read_npy_to_tensor(path):
    """直接读取NPY文件并转为 [1, C, H, W] 的 Tensor"""
    arr = np.load(path).astype(np.float32)

    # 兼容处理各种维度 (H*W, H*W*C, C*H*W)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW 转为 HWC

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # 截断到 0-1 范围，防止异常值
    arr = np.clip(arr, 0.0, 1.0)

    # HWC 转换回 CHW 供 PyTorch 使用
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)  # 增加 Batch 维度: [1, 3, H, W]
    return tensor


def center_crop_to_scale_multiple(tensor, scale):
    """将 H/W 裁剪到能整除 scale，确保 GT/LQ 严格 4 倍关系"""
    _, _, h, w = tensor.shape
    new_h = (h // scale) * scale
    new_w = (w // scale) * scale
    if new_h == h and new_w == w:
        return tensor
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return tensor[:, :, top:top + new_h, left:left + new_w]


def generate_kernels(opt):
    """独立生成当前图片专属的退化模糊核"""
    kernel_range = [7, 9, 11, 13, 15, 17, 19, 21]

    # 1. 第一次模糊核
    kernel_size1 = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob']:
        omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size1 < 13 else np.random.uniform(np.pi / 5, np.pi)
        kernel1 = circular_lowpass_kernel(omega_c, kernel_size1, pad_to=False)
    else:
        kernel1 = random_mixed_kernels(
            opt['kernel_list'], opt['kernel_prob'], kernel_size1,
            opt['blur_sigma'], opt['blur_sigma'], [-math.pi, math.pi],
            opt['betag_range'], opt['betap_range'], noise_range=None)
    pad_size1 = (21 - kernel_size1) // 2
    kernel1 = np.pad(kernel1, ((pad_size1, pad_size1), (pad_size1, pad_size1)))

    # 2. 第二次模糊核
    kernel_size2 = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob2']:
        omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size2 < 13 else np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size2, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            opt['kernel_list2'], opt['kernel_prob2'], kernel_size2,
            opt['blur_sigma2'], opt['blur_sigma2'], [-math.pi, math.pi],
            opt['betag_range2'], opt['betap_range2'], noise_range=None)
    pad_size2 = (21 - kernel_size2) // 2
    kernel2 = np.pad(kernel2, ((pad_size2, pad_size2), (pad_size2, pad_size2)))

    # 3. 最后的 Sinc 核
    if np.random.uniform() < opt['final_sinc_prob']:
        kernel_size3 = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size3, pad_to=21)
    else:
        sinc_kernel = np.zeros((21, 21), dtype=np.float32)
        sinc_kernel[10, 10] = 1.0

    return torch.FloatTensor(kernel1), torch.FloatTensor(kernel2), torch.FloatTensor(sinc_kernel)


def main():
    parser = argparse.ArgumentParser(description="直读直接退化 NPY 矩阵 (告别 Dataset 报错)")
    parser.add_argument('--input_gt_dir', type=str, required=True, help='高精度 GT NPY 目录')
    parser.add_argument('--output_lq_dir', type=str, required=True, help='输出 LQ NPY 目录')
    parser.add_argument('--output_gt_dir', type=str, required=True, help='输出严格对齐的 GT NPY 目录')
    # 保留这个参数以防你旧命令报错，但代码不再需要它了
    parser.add_argument('--meta_info_gt', type=str, required=False, help='(不再需要)')
    parser.add_argument('--scale', type=int, default=4, help='最终的下采样倍数')
    args = parser.parse_args()

    # 创建输出文件夹
    os.makedirs(args.output_lq_dir, exist_ok=True)
    os.makedirs(args.output_gt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 准备 USM 锐化器
    if USMSharp is not None:
        usm_sharpener = USMSharp().to(device)
    else:
        usm_sharpener = lambda x: x

    # 退化参数配置
    opt = {
        'l1_gt_usm': True,
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob': 0.1, 'blur_sigma': [0.2, 3], 'betag_range': [0.5, 4], 'betap_range': [1, 2],

        'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob2': 0.1, 'blur_sigma2': [0.2, 1.5], 'betag_range2': [0.5, 4], 'betap_range2': [1, 2],
        'final_sinc_prob': 0.8,

        'gaussian_noise_prob': 0.5, 'noise_range': [1, 30],
        'gaussian_noise_prob2': 0.5, 'noise_range2': [1, 25],
    }

    # 获取所有 NPY 文件列表
    file_list = [f for f in os.listdir(args.input_gt_dir) if f.endswith('.npy')]
    print(f"✅ 成功找到 {len(file_list)} 个 NPY 文件。开始直接退化...")

    for file_name in tqdm(file_list):
        gt_path = os.path.join(args.input_gt_dir, file_name)

        # 1. 直接读取数据，自带防崩保护
        try:
            gt = read_npy_to_tensor(gt_path).to(device)
        except Exception as e:
            print(f"\n⚠️ 无法读取文件 {file_name}: {e}，已跳过")
            continue

        # 关键：先裁剪到可整除尺寸
        gt = center_crop_to_scale_multiple(gt, args.scale)

        ori_h, ori_w = gt.shape[2], gt.shape[3]
        if ori_h == 0 or ori_w == 0:
            print(f"\n⚠️ 发现空尺寸图片 [1, 3, {ori_h}, {ori_w}] -> {file_name}，已跳过")
            continue

        # 2. 生成随机模糊核
        k1, k2, sinc_k = generate_kernels(opt)
        kernel1 = k1.unsqueeze(0).to(device)
        kernel2 = k2.unsqueeze(0).to(device)
        sinc_kernel = sinc_k.unsqueeze(0).to(device)

        # ==================== 执行高阶退化流水线 ====================
        out = usm_sharpener(gt) if opt['l1_gt_usm'] else gt

        # 第一次模糊与加噪 (去掉了中间无关的缩放)
        out = filter2D(out, kernel1)
        if np.random.uniform() < opt['gaussian_noise_prob']:
            noise_level = np.random.uniform(opt['noise_range'][0], opt['noise_range'][1]) / 255.0
            out = out + torch.randn_like(out) * noise_level

        # 第二次模糊与加噪
        if np.random.uniform() < 0.8:  # second_blur_prob
            out = filter2D(out, kernel2)
        if np.random.uniform() < opt['gaussian_noise_prob2']:
            noise_level = np.random.uniform(opt['noise_range2'][0], opt['noise_range2'][1]) / 255.0
            out = out + torch.randn_like(out) * noise_level

        # 精准下采样 (严格 4 倍)
        target_h, target_w = ori_h // args.scale, ori_w // args.scale
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(target_h, target_w), mode=mode)

        # Sinc 滤波
        if np.random.uniform() < opt['sinc_prob']:
            out = filter2D(out, sinc_kernel)

        out = torch.clamp(out, 0.0, 1.0)

        # ==================== 存盘 ====================
        # 存 LQ
        lq_npy = out.squeeze(0).cpu().numpy()
        lq_npy = np.transpose(lq_npy, (1, 2, 0)) if lq_npy.ndim == 3 else lq_npy
        np.save(os.path.join(args.output_lq_dir, file_name), lq_npy)

        # 存对齐的 GT
        gt_npy = gt.squeeze(0).cpu().numpy()
        gt_npy = np.transpose(gt_npy, (1, 2, 0)) if gt_npy.ndim == 3 else gt_npy
        np.save(os.path.join(args.output_gt_dir, file_name), gt_npy)

    print("✅ 成对离线��据生成完毕，对齐的 GT 和 LQ 已全部分别保存。")


if __name__ == '__main__':
    main()