import sys
import os

# 强制将项目根目录加入到 Python 搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.utils.registry import DATASET_REGISTRY

try:
    from basicsr.utils.img_process_util import filter2D, USMSharp
except ImportError:
    from basicsr.utils import filter2D

    try:
        from basicsr.utils import USMSharp
    except ImportError:
        USMSharp = None

# 必须显式导入这个文件，触发它的 register() 装饰器！
import realesrgan.data.npy_realesrgan_dataset


def main():
    parser = argparse.ArgumentParser(description="离线生成 Real-ESRGAN 二阶退化 NPY 矩阵 (包含对齐GT保存)")
    parser.add_argument('--input_gt_dir', type=str, required=True, help='高精度 GT NPY 目录')
    parser.add_argument('--output_lq_dir', type=str, required=True, help='输出 LQ NPY 目录')
    # 【核心新增】必须加上这个参数，用来保存被裁切后真正对齐的 GT
    parser.add_argument('--output_gt_dir', type=str, required=True, help='输出严格对齐的 GT NPY 目录')
    parser.add_argument('--meta_info_gt', type=str, required=False, help='GT的 meta_info 文件路径')
    parser.add_argument('--scale', type=int, default=4, help='最终的下采样倍数')
    args = parser.parse_args()

    # 创建 LQ 和 GT 两个输出文件夹
    os.makedirs(args.output_lq_dir, exist_ok=True)
    os.makedirs(args.output_gt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 准备 USM 锐化器 (适配你的 import 逻辑)
    if USMSharp is not None:
        usm_sharpener = USMSharp().to(device)
    else:
        usm_sharpener = lambda x: x

    # 2. 配置退化参数字典
    opt = {
        'name': 'OfflineDegradation',
        'type': 'NPYRealESRGANDataset',
        'dataroot_gt': args.input_gt_dir,

        # 🚨 核心修复：这里必须叫 'meta_info'，不能带 _file
        'meta_info': args.meta_info_gt,

        'io_backend': {'type': 'disk'},

        # 离线处理全图
        'gt_size': 0,
        'use_hflip': False, 'use_rot': False,

        # USM 锐化开关
        'l1_gt_usm': True,
        'percep_gt_usm': True,
        'gan_gt_usm': False,

        # 模糊核基础参数
        'blur_kernel_size': 21,
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob': 0.1,
        'blur_sigma': [0.2, 3],
        'betag_range': [0.5, 4],
        'betap_range': [1, 2],

        # 第二阶段核参数（补齐）
        'blur_kernel_size2': 21,
        'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob2': 0.1,
        'blur_sigma2': [0.2, 1.5],
        'betag_range2': [0.5, 4],
        'betap_range2': [1, 2],
        'final_sinc_prob': 0.8,

        # 第一阶退化
        'resize_prob': [0.2, 0.7, 0.1],
        'resize_range': [0.15, 1.5],
        'gaussian_noise_prob': 0.5,
        'noise_range': [1, 30],
        'poisson_scale_range': [0.05, 3],
        'gray_noise_prob': 1,
        'jpeg_range': [80, 100],

        # 第二阶退化
        'second_blur_prob': 0.8,
        'resize_prob2': [0.3, 0.4, 0.3],
        'resize_range2': [0.3, 1.2],
        'gaussian_noise_prob2': 0.5,
        'noise_range2': [1, 25],
        'poisson_scale_range2': [0.05, 2.5],
        'gray_noise_prob2': 1,
        'jpeg_range2': [80, 100],

        'scale': args.scale
    }

    # 3. 初始化 Dataset
    dataset = DATASET_REGISTRY.get('NPYRealESRGANDataset')(opt)
    print(f"Dataset 挂载成功，共检测到 {len(dataset)} 个 GT 矩阵文件。开始批量生成成对数据...")

    for i in tqdm(range(len(dataset))):
        data = dataset[i]

        # 提取原文件名
        gt_path = data['gt_path']
        base_name = os.path.basename(gt_path)

        # 转为 Tensor，增加 Batch 维度，推入 GPU
        gt = data['gt'].unsqueeze(0).to(device)
        kernel1 = data['kernel1'].unsqueeze(0).to(device)
        kernel2 = data['kernel2'].unsqueeze(0).to(device)
        sinc_kernel = data['sinc_kernel'].unsqueeze(0).to(device)

        ori_h, ori_w = gt.size()[2:4]

        # ==================== 执行高阶退化核心流水线 ====================

        # 0. USM 锐化
        if opt['l1_gt_usm']:
            out = usm_sharpener(gt)
        else:
            out = gt

        # 1. 第一次模糊
        out = filter2D(out, kernel1)

        # 2. 第一次随机缩放
        scale = 1
        mode = np.random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # 3. 第一次加噪
        if np.random.uniform() < opt['gaussian_noise_prob']:
            noise_level = np.random.uniform(opt['noise_range'][0], opt['noise_range'][1]) / 255.0
            noise = torch.randn_like(out) * noise_level
            out = out + noise

        # 4. 第二次模糊
        if np.random.uniform() < opt['second_blur_prob']:
            out = filter2D(out, kernel2)

        # 5. 第二次随机缩放
        scale = 1
        mode = np.random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # 6. 第二次加噪
        if np.random.uniform() < opt['gaussian_noise_prob2']:
            noise_level = np.random.uniform(opt['noise_range2'][0], opt['noise_range2'][1]) / 255.0
            noise = torch.randn_like(out) * noise_level
            out = out + noise

        # 7. 最终精准下采样 (在这里真正缩小 4 倍)
        target_h, target_w = ori_h // args.scale, ori_w // args.scale
        mode = np.random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(target_h, target_w), mode=mode)

        # 8. Sinc 滤波模拟振铃效应
        if np.random.uniform() < opt['sinc_prob']:
            out = filter2D(out, sinc_kernel)

        # 防止像素溢出
        out = torch.clamp(out, 0.0, 1.0)

        # ==================== 存盘阶段 (极度重要：成对保存) ====================

        # 存 LQ (经过模糊、加噪、下采样的图)
        lq_npy = out.squeeze(0).cpu().numpy()
        if lq_npy.ndim == 3:
            lq_npy = np.transpose(lq_npy, (1, 2, 0))
        save_path_lq = os.path.join(args.output_lq_dir, base_name)
        np.save(save_path_lq, lq_npy)

        # 存 GT (从 dataloader 拿出来的，物理尺寸和裁剪位置跟 LQ 严格绑定的高清图)
        gt_npy = gt.squeeze(0).cpu().numpy()
        if gt_npy.ndim == 3:
            gt_npy = np.transpose(gt_npy, (1, 2, 0))
        save_path_gt = os.path.join(args.output_gt_dir, base_name)
        np.save(save_path_gt, gt_npy)

    print("✅ 成对离线数据生成完毕，对齐的 GT 和 LQ 已全部分别保存。")


if __name__ == '__main__':
    main()