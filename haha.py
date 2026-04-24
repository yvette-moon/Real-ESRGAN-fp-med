import os
import argparse
import glob
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def main():
    parser = argparse.ArgumentParser(description='计算 SR 图像的 MSE, PSNR, SSIM 和 LPIPS')
    parser.add_argument('--folder_gt', type=str, required=True, help='Ground Truth (真实高清原图) 文件夹路径')
    parser.add_argument('--folder_gen', type=str, required=True, help='生成的超分图片 文件夹路径 (如 results)')
    parser.add_argument('--suffix', type=str, default='_out',
                        help='生成图片相对于GT的后缀，默认是 _out (例如原图01.png, 生成图01_out.png)')
    args = parser.parse_args()

    # 初始化 LPIPS 模型 (通常使用 alexnet 作为 backbone)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading LPIPS model on {device}...")
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    # 获取 GT 文件夹下所有的图片
    gt_paths = sorted(glob.glob(os.path.join(args.folder_gt, '*')))

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    valid_images = 0

    print(f"{'Image Name':<20} | {'MSE':<8} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8}")
    print("-" * 65)

    for gt_path in gt_paths:
        basename = os.path.basename(gt_path)
        name, ext = os.path.splitext(basename)

        # 寻找对应的生成图片
        gen_path = os.path.join(args.folder_gen, f"{name}{args.suffix}.jpg")

        if not os.path.exists(gen_path):
            print(f"Warning: 找不到对应的生成图片 {gen_path}，已跳过。")
            continue

        # 读取图片 (OpenCV默认读取为BGR格式)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        img_gen = cv2.imread(gen_path, cv2.IMREAD_COLOR)

        # 确保尺寸一致 (有些模型可能会在边缘padding，如果尺寸不一致需要裁剪或缩放)
        if img_gt.shape != img_gen.shape:
            print(f"Warning: 图片尺寸不一致 {name} (GT: {img_gt.shape}, Gen: {img_gen.shape})，调整为GT尺寸。")
            img_gen = cv2.resize(img_gen, (img_gt.shape[1], img_gt.shape[0]))

        # 转换为 RGB 格式
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gen_rgb = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

        # ==========================
        # 1. 计算 MSE, PSNR, SSIM (使用 skimage)
        # ==========================
        # data_range=255 表示像素值范围是 0-255；channel_axis=2 表示第3个维度是颜色通道
        # 1. 计算 MSE, PSNR, SSIM
        # 将图片归一化到 0~1 的浮点数，再计算指标
        img_gt_norm = img_gt_rgb.astype(np.float32) / 255.0
        img_gen_norm = img_gen_rgb.astype(np.float32) / 255.0

        mse_val = mean_squared_error(img_gt_norm, img_gen_norm)
        # 注意：因为值域变成了0~1，所以这里的 data_range 必须改成 1.0
        psnr_val = peak_signal_noise_ratio(img_gt_norm, img_gen_norm, data_range=1.0)
        ssim_val = structural_similarity(img_gt_norm, img_gen_norm, data_range=1.0, channel_axis=2)

        # ==========================
        # 2. 计算 LPIPS (使用 lpips 库)
        # ==========================
        # LPIPS 库要求输入是 PyTorch Tensor，且值域在 [-1, 1] 之间，形状为 (N, C, H, W)
        img_gt_tensor = (torch.from_numpy(img_gt_rgb.transpose(2, 0, 1)).float() / 127.5) - 1.0
        img_gen_tensor = (torch.from_numpy(img_gen_rgb.transpose(2, 0, 1)).float() / 127.5) - 1.0

        # 增加 Batch 维度 (1, C, H, W) 并移到对应��设备
        img_gt_tensor = img_gt_tensor.unsqueeze(0).to(device)
        img_gen_tensor = img_gen_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            lpips_val = loss_fn_lpips(img_gen_tensor, img_gt_tensor).item()

        # 累加结果
        total_mse += mse_val
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_lpips += lpips_val
        valid_images += 1

        print(f"{name:<20} | {mse_val:<10.6f} | {psnr_val:<10.4f} | {ssim_val:<10.6f} | {lpips_val:<10.6f}")

    # ==========================
    # 输出平均指标
    # ==========================
    if valid_images > 0:
        print("-" * 65)
        print(
            f"{'Average':<20} | {total_mse / valid_images:<8.2f} | {total_psnr / valid_images:<8.2f} | {total_ssim / valid_images:<8.4f} | {total_lpips / valid_images:<8.4f}")
    else:
        print("未找到任何匹配的图片对进行计算。")


if __name__ == '__main__':
    main()