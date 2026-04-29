import numpy as np
import cv2
import os
from glob import glob

# 输入和输出路径
input_dir = './datasets/S20-Chest/paired_gt_npy'
output_dir = './results_vis3'
os.makedirs(output_dir, exist_ok=True)

npy_files = glob(os.path.join(input_dir, '*.npy'))

for path in npy_files:
    name = os.path.basename(path).replace('.npy', '.png')

    # 读取数据并限幅到 [0, 1]
    data = np.load(path).astype(np.float32)
    data = np.clip(data, 0, 1)

    # 转换为 8位灰度图 (0-255)
    img_8bit = (data * 255.0).round().astype(np.uint8)

    # 保存图片
    cv2.imwrite(os.path.join(output_dir, name), img_8bit)
    print(f"已保存: {name}")

print(f"全部转换完成，请查看目录: {output_dir}")