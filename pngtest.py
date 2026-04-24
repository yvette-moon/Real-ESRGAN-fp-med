import os
import pydicom
import numpy as np
import cv2

# ===== 参数配置 =====
input_dir = r"D:\model\Real-ESRGAN-master1\datasets\S20-Chest\val\dcm"   # DICOM文件夹
output_dir = r"D:\model\Real-ESRGAN-master1\datasets\S20-Chest\val\gt"   # 输出PNG文件夹

WL = -450
WW = 1300




os.makedirs(output_dir, exist_ok=True)


def windowing(img, WL, WW):
    """
    应用窗宽窗位
    """
    lower = WL - WW / 2
    upper = WL + WW / 2

    img = (img - lower) / (upper - lower)
    img = np.clip(img, 0, 1)
    return img


def dcm_to_png(dcm_path, save_path):
    try:
        ds = pydicom.dcmread(dcm_path)

        # 读取像素
        img = ds.pixel_array.astype(np.float32)

        # 获取 HU 转换参数
        slope = getattr(ds, "RescaleSlope", 1)
        intercept = getattr(ds, "RescaleIntercept", 0)

        img = img * slope + intercept

        # windowing
        img = windowing(img, WL, WW)

        # 转 8-bit
        img = (img * 255).astype(np.uint8)

        # 保存
        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"处理失败: {dcm_path}, 错误: {e}")


# ===== 批量处理 =====
files = os.listdir(input_dir)

for i, file in enumerate(files):
    dcm_path = os.path.join(input_dir, file)

    # 跳过非文件
    if not os.path.isfile(dcm_path):
        continue

    save_name = f"{i:05d}.png"
    save_path = os.path.join(output_dir, save_name)

    dcm_to_png(dcm_path, save_path)

print("全部转换完成！")