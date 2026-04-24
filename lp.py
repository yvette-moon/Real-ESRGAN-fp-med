import cv2
import os

gt_dir = r'D:/model/Real-ESRGAN-master1/datasets/S20-Chest/val/gt'
lq_dir = r'D:/model/Real-ESRGAN-master1/datasets/S20-Chest/val/lq'

os.makedirs(lq_dir, exist_ok=True)

for name in os.listdir(gt_dir):
    gt_path = os.path.join(gt_dir, name)
    lq_path = os.path.join(lq_dir, name)

    img = cv2.imread(gt_path)
    if img is None:
        print(f"读取失败: {name}")
        continue

    h, w = img.shape[:2]

    # ↓↓↓ 降采样（模拟低质 LQ）
    lq = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(lq_path, lq)

print("完成！")