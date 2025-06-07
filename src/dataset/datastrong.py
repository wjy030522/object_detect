import os
import cv2
import shutil
from PIL import Image
import albumentations as A

# === 配置路径 ===
image_dir = "images"   # 原始图片目录
label_dir = "labels"   # YOLO标签目录
suffixes = ['_light', '_blur']

# === 定义光照增强 ===
light_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
])

# === 定义模糊增强 ===
blur_transform = A.OneOf([
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.MotionBlur(blur_limit=5, p=1.0),
], p=1.0)

# === 支持的图片后缀 ===
img_exts = ['.jpg', '.jpeg', '.png', '.bmp']

# === 遍历图像文件夹并增强 + 标签复制 ===
for filename in os.listdir(image_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in img_exts:
        continue

    img_path = os.path.join(image_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"[跳过] 无法读取图片: {filename}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === 增强类型及对应操作 ===
    transforms = {
        '_light': light_transform,
        '_blur': blur_transform
    }

    for suffix, transform in transforms.items():
        # 增强图像保存
        aug_image = transform(image=image)['image']
        new_img_name = name + suffix + ext
        save_img_path = os.path.join(image_dir, new_img_name)
        Image.fromarray(aug_image).save(save_img_path)

        # 对应标签复制
        src_txt_path = os.path.join(label_dir, name + '.txt')
        dst_txt_path = os.path.join(label_dir, name + suffix + '.txt')

        if os.path.exists(src_txt_path):
            shutil.copyfile(src_txt_path, dst_txt_path)
        else:
            print(f"[警告] 标签文件不存在: {src_txt_path}")

print("done")
