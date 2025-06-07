import os
import shutil
import random

# === 配置路径 ===
image_dir = "images"
label_dir = "labels"
train_ratio = 0.75

# === 支持的图片扩展名 ===
img_exts = ['.jpg', '.jpeg', '.png', '.bmp']

# === 收集所有图像文件名（无扩展名）===
all_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir)
             if os.path.splitext(f)[1].lower() in img_exts]

# === 打乱并划分 ===
random.shuffle(all_files)
split_index = int(len(all_files) * train_ratio)
train_files = all_files[:split_index]
val_files = all_files[split_index:]

# === 创建子文件夹 ===
os.makedirs(os.path.join(image_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(image_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(label_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(label_dir, "val"), exist_ok=True)

# === 拷贝函数 ===
def move_files(file_list, subset):
    for name in file_list:
        # === 移动图片 ===
        for ext in img_exts:
            src_img = os.path.join(image_dir, name + ext)
            if os.path.exists(src_img):
                dst_img = os.path.join(image_dir, subset, name + ext)
                shutil.move(src_img, dst_img)
                break

        # === 移动标签 ===
        src_label = os.path.join(label_dir, name + '.txt')
        dst_label = os.path.join(label_dir, subset, name + '.txt')
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"[警告] 缺少标签文件: {name}.txt")

# === 执行移动 ===
move_files(train_files, 'train')
move_files(val_files, 'val')

print("done")
