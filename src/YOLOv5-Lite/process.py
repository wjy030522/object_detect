import os
from PIL import Image
from detect import run_detection  # 从封装后的 detect.py 中导入

def process_single_image(img_path):
    # 获取图片文件名和目录
    fname = os.path.basename(img_path)
    print(f'Processing {fname}...')

    # 运行 YOLO 检测
    run_detection(img_path=img_path)

    # 获取图像尺寸
    with Image.open(img_path) as im:
        img_width, img_height = im.size

    # 构造检测结果路径（默认输出在 runs/detect/exp/labels/）
    txt_name = os.path.splitext(fname)[0] + '.txt'
    txt_path = f'runs/detect/exp/labels/{txt_name}'

    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x_center, y_center, w, h = map(float, parts[:5])
                    x = int(x_center * img_width - w * img_width / 2)
                    y = int(y_center * img_height - h * img_height / 2)
                    w = int(w * img_width)
                    h = int(h * img_height)
                    boxes.append({"x": x, "y": y, "w": w, "h": h})

    return boxes  # 返回单张图片的检测框列表

# 示例调用方式
if __name__ == '__main__':
    img_path = '2.jpg'
    result = process_single_image(img_path)
    print(result)
