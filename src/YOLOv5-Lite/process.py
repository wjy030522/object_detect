import os
import json
from PIL import Image
from detect import run_detection  # 从封装后的 detect.py 中导入

def process_img(img_folder):
    results = {}

    for fname in os.listdir(img_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(img_folder, fname)
            print(f'Processing {fname}...')

            # 运行 YOLO 检测
            run_detection(img_path=img_path)

            # 获取图像尺寸
            with Image.open(img_path) as im:
                img_width, img_height = im.size

            # 读取检测结果
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
            results[fname] = boxes

    # 保存总结果到 txt 文件
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))

    print("done")
if __name__ == '__main__':
    process_img('testimages')  # 输入的是图片目录，不是单图
