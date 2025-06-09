import os
import time

from ultralytics import YOLO

model = YOLO(model='light.pt')


def process_img(img_path):
    # 执行预测
    results = model.predict(source=img_path, save=False, show=False, verbose=False)

    output_list = []

    # 通常每张图片返回一个结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)
            w = x2 - x1
            h = y2 - y1
            output_list.append({"x": x1, "y": y1, "w": w, "h": h})

    return output_list
if __name__ == '__main__':
    img_path = '1.jpg'  # 替换为你的图片路径
    result = process_img(img_path)
    print(result)
