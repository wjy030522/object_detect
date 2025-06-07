# -*- coding: utf-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model='best.pt')
    
    # 执行预测，返回结果
    results = model.predict(source='3/3/', save=True, show=True)

    # 遍历结果
    for result in results:
        boxes = result.boxes  # 边界框信息
        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # 边界框左上和右下坐标：[x1, y1, x2, y2]
            conf = box.conf[0].item()     # 置信度
            cls = int(box.cls[0].item())  # 类别索引
            print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")
