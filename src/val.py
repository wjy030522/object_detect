from ultralytics import YOLO

def evaluate_model(weights_path, data_yaml, img_size=640, batch_size=16, device='cuda:0'):
    model = YOLO(weights_path)
    
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=0
    )

    # 输出关键评估指标
    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return metrics


# 示例用法
if __name__ == "__main__":
    evaluate_model(
        weights_path="normal.pt",
        data_yaml="tennis.yaml",
        img_size=640,
        batch_size=16,
        device=''
    )
