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

    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    speed = metrics.speed

    mean_precision = metrics.box.mp
    mean_recall = metrics.box.mr

    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (mean): {mean_precision:.4f}")
    print(f"Recall (mean): {mean_recall:.4f}")
    print(f"Parameters (M): {params:.2f}")
    print(f"Inference Speed (ms/img): {speed['inference']:.2f}")

    return metrics

if __name__ == "__main__":
    evaluate_model(
        weights_path="normal.pt",
        data_yaml="tennis.yaml",
        img_size=640,
        batch_size=16,
        device=''
    )
