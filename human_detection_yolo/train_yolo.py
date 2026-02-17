from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # use n.pt if GPU is weak

    model.train(
        data="dataset/yolo_human/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="models/yolo",
        name="human_detector"
    )

if __name__ == "__main__":
    main()