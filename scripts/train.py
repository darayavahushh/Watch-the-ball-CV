from ultralytics import YOLO

CFG='configs/config.yaml'
CFG_YOLO = 'configs/person_ball.yaml'

def main():
    # Load a pretrained YOLOv8 model
    model = YOLO("resources/weights/yolov8s.pt")

    # Train the model
    model.train(
        data=CFG_YOLO,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # GPU 0
        name="yolov8_person_ball",
        workers=4,
        pretrained=True
    )

    # Evaluate
    model.val()

    # Export to ONNX
    model.export(format="onnx")

if __name__ == "__main__":
    main()