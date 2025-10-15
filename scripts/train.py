from ultralytics import YOLO
import os
import yaml

CFG='configs/config.yaml'
CFG_YOLO = 'configs/person_ball.yaml'

def main():
    # Load configuration
    CFG_PATH = os.path.join(os.getcwd(), CFG)
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Load a pretrained YOLOv8 model
    model = YOLO("resources/weights/yolov8s.pt")

    # Train the model
    model.train(
        data=CFG_YOLO,
        epochs=cfg['training']['epochs'],
        imgsz=cfg['training']['image_size'],
        batch=cfg['training']['batch_size'],
        device=0,  # GPU 0
        name="yolov8_person_ball",
        workers=cfg['training']['workers'],
        pretrained=True
    )

    # Evaluate
    model.val()

    # Export to ONNX
    model.export(format="onnx")

if __name__ == "__main__":
    main()