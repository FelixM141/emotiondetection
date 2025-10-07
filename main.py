from ultralytics import YOLO
import os

YAML_path = ".\Emotion.v1i.yolov8-obb/data.yaml"

model = YOLO("yolov8n.pt")
results = model.train(
    data=YAML_path,
    epochs=20,
    imgsz =640,
    batch=20,
    patience=5,
    device=dml
)
print(results)
