from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run detection on an image/video/webcam
results = model.predict(
    source=0,
    show=True,
    conf=0.5
)
