import cv2
from ultralytics import YOLO

# YOLO pose model
model = YOLO("yolo11n-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    # Pose estimation
    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        tracker="custom_bytetrack.yaml",
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "SIH26174 - Pose Detection",
        annotated_frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
