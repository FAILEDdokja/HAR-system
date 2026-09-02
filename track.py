import cv2
import math
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

# Store previous center position for each object
previous_positions = {}

# Minimum pixel movement to consider an object moving
MOVEMENT_THRESHOLD = 5

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        classes=[0, 39]
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = box

            # Calculate center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (center_x, center_y)

            # Calculate movement
            if track_id in previous_positions:

                previous_x, previous_y = previous_positions[track_id]

                distance = math.sqrt(
                    (center_x - previous_x) ** 2 +
                    (center_y - previous_y) ** 2
                )

                if distance > MOVEMENT_THRESHOLD:
                    movement_status = "MOVING"
                else:
                    movement_status = "STATIONARY"

            else:
                distance = 0
                movement_status = "NEW"

            # Save current position
            previous_positions[track_id] = current_position

            # Draw center point
            cv2.circle(
                frame,
                current_position,
                5,
                (0, 255, 0),
                -1
            )

            # Display movement information
            label = f"ID {track_id} | {movement_status} | {distance:.1f}px"

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Draw bounding boxes
    annotated_frame = result.plot()

    # Add our movement information on top
    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            if track_id in previous_positions:
                center = previous_positions[track_id]

                cv2.circle(
                    annotated_frame,
                    center,
                    5,
                    (0, 255, 0),
                    -1
                )

    cv2.imshow("SIH26174 - Movement Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
