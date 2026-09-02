import cv2
import math
from ultralytics import YOLO

# Models
object_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")

# Camera
cap = cv2.VideoCapture(0)

# Distance at which we consider the hand close to an object
INTERACTION_THRESHOLD = 80


def point_to_box_distance(point, box):
    """
    Calculate the minimum distance between a point
    and a bounding box.
    """
    px, py = point
    x1, y1, x2, y2 = box

    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))

    return math.sqrt(
        (px - closest_x) ** 2 +
        (py - closest_y) ** 2
    )


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    # ------------------------------------------------
    # 1. Detect and track objects
    # ------------------------------------------------

    object_results = object_model.track(
        frame,
        persist=True,
        tracker="custom_bytetrack.yaml",
        conf=0.35,
        classes=[39],  # bottle
        verbose=False
    )

    object_result = object_results[0]

    # ------------------------------------------------
    # 2. Detect human pose
    # ------------------------------------------------

    pose_results = pose_model.track(
        frame,
        persist=True,
        tracker="custom_bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    pose_result = pose_results[0]

    # Start with object detections
    output = object_result.plot()

    # ------------------------------------------------
    # 3. Get object bounding boxes
    # ------------------------------------------------

    objects = []

    if object_result.boxes.id is not None:

        boxes = object_result.boxes.xyxy.cpu().numpy()
        ids = object_result.boxes.id.cpu().numpy().astype(int)

        for box, object_id in zip(boxes, ids):

            objects.append({
                "id": object_id,
                "box": box
            })

    # ------------------------------------------------
    # 4. Get person's wrists
    # ------------------------------------------------

    if pose_result.keypoints is not None:

        keypoints = pose_result.keypoints.xy.cpu().numpy()

        for person_keypoints in keypoints:

            # COCO pose indices:
            # 9  = left wrist
            # 10 = right wrist

            left_wrist = person_keypoints[9]
            right_wrist = person_keypoints[10]

            wrists = [
                ("LEFT HAND", left_wrist),
                ("RIGHT HAND", right_wrist)
            ]

            # ----------------------------------------
            # 5. Compare wrists with objects
            # ----------------------------------------

            for hand_name, wrist in wrists:

                wrist_x = int(wrist[0])
                wrist_y = int(wrist[1])

                # Ignore invalid keypoints
                if wrist_x <= 0 or wrist_y <= 0:
                    continue

                # Draw wrist
                cv2.circle(
                    output,
                    (wrist_x, wrist_y),
                    7,
                    (0, 255, 0),
                    -1
                )

                for obj in objects:

                    object_id = obj["id"]
                    box = obj["box"]

                    distance = point_to_box_distance(
                        (wrist_x, wrist_y),
                        box
                    )

                    # --------------------------------
                    # Interaction state
                    # --------------------------------

                    if distance < INTERACTION_THRESHOLD:

                        status = "INTERACTING"

                    else:

                        status = "NO INTERACTION"

                    # Object center
                    x1, y1, x2, y2 = box

                    object_center = (
                        int((x1 + x2) / 2),
                        int((y1 + y2) / 2)
                    )

                    # Draw line between wrist and object
                    cv2.line(
                        output,
                        (wrist_x, wrist_y),
                        object_center,
                        (255, 255, 255),
                        2
                    )

                    # Display information
                    text = (
                        f"{hand_name} -> "
                        f"Bottle {object_id}: "
                        f"{distance:.1f}px "
                        f"{status}"
                    )

                    cv2.putText(
                        output,
                        text,
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

    # ------------------------------------------------
    # Display
    # ------------------------------------------------

    cv2.imshow(
        "SIH26174 - Hand Object Interaction",
        output
    )

    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
