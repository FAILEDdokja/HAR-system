import cv2
import math
from collections import deque
from ultralytics import YOLO


# =========================================================
# MODEL
# =========================================================

model = YOLO("yolo11n.pt")


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()


# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE = 0.35

# Maximum distance for associating a new detection
# with an existing logical object.
MATCH_DISTANCE = 120

# Number of frames we keep an object alive after
# YOLO temporarily loses it.
MAX_MISSED_FRAMES = 15

# Position history
HISTORY_SIZE = 20


# =========================================================
# LOGICAL OBJECTS
# =========================================================

objects = {}

next_object_id = 1


# =========================================================
# DISTANCE
# =========================================================

def distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


# =========================================================
# CREATE OBJECT
# =========================================================

def create_object(center, box, tracker_id):

    global next_object_id

    object_id = next_object_id

    next_object_id += 1

    objects[object_id] = {

        "center": center,

        "box": box,

        "tracker_id": tracker_id,

        "missed": 0,

        "history": deque(
            [center],
            maxlen=HISTORY_SIZE
        )
    }

    return object_id


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to read camera")

        break


    # =====================================================
    # YOLO TRACKING
    # =====================================================

    results = model.track(

        frame,

        persist=True,

        tracker="custom_bytetrack.yaml",

        conf=CONFIDENCE,

        classes=[39],

        verbose=False
    )

    result = results[0]


    # =====================================================
    # CURRENT DETECTIONS
    # =====================================================

    detections = []


    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        tracker_ids = (
            result
            .boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, tracker_id in zip(
            boxes,
            tracker_ids
        ):

            x1, y1, x2, y2 = box

            center = (

                int((x1 + x2) / 2),

                int((y1 + y2) / 2)
            )


            detections.append({

                "center": center,

                "box": box,

                "tracker_id": tracker_id
            })


    # =====================================================
    # MARK ALL EXISTING OBJECTS AS MISSED
    # =====================================================

    for object_id in objects:

        objects[object_id]["missed"] += 1


    # =====================================================
    # MATCH DETECTIONS TO LOGICAL OBJECTS
    # =====================================================

    matched_objects = set()


    for detection in detections:

        center = detection["center"]

        tracker_id = detection["tracker_id"]


        best_object = None

        best_distance = float("inf")


        # -------------------------------------------------
        # First preference:
        # same YOLO tracker ID
        # -------------------------------------------------

        for object_id, obj in objects.items():

            if object_id in matched_objects:

                continue


            if obj["tracker_id"] == tracker_id:

                d = distance(

                    obj["center"],

                    center
                )


                if d < best_distance:

                    best_distance = d

                    best_object = object_id


        # -------------------------------------------------
        # Second preference:
        # nearest logical object
        # -------------------------------------------------

        if best_object is None:

            for object_id, obj in objects.items():

                if object_id in matched_objects:

                    continue


                # Don't match an object that has been
                # missing for too long.

                if (
                    obj["missed"]
                    > MAX_MISSED_FRAMES
                ):

                    continue


                d = distance(

                    obj["center"],

                    center
                )


                if (
                    d < MATCH_DISTANCE
                    and
                    d < best_distance
                ):

                    best_distance = d

                    best_object = object_id


        # -------------------------------------------------
        # Update existing object
        # -------------------------------------------------

        if best_object is not None:

            obj = objects[best_object]


            obj["center"] = center

            obj["box"] = detection["box"]

            obj["tracker_id"] = tracker_id

            obj["missed"] = 0

            obj["history"].append(center)


            matched_objects.add(best_object)


        # -------------------------------------------------
        # Create new logical object
        # -------------------------------------------------

        else:

            new_id = create_object(

                center,

                detection["box"],

                tracker_id
            )


            matched_objects.add(new_id)


    # =====================================================
    # REMOVE OBJECTS LOST FOR TOO LONG
    # =====================================================

    objects_to_delete = []


    for object_id, obj in objects.items():

        if (
            obj["missed"]
            > MAX_MISSED_FRAMES
        ):

            objects_to_delete.append(object_id)


    for object_id in objects_to_delete:

        del objects[object_id]


    # =====================================================
    # DRAW LOGICAL OBJECTS
    # =====================================================

    for object_id, obj in objects.items():

        center = obj["center"]

        box = obj["box"]

        missed = obj["missed"]

        x1, y1, x2, y2 = box


        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        cv2.rectangle(

            frame,

            (int(x1), int(y1)),

            (int(x2), int(y2)),

            (0, 255, 0),

            2
        )


        # -------------------------------------------------
        # Logical ID
        # -------------------------------------------------

        label = (

            f"Bottle {object_id}"

            f" | YOLO {obj['tracker_id']}"

        )


        if missed > 0:

            label += (

                f" | LOST {missed}"
            )


        cv2.putText(

            frame,

            label,

            (
                int(x1),

                max(int(y1) - 10, 20)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


        # -------------------------------------------------
        # Center
        # -------------------------------------------------

        cv2.circle(

            frame,

            center,

            5,

            (0, 0, 255),

            -1
        )


        # -------------------------------------------------
        # Draw history
        # -------------------------------------------------

        history = obj["history"]

        for i in range(1, len(history)):

            cv2.line(

                frame,

                history[i - 1],

                history[i],

                (255, 255, 255),

                2
            )


    # =====================================================
    # INFORMATION
    # =====================================================

    cv2.putText(

        frame,

        f"Logical objects: {len(objects)}",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(

        "SIH26174 - Persistent Object Tracking",

        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()