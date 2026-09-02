import cv2
import math
from collections import deque
from ultralytics import YOLO


# =========================================================
# MODELS
# =========================================================

object_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")


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

# Hand/object proximity
HAND_NEAR_THRESHOLD = 100
HAND_FAR_THRESHOLD = 150

# Movement thresholds
MIN_HAND_MOVEMENT = 10
MIN_OBJECT_MOVEMENT = 10

# How close the movement directions must be
MAX_RELATIVE_MOVEMENT = 50

# Confirmation
PICKUP_CONFIRM_FRAMES = 8
RELEASE_CONFIRM_FRAMES = 10

# Persistent tracking
MATCH_DISTANCE = 140
MAX_MISSED_FRAMES = 15

# History
HISTORY_SIZE = 12


# =========================================================
# PERSISTENT OBJECT STORAGE
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
# MOVEMENT VECTOR
# =========================================================

def movement_vector(history):

    if len(history) < 2:
        return (0, 0)

    old = history[0]
    new = history[-1]

    return (
        new[0] - old[0],
        new[1] - old[1]
    )


# =========================================================
# VECTOR MAGNITUDE
# =========================================================

def vector_magnitude(vector):

    return math.sqrt(
        vector[0] ** 2 +
        vector[1] ** 2
    )


# =========================================================
# VECTOR DIFFERENCE
# =========================================================

def vector_difference(v1, v2):

    return math.sqrt(
        (v1[0] - v2[0]) ** 2 +
        (v1[1] - v2[1]) ** 2
    )


# =========================================================
# CREATE LOGICAL OBJECT
# =========================================================

def create_object(center, box, tracker_id):

    global next_object_id

    object_id = next_object_id

    next_object_id += 1

    objects[object_id] = {

        # Persistent identity
        "center": center,
        "box": box,
        "tracker_id": tracker_id,

        # Tracking
        "missed": 0,

        # Movement history
        "history": deque(
            [center],
            maxlen=HISTORY_SIZE
        ),

        # Interaction state
        "state": "IDLE",

        # Counters
        "pickup_counter": 0,
        "release_counter": 0,

        # Associated hand
        "hand_name": None,

        # Hand history
        "hand_history": deque(
            maxlen=HISTORY_SIZE
        )
    }

    print(
        f"[Bottle {object_id}] CREATED"
    )

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
    # OBJECT TRACKING
    # =====================================================

    object_results = object_model.track(

        frame,

        persist=True,

        tracker="custom_bytetrack.yaml",

        conf=CONFIDENCE,

        classes=[39],

        verbose=False
    )

    object_result = object_results[0]


    # =====================================================
    # POSE
    # =====================================================

    pose_results = pose_model.track(

        frame,

        persist=True,

        tracker="custom_bytetrack.yaml",

        conf=CONFIDENCE,

        verbose=False
    )

    pose_result = pose_results[0]


    # =====================================================
    # CURRENT DETECTIONS
    # =====================================================

    detections = []


    if object_result.boxes.id is not None:

        boxes = object_result.boxes.xyxy.cpu().numpy()

        tracker_ids = (
            object_result
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
    # MARK OBJECTS AS TEMPORARILY MISSED
    # =====================================================

    for object_id in objects:

        objects[object_id]["missed"] += 1


    # =====================================================
    # MATCH DETECTIONS TO PERSISTENT OBJECTS
    # =====================================================

    matched_objects = set()


    for detection in detections:

        center = detection["center"]

        tracker_id = detection["tracker_id"]


        best_object = None

        best_distance = float("inf")


        # -------------------------------------------------
        # First: same YOLO tracker ID
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
        # Second: nearest persistent object
        # -------------------------------------------------

        if best_object is None:

            for object_id, obj in objects.items():

                if object_id in matched_objects:
                    continue


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
        # Create new object
        # -------------------------------------------------

        else:

            new_id = create_object(
                center,
                detection["box"],
                tracker_id
            )

            matched_objects.add(new_id)


    # =====================================================
    # GET WRISTS
    # =====================================================

    wrists = []


    if pose_result.keypoints is not None:

        keypoints = (
            pose_result
            .keypoints
            .xy
            .cpu()
            .numpy()
        )


        for person in keypoints:

            # COCO:
            # 9  = left wrist
            # 10 = right wrist

            left = person[9]

            right = person[10]


            if (
                left[0] > 0
                and
                left[1] > 0
            ):

                wrists.append(
                    (
                        "LEFT HAND",
                        (
                            int(left[0]),
                            int(left[1])
                        )
                    )
                )


            if (
                right[0] > 0
                and
                right[1] > 0
            ):

                wrists.append(
                    (
                        "RIGHT HAND",
                        (
                            int(right[0]),
                            int(right[1])
                        )
                    )
                )


    # =====================================================
    # PROCESS EACH PERSISTENT OBJECT
    # =====================================================

    for object_id, obj in list(objects.items()):

        center = obj["center"]

        state = obj["state"]


        # =================================================
        # FIND CLOSEST HAND
        # =================================================

        closest_hand = None

        closest_distance = float("inf")


        for hand_name, hand_position in wrists:

            d = distance(
                center,
                hand_position
            )


            if d < closest_distance:

                closest_distance = d

                closest_hand = (
                    hand_name,
                    hand_position
                )


        # =================================================
        # UPDATE HAND HISTORY
        # =================================================

        if closest_hand is not None:

            hand_name, hand_position = closest_hand

            obj["hand_name"] = hand_name

            obj["hand_history"].append(
                hand_position
            )


        # =================================================
        # OBJECT MOVEMENT
        # =================================================

        object_vector = movement_vector(
            obj["history"]
        )

        object_movement = vector_magnitude(
            object_vector
        )


        # =================================================
        # HAND MOVEMENT
        # =================================================

        hand_vector = (0, 0)

        hand_movement = 0

        relative_movement = 999


        if len(obj["hand_history"]) >= 2:

            hand_vector = movement_vector(
                obj["hand_history"]
            )

            hand_movement = vector_magnitude(
                hand_vector
            )


            relative_movement = vector_difference(
                object_vector,
                hand_vector
            )


        # =================================================
        # PICKUP LOGIC
        # =================================================

        if state == "IDLE":

            obj["pickup_counter"] = 0

            if (
                closest_hand is not None
                and
                closest_distance
                < HAND_NEAR_THRESHOLD
            ):

                obj["state"] = "NEAR_OBJECT"

                print(
                    f"[Bottle {object_id}] "
                    f"NEAR OBJECT"
                )


        # =================================================
        # NEAR OBJECT
        # =================================================

        elif state == "NEAR_OBJECT":

            # ---------------------------------------------
            # Hand moved away
            # ---------------------------------------------

            if (
                closest_hand is None
                or
                closest_distance
                > HAND_FAR_THRESHOLD
            ):

                obj["pickup_counter"] = 0

                obj["state"] = "IDLE"


            else:

                # -----------------------------------------
                # Check whether both are moving
                # -----------------------------------------

                hand_moving = (
                    hand_movement
                    > MIN_HAND_MOVEMENT
                )


                object_moving = (
                    object_movement
                    > MIN_OBJECT_MOVEMENT
                )


                moving_together = (
                    relative_movement
                    < MAX_RELATIVE_MOVEMENT
                )


                # -----------------------------------------
                # Strong pickup evidence
                # -----------------------------------------

                if (
                    hand_moving
                    and
                    object_moving
                    and
                    moving_together
                ):

                    obj["pickup_counter"] += 1

                else:

                    obj["pickup_counter"] = max(
                        0,
                        obj["pickup_counter"] - 1
                    )


                # -----------------------------------------
                # Confirm pickup
                # -----------------------------------------

                if (
                    obj["pickup_counter"]
                    >= PICKUP_CONFIRM_FRAMES
                ):

                    obj["state"] = "PICKED UP"

                    obj["pickup_counter"] = 0

                    print(
                        f"[Bottle {object_id}] "
                        f"PICKED UP"
                    )


        # =================================================
        # PICKED UP
        # =================================================

        elif state == "PICKED UP":

            obj["state"] = "CARRYING"

            print(
                f"[Bottle {object_id}] "
                f"CARRYING"
            )


        # =================================================
        # CARRYING
        # =================================================

        elif state == "CARRYING":

            # ---------------------------------------------
            # Bottle is considered released only after
            # sustained separation.
            # ---------------------------------------------

            if (
                closest_hand is None
                or
                closest_distance
                > HAND_FAR_THRESHOLD
            ):

                obj["release_counter"] += 1

            else:

                obj["release_counter"] = max(
                    0,
                    obj["release_counter"] - 1
                )


            if (
                obj["release_counter"]
                >= RELEASE_CONFIRM_FRAMES
            ):

                obj["state"] = "RELEASED"

                obj["release_counter"] = 0

                print(
                    f"[Bottle {object_id}] "
                    f"RELEASED"
                )


        # =================================================
        # RELEASED
        # =================================================

        elif state == "RELEASED":

            # Wait until bottle settles

            if (
                object_movement
                < MIN_OBJECT_MOVEMENT
            ):

                obj["state"] = "IDLE"


        # =================================================
        # DRAW OBJECT
        # =================================================

        x1, y1, x2, y2 = obj["box"]


        cv2.rectangle(

            frame,

            (int(x1), int(y1)),

            (int(x2), int(y2)),

            (0, 255, 0),

            2
        )


        # =================================================
        # DRAW LABEL
        # =================================================

        label = (
            f"Bottle {object_id}: "
            f"{obj['state']}"
        )


        cv2.putText(

            frame,

            label,

            (
                int(x1),
                max(int(y1) - 10, 20)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2
        )


        # =================================================
        # DEBUG VALUES
        # =================================================

        cv2.putText(

            frame,

            f"Hand: {closest_distance:.1f}px",

            (
                int(x1),
                int(y2) + 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Obj move: {object_movement:.1f}",

            (
                int(x1),
                int(y2) + 40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Hand move: {hand_movement:.1f}",

            (
                int(x1),
                int(y2) + 60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Relative: {relative_movement:.1f}",

            (
                int(x1),
                int(y2) + 80
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Pickup: {obj['pickup_counter']}/"
            f"{PICKUP_CONFIRM_FRAMES}",

            (
                int(x1),
                int(y2) + 100
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2
        )


        # =================================================
        # DRAW HAND CONNECTION
        # =================================================

        if closest_hand is not None:

            hand_name, hand_position = closest_hand


            cv2.circle(

                frame,

                hand_position,

                7,

                (0, 255, 0),

                -1
            )


            cv2.line(

                frame,

                hand_position,

                center,

                (255, 255, 255),

                2
            )


    # =====================================================
    # REMOVE OBJECTS LOST TOO LONG
    # =====================================================

    for object_id in list(objects.keys()):

        if (
            objects[object_id]["missed"]
            > MAX_MISSED_FRAMES
        ):

            print(
                f"[Bottle {object_id}] "
                f"REMOVED"
            )

            del objects[object_id]


    # =====================================================
    # GLOBAL INFO
    # =====================================================

    cv2.putText(

        frame,

        f"Persistent objects: {len(objects)}",

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

        "SIH26174 - Step 6D",

        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()