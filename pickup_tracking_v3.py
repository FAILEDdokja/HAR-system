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

# Hand distance
HAND_NEAR_THRESHOLD = 110
HAND_FAR_THRESHOLD = 170

# Movement
MIN_HAND_MOVEMENT = 8
MIN_OBJECT_MOVEMENT = 8

# Pickup
MAX_RELATIVE_MOVEMENT = 55
PICKUP_CONFIRM_FRAMES = 8

# Release
RELEASE_CONFIRM_FRAMES = 10

# Tracking
NORMAL_MATCH_DISTANCE = 140
CARRY_MATCH_DISTANCE = 350
MAX_MISSED_FRAMES = 20

# Prediction
PREDICTION_WEIGHT = 1.0

# History
HISTORY_SIZE = 15


# =========================================================
# PERSISTENT OBJECTS
# =========================================================

objects = {}

next_object_id = 1


# =========================================================
# BASIC FUNCTIONS
# =========================================================

def distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def magnitude(v):

    return math.sqrt(
        v[0] ** 2 +
        v[1] ** 2
    )


def vector_difference(v1, v2):

    return math.sqrt(
        (v1[0] - v2[0]) ** 2 +
        (v1[1] - v2[1]) ** 2
    )


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
# DISTANCE FROM POINT TO BOUNDING BOX
# =========================================================

def point_to_box_distance(point, box):

    px, py = point

    x1, y1, x2, y2 = box

    closest_x = max(
        x1,
        min(px, x2)
    )

    closest_y = max(
        y1,
        min(py, y2)
    )

    return math.sqrt(
        (px - closest_x) ** 2 +
        (py - closest_y) ** 2
    )


# =========================================================
# PREDICT NEXT POSITION
# =========================================================

def predict_position(obj):

    history = obj["history"]

    if len(history) < 2:

        return obj["center"]

    velocity = movement_vector(history)

    current = obj["center"]

    predicted = (

        int(
            current[0]
            + velocity[0]
            * PREDICTION_WEIGHT
        ),

        int(
            current[1]
            + velocity[1]
            * PREDICTION_WEIGHT
        )
    )

    return predicted


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
        ),

        "state": "IDLE",

        "pickup_counter": 0,

        "release_counter": 0,

        "hand_history": deque(
            maxlen=HISTORY_SIZE
        ),

        "hand_name": None,

        # Once pickup is confirmed,
        # identity becomes locked.
        "identity_locked": False
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
    # OBJECT DETECTION + TRACKING
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
    # GET CURRENT BOTTLE DETECTIONS
    # =====================================================

    detections = []


    if object_result.boxes.id is not None:

        boxes = (
            object_result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

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
    # MARK ALL OBJECTS AS MISSED
    # =====================================================

    for object_id in objects:

        objects[object_id]["missed"] += 1


    matched_objects = set()


    # =====================================================
    # MATCH DETECTIONS
    # =====================================================

    for detection in detections:

        center = detection["center"]

        tracker_id = detection["tracker_id"]


        best_object = None

        best_score = float("inf")


        # -------------------------------------------------
        # SEARCH EXISTING OBJECTS
        # -------------------------------------------------

        for object_id, obj in objects.items():

            if object_id in matched_objects:
                continue


            if (
                obj["missed"]
                > MAX_MISSED_FRAMES
            ):
                continue


            # -------------------------------------------------
            # PREDICT WHERE OBJECT SHOULD BE
            # -------------------------------------------------

            predicted = predict_position(obj)


            prediction_distance = distance(
                predicted,
                center
            )


            current_distance = distance(
                obj["center"],
                center
            )


            # -------------------------------------------------
            # MATCH RADIUS DEPENDS ON STATE
            # -------------------------------------------------

            if obj["state"] in [
                "PICKED UP",
                "CARRYING"
            ]:

                match_distance = (
                    CARRY_MATCH_DISTANCE
                )

            else:

                match_distance = (
                    NORMAL_MATCH_DISTANCE
                )


            # -------------------------------------------------
            # Same YOLO ID gets priority
            # -------------------------------------------------

            same_tracker = (
                obj["tracker_id"]
                == tracker_id
            )


            # -------------------------------------------------
            # Calculate matching score
            # -------------------------------------------------

            if same_tracker:

                score = prediction_distance * 0.5

            else:

                score = prediction_distance


            # -------------------------------------------------
            # Accept match
            # -------------------------------------------------

            if (
                prediction_distance
                <= match_distance
            ):

                if score < best_score:

                    best_score = score

                    best_object = object_id


            # -------------------------------------------------
            # Fallback to current position
            # -------------------------------------------------

            elif (
                current_distance
                <= match_distance
            ):

                if current_distance < best_score:

                    best_score = current_distance

                    best_object = object_id


        # =====================================================
        # UPDATE EXISTING OBJECT
        # =====================================================

        if best_object is not None:

            obj = objects[best_object]

            obj["center"] = center

            obj["box"] = detection["box"]

            obj["tracker_id"] = tracker_id

            obj["missed"] = 0

            obj["history"].append(center)

            matched_objects.add(best_object)


        # =====================================================
        # CREATE NEW OBJECT
        # =====================================================

        else:

            create_object(

                center,

                detection["box"],

                tracker_id
            )


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
    # PROCESS OBJECTS
    # =====================================================

    for object_id, obj in list(objects.items()):

        center = obj["center"]

        box = obj["box"]

        state = obj["state"]


        # =================================================
        # FIND CLOSEST HAND
        # =================================================

        closest_hand = None

        closest_hand_distance = float("inf")


        for hand_name, hand_position in wrists:

            d = point_to_box_distance(
                hand_position,
                box
            )


            if d < closest_hand_distance:

                closest_hand_distance = d

                closest_hand = (
                    hand_name,
                    hand_position
                )


        # =================================================
        # HAND HISTORY
        # =================================================

        if closest_hand is not None:

            hand_name, hand_position = closest_hand

            obj["hand_name"] = hand_name

            obj["hand_history"].append(
                hand_position
            )


        # =================================================
        # MOVEMENT
        # =================================================

        object_vector = movement_vector(
            obj["history"]
        )

        object_movement = magnitude(
            object_vector
        )


        hand_vector = (0, 0)

        hand_movement = 0

        relative_movement = 999


        if len(obj["hand_history"]) >= 2:

            hand_vector = movement_vector(
                obj["hand_history"]
            )

            hand_movement = magnitude(
                hand_vector
            )

            relative_movement = (
                vector_difference(
                    object_vector,
                    hand_vector
                )
            )


        # =================================================
        # IDLE
        # =================================================

        if state == "IDLE":

            obj["pickup_counter"] = 0

            if (
                closest_hand is not None
                and
                closest_hand_distance
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

            if (
                closest_hand is None
                or
                closest_hand_distance
                > HAND_FAR_THRESHOLD
            ):

                obj["pickup_counter"] = 0

                obj["state"] = "IDLE"


            else:

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


                if (
                    obj["pickup_counter"]
                    >= PICKUP_CONFIRM_FRAMES
                ):

                    obj["state"] = "PICKED UP"

                    obj["pickup_counter"] = 0

                    # Lock logical identity
                    obj["identity_locked"] = True

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

            # -------------------------------------------------
            # If the hand is still near the bottle,
            # reset release counter.
            # -------------------------------------------------

            if (
                closest_hand is not None
                and
                closest_hand_distance
                <= HAND_FAR_THRESHOLD
            ):

                obj["release_counter"] = max(
                    0,
                    obj["release_counter"] - 1
                )


            else:

                obj["release_counter"] += 1


            # -------------------------------------------------
            # Confirm release
            # -------------------------------------------------

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

            if (
                object_movement
                < MIN_OBJECT_MOVEMENT
            ):

                obj["state"] = "IDLE"

                obj["identity_locked"] = False

                print(
                    f"[Bottle {object_id}] "
                    f"IDLE"
                )


        # =================================================
        # DRAW BOX
        # =================================================

        x1, y1, x2, y2 = box


        cv2.rectangle(

            frame,

            (int(x1), int(y1)),

            (int(x2), int(y2)),

            (0, 255, 0),

            2
        )


        # =================================================
        # DRAW STATE
        # =================================================

        cv2.putText(

            frame,

            f"Bottle {object_id}: {obj['state']}",

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

        debug_y = int(y2) + 20


        cv2.putText(

            frame,

            f"Hand: "
            f"{closest_hand_distance:.1f}px",

            (
                int(x1),
                debug_y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Obj: "
            f"{object_movement:.1f}px",

            (
                int(x1),
                debug_y + 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Hand move: "
            f"{hand_movement:.1f}px",

            (
                int(x1),
                debug_y + 40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Relative: "
            f"{relative_movement:.1f}px",

            (
                int(x1),
                debug_y + 60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Pickup: "
            f"{obj['pickup_counter']}/"
            f"{PICKUP_CONFIRM_FRAMES}",

            (
                int(x1),
                debug_y + 80
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        # =================================================
        # PREDICTION POINT
        # =================================================

        predicted = predict_position(obj)


        cv2.circle(

            frame,

            predicted,

            6,

            (255, 0, 255),

            -1
        )


        # =================================================
        # HAND
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

            # Don't immediately delete an object
            # that has been picked up.
            if objects[object_id]["state"] not in [
                "PICKED UP",
                "CARRYING"
            ]:

                print(
                    f"[Bottle {object_id}] "
                    f"REMOVED"
                )

                del objects[object_id]


    # =====================================================
    # GLOBAL INFORMATION
    # =====================================================

    cv2.putText(

        frame,

        f"Objects: {len(objects)}",

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

        "SIH26174 - Step 6E",

        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()