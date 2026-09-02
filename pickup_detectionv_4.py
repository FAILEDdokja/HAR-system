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
    print("ERROR: Could not open camera")
    exit()


# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE = 0.45

# Bottle filtering
MIN_BOTTLE_WIDTH = 15
MIN_BOTTLE_HEIGHT = 20
MAX_BOTTLE_WIDTH = 500
MAX_BOTTLE_HEIGHT = 700

MIN_ASPECT_RATIO = 0.20
MAX_ASPECT_RATIO = 2.50

# Tracking
NORMAL_MATCH_DISTANCE = 160
CARRY_MATCH_DISTANCE = 450

MAX_MISSED_FRAMES = 20

# Prediction
PREDICTION_WEIGHT = 1.0

# History
HISTORY_SIZE = 15

# Hand distance
HAND_NEAR_THRESHOLD = 100
HAND_FAR_THRESHOLD = 170

# Movement
MIN_HAND_MOVEMENT = 8
MIN_OBJECT_MOVEMENT = 8
MAX_RELATIVE_MOVEMENT = 60

# Pickup
PICKUP_CONFIRM_FRAMES = 8

# Release
RELEASE_CONFIRM_FRAMES = 12


# =========================================================
# SINGLE TARGET
# =========================================================

target = None


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def magnitude(vector):

    return math.sqrt(
        vector[0] ** 2 +
        vector[1] ** 2
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


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


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
# BOTTLE DETECTION FILTER
# =========================================================

def valid_bottle_detection(box, confidence):

    if confidence < CONFIDENCE:
        return False


    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1


    # ---------------------------------------------
    # Size filter
    # ---------------------------------------------

    if width < MIN_BOTTLE_WIDTH:
        return False

    if height < MIN_BOTTLE_HEIGHT:
        return False

    if width > MAX_BOTTLE_WIDTH:
        return False

    if height > MAX_BOTTLE_HEIGHT:
        return False


    # ---------------------------------------------
    # Aspect ratio filter
    # ---------------------------------------------

    if height == 0:
        return False

    ratio = width / height

    if ratio < MIN_ASPECT_RATIO:
        return False

    if ratio > MAX_ASPECT_RATIO:
        return False


    return True


# =========================================================
# PREDICT TARGET POSITION
# =========================================================

def predict_position():

    global target

    if target is None:
        return None

    history = target["history"]

    if len(history) < 2:

        return target["center"]


    velocity = movement_vector(history)

    current = target["center"]


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
# CREATE TARGET
# =========================================================

def create_target(
    center,
    box,
    tracker_id,
    confidence
):

    global target

    target = {

        "center": center,

        "box": box,

        "tracker_id": tracker_id,

        "confidence": confidence,

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

        "identity_locked": False
    }


    print(
        "[Bottle 1] CREATED"
    )


# =========================================================
# FIND BEST DETECTION
# =========================================================

def choose_detection(detections):

    global target


    if len(detections) == 0:
        return None


    # =====================================================
    # NO EXISTING TARGET
    # =====================================================

    if target is None:

        # Highest confidence valid detection

        detections.sort(
            key=lambda d: d["confidence"],
            reverse=True
        )

        return detections[0]


    # =====================================================
    # EXISTING TARGET
    # =====================================================

    predicted = predict_position()


    best = None

    best_score = float("inf")


    # =====================================================
    # MATCH CANDIDATES
    # =====================================================

    for detection in detections:

        center = detection["center"]


        prediction_distance = distance(
            predicted,
            center
        )


        current_distance = distance(
            target["center"],
            center
        )


        # -------------------------------------------------
        # State-dependent tracking radius
        # -------------------------------------------------

        if target["state"] in [
            "PICKED UP",
            "CARRYING"
        ]:

            max_distance = (
                CARRY_MATCH_DISTANCE
            )

        else:

            max_distance = (
                NORMAL_MATCH_DISTANCE
            )


        # -------------------------------------------------
        # If identity is locked, use a much more
        # forgiving radius.
        # -------------------------------------------------

        if target["identity_locked"]:

            max_distance = max(
                max_distance,
                CARRY_MATCH_DISTANCE
            )


        # -------------------------------------------------
        # Candidate must be reasonably close
        # -------------------------------------------------

        if (
            prediction_distance > max_distance
            and
            current_distance > max_distance
        ):

            continue


        # -------------------------------------------------
        # Confidence bonus
        # -------------------------------------------------

        confidence_penalty = (
            1.0 - detection["confidence"]
        ) * 100


        # -------------------------------------------------
        # Prefer predicted position
        # -------------------------------------------------

        score = (

            prediction_distance * 0.75

            +

            current_distance * 0.25

            +

            confidence_penalty
        )


        # -------------------------------------------------
        # Same YOLO ID gets strong preference
        # -------------------------------------------------

        if (
            detection["tracker_id"]
            ==
            target["tracker_id"]
        ):

            score -= 100


        if score < best_score:

            best_score = score

            best = detection


    return best


# =========================================================
# GET WRISTS
# =========================================================

def get_wrists(pose_result):

    wrists = []


    if pose_result.keypoints is None:
        return wrists


    keypoints = (
        pose_result
        .keypoints
        .xy
        .cpu()
        .numpy()
    )


    for person in keypoints:

        # COCO pose:
        # 9 = left wrist
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


    return wrists


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "ERROR: Camera frame failed"
        )

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
    # COLLECT VALID DETECTIONS
    # =====================================================

    detections = []


    if object_result.boxes is not None:

        boxes = (
            object_result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )


        confidences = (
            object_result
            .boxes
            .conf
            .cpu()
            .numpy()
        )


        if object_result.boxes.id is not None:

            tracker_ids = (
                object_result
                .boxes
                .id
                .cpu()
                .numpy()
                .astype(int)
            )

        else:

            tracker_ids = [
                -1
                for _ in boxes
            ]


        for box, confidence, tracker_id in zip(
            boxes,
            confidences,
            tracker_ids
        ):


            # -------------------------------------------------
            # FILTER FALSE DETECTIONS
            # -------------------------------------------------

            if not valid_bottle_detection(
                box,
                confidence
            ):

                continue


            center = box_center(box)


            detections.append({

                "box": box,

                "center": center,

                "tracker_id": int(
                    tracker_id
                ),

                "confidence": float(
                    confidence
                )
            })


    # =====================================================
    # SELECT ONLY ONE TARGET
    # =====================================================

    selected = choose_detection(
        detections
    )


    # =====================================================
    # TARGET UPDATE
    # =====================================================

    if selected is not None:

        if target is None:

            create_target(

                selected["center"],

                selected["box"],

                selected["tracker_id"],

                selected["confidence"]
            )


        else:

            target["center"] = (
                selected["center"]
            )

            target["box"] = (
                selected["box"]
            )

            target["tracker_id"] = (
                selected["tracker_id"]
            )

            target["confidence"] = (
                selected["confidence"]
            )

            target["missed"] = 0

            target["history"].append(
                selected["center"]
            )


    # =====================================================
    # NO DETECTION
    # =====================================================

    else:

        if target is not None:

            target["missed"] += 1


    # =====================================================
    # WRISTS
    # =====================================================

    wrists = get_wrists(
        pose_result
    )


    # =====================================================
    # PROCESS TARGET
    # =====================================================

    if target is not None:

        center = target["center"]

        box = target["box"]

        state = target["state"]


        # =================================================
        # FIND CLOSEST HAND
        # =================================================

        closest_hand = None

        closest_hand_distance = float(
            "inf"
        )


        for hand_name, hand_position in wrists:

            d = point_to_box_distance(
                hand_position,
                box
            )


            if (
                d
                <
                closest_hand_distance
            ):

                closest_hand_distance = d

                closest_hand = (
                    hand_name,
                    hand_position
                )


        # =================================================
        # HAND HISTORY
        # =================================================

        if closest_hand is not None:

            hand_name, hand_position = (
                closest_hand
            )


            target["hand_name"] = (
                hand_name
            )


            target["hand_history"].append(
                hand_position
            )


        # =================================================
        # MOVEMENT
        # =================================================

        object_vector = movement_vector(
            target["history"]
        )


        object_movement = magnitude(
            object_vector
        )


        hand_vector = (0, 0)

        hand_movement = 0

        relative_movement = 999


        if len(
            target["hand_history"]
        ) >= 2:


            hand_vector = movement_vector(
                target["hand_history"]
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

            target["pickup_counter"] = 0


            if (
                closest_hand is not None
                and
                closest_hand_distance
                <
                HAND_NEAR_THRESHOLD
            ):

                target["state"] = (
                    "NEAR_OBJECT"
                )


                print(
                    "[Bottle 1] "
                    "NEAR OBJECT"
                )


        # =================================================
        # NEAR OBJECT
        # =================================================

        elif state == "NEAR_OBJECT":

            if (
                closest_hand is None
                or
                closest_hand_distance
                >
                HAND_FAR_THRESHOLD
            ):

                target["pickup_counter"] = (
                    max(
                        0,
                        target["pickup_counter"] - 1
                    )
                )


                if (
                    target["pickup_counter"]
                    == 0
                ):

                    target["state"] = (
                        "IDLE"
                    )


            else:

                hand_moving = (
                    hand_movement
                    >
                    MIN_HAND_MOVEMENT
                )


                object_moving = (
                    object_movement
                    >
                    MIN_OBJECT_MOVEMENT
                )


                moving_together = (
                    relative_movement
                    <
                    MAX_RELATIVE_MOVEMENT
                )


                if (
                    hand_moving
                    and
                    object_moving
                    and
                    moving_together
                ):

                    target[
                        "pickup_counter"
                    ] += 1

                else:

                    target[
                        "pickup_counter"
                    ] = max(
                        0,
                        target[
                            "pickup_counter"
                        ] - 1
                    )


                # -----------------------------------------
                # PICKUP CONFIRMED
                # -----------------------------------------

                if (
                    target[
                        "pickup_counter"
                    ]
                    >=
                    PICKUP_CONFIRM_FRAMES
                ):

                    target["state"] = (
                        "PICKED UP"
                    )


                    target[
                        "identity_locked"
                    ] = True


                    target[
                        "pickup_counter"
                    ] = 0


                    print(
                        "[Bottle 1] "
                        "PICKED UP"
                    )


        # =================================================
        # PICKED UP
        # =================================================

        elif state == "PICKED UP":

            target["state"] = (
                "CARRYING"
            )


            target[
                "identity_locked"
            ] = True


            print(
                "[Bottle 1] "
                "CARRYING"
            )


        # =================================================
        # CARRYING
        # =================================================

        elif state == "CARRYING":

            # ---------------------------------------------
            # IMPORTANT:
            #
            # Don't release just because one frame loses
            # the hand.
            # ---------------------------------------------

            if (
                closest_hand is not None
                and
                closest_hand_distance
                <=
                HAND_FAR_THRESHOLD
            ):

                target[
                    "release_counter"
                ] = max(
                    0,
                    target[
                        "release_counter"
                    ] - 1
                )


            else:

                target[
                    "release_counter"
                ] += 1


            # ---------------------------------------------
            # RELEASE CONFIRMED
            # ---------------------------------------------

            if (
                target[
                    "release_counter"
                ]
                >=
                RELEASE_CONFIRM_FRAMES
            ):

                target["state"] = (
                    "RELEASED"
                )


                target[
                    "release_counter"
                ] = 0


                print(
                    "[Bottle 1] "
                    "RELEASED"
                )


        # =================================================
        # RELEASED
        # =================================================

        elif state == "RELEASED":

            # Wait for bottle to become relatively still

            if (
                object_movement
                <
                MIN_OBJECT_MOVEMENT
            ):

                target["state"] = (
                    "IDLE"
                )


                target[
                    "identity_locked"
                ] = False


                print(
                    "[Bottle 1] "
                    "IDLE"
                )


        # =================================================
        # DRAW BOTTLE
        # =================================================

        x1, y1, x2, y2 = box


        cv2.rectangle(

            frame,

            (
                int(x1),
                int(y1)
            ),

            (
                int(x2),
                int(y2)
            ),

            (0, 255, 0),

            2
        )


        # =================================================
        # STATE LABEL
        # =================================================

        cv2.putText(

            frame,

            (
                f"Bottle 1: "
                f"{target['state']}"
            ),

            (
                int(x1),
                max(
                    int(y1) - 10,
                    20
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2
        )


        # =================================================
        # YOLO ID
        # =================================================

        cv2.putText(

            frame,

            (
                f"YOLO ID: "
                f"{target['tracker_id']}"
            ),

            (
                int(x1),
                int(y2) + 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        # =================================================
        # HAND DISTANCE
        # =================================================

        hand_text = (
            "Hand: --"
            if closest_hand is None
            else
            f"Hand: "
            f"{closest_hand_distance:.1f}px"
        )


        cv2.putText(

            frame,

            hand_text,

            (
                int(x1),
                int(y2) + 40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        # =================================================
        # MOVEMENT
        # =================================================

        cv2.putText(

            frame,

            (
                f"Object: "
                f"{object_movement:.1f}px"
            ),

            (
                int(x1),
                int(y2) + 60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            (
                f"Hand move: "
                f"{hand_movement:.1f}px"
            ),

            (
                int(x1),
                int(y2) + 80
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            (
                f"Relative: "
                f"{relative_movement:.1f}px"
            ),

            (
                int(x1),
                int(y2) + 100
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            (
                f"Pickup: "
                f"{target['pickup_counter']}/"
                f"{PICKUP_CONFIRM_FRAMES}"
            ),

            (
                int(x1),
                int(y2) + 120
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            2
        )


        # =================================================
        # PREDICTION POINT
        # =================================================

        predicted = predict_position()


        if predicted is not None:

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

            hand_name, hand_position = (
                closest_hand
            )


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
    # TARGET LOST
    # =====================================================

    if target is not None:

        if (
            target["missed"]
            > MAX_MISSED_FRAMES
        ):

            # -------------------------------------------------
            # If bottle was being carried, DON'T create a
            # new bottle. Keep the logical target alive.
            # -------------------------------------------------

            if target["identity_locked"]:

                target["missed"] = (
                    MAX_MISSED_FRAMES
                )

            else:

                print(
                    "[Bottle 1] "
                    "TARGET LOST"
                )

                target = None


    # =====================================================
    # GLOBAL INFORMATION
    # =====================================================

    object_count = (
        0
        if target is None
        else 1
    )


    cv2.putText(

        frame,

        f"Tracked bottles: {object_count}",

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

        "SIH26174 - Step 6F",

        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if (
        cv2.waitKey(1) & 0xFF
        ==
        ord("q")
    ):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()