import cv2
import math
from collections import defaultdict, deque
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

HAND_NEAR_THRESHOLD = 100

HAND_FAR_THRESHOLD = 150

MIN_HAND_MOVEMENT = 12

MIN_OBJECT_MOVEMENT = 12

MAX_RELATIVE_MOVEMENT = 45

PICKUP_CONFIRM_FRAMES = 8

RELEASE_CONFIRM_FRAMES = 8

HISTORY_SIZE = 12


# =========================================================
# HISTORY
# =========================================================

object_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)

hand_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)


# =========================================================
# STATE
# =========================================================

states = defaultdict(lambda: "IDLE")

pickup_counter = defaultdict(int)

release_counter = defaultdict(int)


# =========================================================
# FUNCTIONS
# =========================================================

def distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
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


def vector_distance(v1, v2):

    return math.sqrt(
        (v1[0] - v2[0]) ** 2 +
        (v1[1] - v2[1]) ** 2
    )


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
        conf=0.35,
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
        conf=0.35,
        verbose=False
    )

    pose_result = pose_results[0]


    output = object_result.plot()


    # =====================================================
    # GET BOTTLES
    # =====================================================

    bottles = []

    if object_result.boxes.id is not None:

        boxes = object_result.boxes.xyxy.cpu().numpy()

        ids = (
            object_result
            .boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, object_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            bottles.append({
                "id": object_id,
                "box": box,
                "center": center
            })

            object_history[object_id].append(center)


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

            if left[0] > 0 and left[1] > 0:

                wrists.append(
                    ("LEFT HAND", (int(left[0]), int(left[1])))
                )

            if right[0] > 0 and right[1] > 0:

                wrists.append(
                    ("RIGHT HAND", (int(right[0]), int(right[1])))
                )


    # =====================================================
    # PROCESS EACH BOTTLE
    # =====================================================

    for bottle in bottles:

        object_id = bottle["id"]

        object_center = bottle["center"]

        box = bottle["box"]


        # -------------------------------------------------
        # Find closest hand
        # -------------------------------------------------

        closest_hand = None

        closest_distance = float("inf")

        for hand_name, hand_position in wrists:

            d = distance(
                object_center,
                hand_position
            )

            if d < closest_distance:

                closest_distance = d

                closest_hand = (
                    hand_name,
                    hand_position
                )


        # -------------------------------------------------
        # Need a hand
        # -------------------------------------------------

        if closest_hand is not None:

            hand_name, hand_position = closest_hand

            hand_key = (
                object_id,
                hand_name
            )

            hand_history[hand_key].append(
                hand_position
            )

        else:

            hand_key = None


        # -------------------------------------------------
        # Calculate movement
        # -------------------------------------------------

        object_vector = movement_vector(
            object_history[object_id]
        )

        object_movement = math.sqrt(
            object_vector[0] ** 2 +
            object_vector[1] ** 2
        )


        hand_vector = (0, 0)

        hand_movement = 0

        relative_movement = 999


        if hand_key is not None:

            history = hand_history[hand_key]

            hand_vector = movement_vector(history)

            hand_movement = math.sqrt(
                hand_vector[0] ** 2 +
                hand_vector[1] ** 2
            )


            # ---------------------------------------------
            # Compare hand and object movement
            # ---------------------------------------------

            if len(history) >= 5:

                relative_movement = vector_distance(
                    object_vector,
                    hand_vector
                )


        # =================================================
        # STATE
        # =================================================

        state = states[object_id]


        # =================================================
        # IDLE
        # =================================================

        if state == "IDLE":

            pickup_counter[object_id] = 0

            if (
                closest_hand is not None
                and
                closest_distance < HAND_NEAR_THRESHOLD
            ):

                state = "NEAR_OBJECT"

                print(
                    f"[Bottle {object_id}] "
                    f"NEAR OBJECT"
                )


        # =================================================
        # NEAR OBJECT
        # =================================================

        elif state == "NEAR_OBJECT":

            # Hand moved away
            if (
                closest_hand is None
                or
                closest_distance > HAND_FAR_THRESHOLD
            ):

                state = "IDLE"

                pickup_counter[object_id] = 0

            else:

                # -----------------------------------------
                # The important condition
                # -----------------------------------------

                hand_moving = (
                    hand_movement > MIN_HAND_MOVEMENT
                )

                object_moving = (
                    object_movement > MIN_OBJECT_MOVEMENT
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

                    pickup_counter[object_id] += 1

                else:

                    pickup_counter[object_id] = max(
                        0,
                        pickup_counter[object_id] - 1
                    )


                # -----------------------------------------
                # Confirm pickup
                # -----------------------------------------

                if (
                    pickup_counter[object_id]
                    >= PICKUP_CONFIRM_FRAMES
                ):

                    state = "PICKED_UP"

                    pickup_counter[object_id] = 0

                    print(
                        f"[Bottle {object_id}] "
                        f"PICKED UP"
                    )


        # =================================================
        # PICKED UP
        # =================================================

        elif state == "PICKED_UP":

            state = "CARRYING"

            print(
                f"[Bottle {object_id}] "
                f"CARRYING"
            )


        # =================================================
        # CARRYING
        # =================================================

        elif state == "CARRYING":

            # ------------------------------------------------
            # Release should NOT happen merely because
            # the hand moves away for one frame.
            # ------------------------------------------------

            if (
                closest_hand is None
                or
                closest_distance > HAND_FAR_THRESHOLD
            ):

                release_counter[object_id] += 1

            else:

                release_counter[object_id] = max(
                    0,
                    release_counter[object_id] - 1
                )


            if (
                release_counter[object_id]
                >= RELEASE_CONFIRM_FRAMES
            ):

                state = "RELEASED"

                release_counter[object_id] = 0

                print(
                    f"[Bottle {object_id}] "
                    f"RELEASED"
                )


        # =================================================
        # RELEASED
        # =================================================

        elif state == "RELEASED":

            # Wait for the object to settle
            if object_movement < MIN_OBJECT_MOVEMENT:

                state = "IDLE"


        # Save state
        states[object_id] = state


        # =================================================
        # VISUALIZATION
        # =================================================

        x1, y1, x2, y2 = box

        text_x = int(x1)

        text_y = max(
            int(y1) - 15,
            20
        )


        cv2.putText(
            output,
            f"Bottle {object_id}: {state}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            f"Hand: {closest_distance:.1f}px",
            (text_x, text_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            f"Object move: {object_movement:.1f}px",
            (text_x, text_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            f"Hand move: {hand_movement:.1f}px",
            (text_x, text_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            f"Relative: {relative_movement:.1f}px",
            (text_x, text_y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            f"Confirm: {pickup_counter[object_id]}",
            (text_x, text_y + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        # -------------------------------------------------
        # Draw hand ↔ bottle
        # -------------------------------------------------

        if closest_hand is not None:

            hand_name, hand_position = closest_hand

            cv2.circle(
                output,
                hand_position,
                7,
                (0, 255, 0),
                -1
            )

            cv2.line(
                output,
                hand_position,
                object_center,
                (255, 255, 255),
                2
            )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "SIH26174 - Step 6B",
        output
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()