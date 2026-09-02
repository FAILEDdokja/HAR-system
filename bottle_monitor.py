"""Single-target bottle interaction monitor.

The detector may return many bottle-class boxes and may change ByteTrack IDs.
This module deliberately maintains one application object only: ``Bottle 1``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


def center(box: BBox) -> Point:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_size(box: BBox) -> tuple[float, float]:
    return (max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def box_area(box: BBox) -> float:
    width, height = box_size(box)
    return width * height


def box_iou(first: BBox, second: BBox) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union else 0.0


def point_to_box_distance(point: Point, box: BBox) -> float:
    """Euclidean distance from a wrist to the nearest point on a box."""
    x, y = point
    nearest_x = min(max(x, box[0]), box[2])
    nearest_y = min(max(y, box[1]), box[3])
    return math.hypot(x - nearest_x, y - nearest_y)


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _add_box(box: BBox, velocity: tuple[float, float, float, float], steps: int) -> BBox:
    return tuple(value + delta * steps for value, delta in zip(box, velocity))  # type: ignore[return-value]


@dataclass(frozen=True)
class Detection:
    box: BBox
    confidence: float
    yolo_track_id: Optional[int] = None


@dataclass
class TrackerConfig:
    min_confidence: float = 0.45
    min_area_fraction: float = 0.0008
    max_area_fraction: float = 0.18
    min_aspect_ratio: float = 0.18
    max_aspect_ratio: float = 2.2
    acquire_frames: int = 4
    acquire_distance_fraction: float = 0.08
    acquire_iou: float = 0.08
    association_distance_fraction: float = 0.16
    carrying_distance_fraction: float = 0.34
    reacquire_growth_per_miss: float = 0.035
    max_reacquire_distance_fraction: float = 0.45
    association_iou: float = 0.02
    max_prediction_misses: int = 14
    roi: Optional[BBox] = None


@dataclass(frozen=True)
class TrackResult:
    logical_id: Optional[int]
    box: Optional[BBox]
    measured: bool
    lost_frames: int
    acquiring_hits: int = 0
    candidate_count: int = 0


class SingleBottleTracker:
    """Associates detections to one logical target without ever allocating ID 2."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.logical_id: Optional[int] = None
        self.box: Optional[BBox] = None
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self.last_yolo_track_id: Optional[int] = None
        self.lost_frames = 0
        self.locked = False
        self._tentative: Optional[Detection] = None
        self._tentative_hits = 0

    def reset_for_new_scene(self) -> None:
        """Forget the current target; a subsequently acquired target is still Bottle 1."""
        self.logical_id = None
        self.box = None
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self.last_yolo_track_id = None
        self.lost_frames = 0
        self.locked = False
        self._tentative = None
        self._tentative_hits = 0

    def set_locked(self, locked: bool) -> None:
        self.locked = locked

    def valid_detections(
        self, detections: Iterable[Detection], frame_size: tuple[int, int]
    ) -> list[Detection]:
        """Reject implausible boxes before they can influence application state."""
        frame_width, frame_height = frame_size
        frame_area = frame_width * frame_height
        valid: list[Detection] = []
        for detection in detections:
            width, height = box_size(detection.box)
            area_fraction = box_area(detection.box) / frame_area if frame_area else 0.0
            aspect = width / height if height else 0.0
            if detection.confidence < self.config.min_confidence:
                continue
            if area_fraction < self.config.min_area_fraction:
                continue
            if area_fraction > self.config.max_area_fraction:
                continue
            if not (self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio):
                continue
            if self.config.roi and box_iou(detection.box, self.config.roi) <= 0.0:
                continue
            valid.append(detection)
        return valid

    def update(
        self, detections: Iterable[Detection], frame_size: tuple[int, int]
    ) -> TrackResult:
        valid = self.valid_detections(detections, frame_size)
        if self.box is None:
            return self._acquire(valid, frame_size)

        match = self._best_match(valid, frame_size)
        if match is None:
            self.lost_frames += 1
            if self.lost_frames <= self.config.max_prediction_misses:
                self.box = _add_box(self.box, self.velocity, 1)
            return TrackResult(1, self.box, False, self.lost_frames, candidate_count=len(valid))

        old_box = self.box
        self.box = match.box
        self.velocity = tuple(new - old for new, old in zip(match.box, old_box))  # type: ignore[assignment]
        self.logical_id = 1
        self.last_yolo_track_id = match.yolo_track_id
        self.lost_frames = 0
        self._tentative = None
        self._tentative_hits = 0
        return TrackResult(1, self.box, True, 0, candidate_count=len(valid))

    def _acquire(
        self, valid: list[Detection], frame_size: tuple[int, int]
    ) -> TrackResult:
        if not valid:
            self._tentative = None
            self._tentative_hits = 0
            return TrackResult(None, None, False, 0, candidate_count=0)

        candidate = max(valid, key=lambda item: (item.confidence, box_area(item.box)))
        if self._tentative is None:
            self._tentative = candidate
            self._tentative_hits = 1
        elif self._same_candidate(candidate, self._tentative, frame_size):
            self._tentative = candidate
            self._tentative_hits += 1
        else:
            self._tentative = candidate
            self._tentative_hits = 1

        if self._tentative_hits < self.config.acquire_frames:
            return TrackResult(
                None,
                candidate.box,
                False,
                0,
                acquiring_hits=self._tentative_hits,
                candidate_count=len(valid),
            )

        self.logical_id = 1
        self.box = candidate.box
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self.last_yolo_track_id = candidate.yolo_track_id
        return TrackResult(1, self.box, True, 0, self._tentative_hits, len(valid))

    def _same_candidate(
        self, current: Detection, previous: Detection, frame_size: tuple[int, int]
    ) -> bool:
        diagonal = math.hypot(*frame_size)
        return (
            _distance(center(current.box), center(previous.box))
            <= diagonal * self.config.acquire_distance_fraction
            or box_iou(current.box, previous.box) >= self.config.acquire_iou
        )

    def _best_match(
        self, valid: list[Detection], frame_size: tuple[int, int]
    ) -> Optional[Detection]:
        if self.box is None:
            return None

        diagonal = math.hypot(*frame_size)
        predicted_box = _add_box(self.box, self.velocity, max(1, self.lost_frames + 1))
        base_fraction = (
            self.config.carrying_distance_fraction
            if self.locked
            else self.config.association_distance_fraction
        )
        distance_fraction = min(
            self.config.max_reacquire_distance_fraction,
            base_fraction + self.lost_frames * self.config.reacquire_growth_per_miss,
        )
        max_distance = diagonal * distance_fraction

        best_detection: Optional[Detection] = None
        best_score = float("inf")
        for detection in valid:
            predicted_distance = _distance(center(detection.box), center(predicted_box))
            current_distance = _distance(center(detection.box), center(self.box))
            iou = max(box_iou(detection.box, predicted_box), box_iou(detection.box, self.box))
            same_yolo_id = (
                self.last_yolo_track_id is not None
                and detection.yolo_track_id == self.last_yolo_track_id
            )
            if predicted_distance > max_distance and current_distance > max_distance:
                if iou < self.config.association_iou:
                    continue

            score = min(predicted_distance, current_distance) - iou * 80.0
            if same_yolo_id:
                score -= 40.0
            score -= detection.confidence * 10.0
            if score < best_score:
                best_score = score
                best_detection = detection
        return best_detection


class InteractionState(str, Enum):
    IDLE = "IDLE"
    NEAR_OBJECT = "NEAR_OBJECT"
    PICKED_UP = "PICKED UP"
    CARRYING = "CARRYING"
    RELEASED = "RELEASED"


@dataclass
class InteractionConfig:
    near_distance_fraction: float = 0.055
    far_distance_fraction: float = 0.09
    movement_fraction: float = 0.012
    relative_movement_fraction: float = 0.08
    near_frames: int = 2
    pickup_frames: int = 6
    picked_up_frames: int = 2
    release_frames: int = 8
    stable_frames: int = 8


@dataclass(frozen=True)
class InteractionResult:
    state: InteractionState
    closest_hand: Optional[Point]
    hand_distance: Optional[float]
    object_movement: float
    hand_movement: float
    relative_movement: float
    pickup_counter: int
    release_counter: int


class InteractionMachine:
    """State machine that only consumes the selected logical target."""

    def __init__(self, config: InteractionConfig | None = None) -> None:
        self.config = config or InteractionConfig()
        self.state = InteractionState.IDLE
        self.pickup_counter = 0
        self.release_counter = 0
        self.near_counter = 0
        self.stable_counter = 0
        self.picked_up_counter = 0
        self._previous_box: Optional[BBox] = None
        self._previous_hand: Optional[Point] = None

    def update(
        self,
        measured: bool,
        box: Optional[BBox],
        wrists: Iterable[Point],
        frame_size: tuple[int, int],
    ) -> InteractionResult:
        if box is None:
            return self._result(None, None, 0.0, 0.0, 999.0)

        diagonal = math.hypot(*frame_size)
        near_threshold = diagonal * self.config.near_distance_fraction
        far_threshold = diagonal * self.config.far_distance_fraction
        movement_threshold = diagonal * self.config.movement_fraction
        relative_threshold = diagonal * self.config.relative_movement_fraction

        closest_hand, hand_distance = self._closest_hand(box, wrists)
        object_movement = (
            _distance(center(self._previous_box), center(box))
            if self._previous_box and measured
            else 0.0
        )
        hand_movement = (
            _distance(self._previous_hand, closest_hand)
            if self._previous_hand and closest_hand
            else 0.0
        )
        relative_movement = abs(object_movement - hand_movement)

        hand_near = hand_distance is not None and hand_distance <= near_threshold
        hand_far = hand_distance is None or hand_distance >= far_threshold
        object_moving = object_movement >= movement_threshold
        hand_moving = hand_movement >= movement_threshold
        moving_together = relative_movement <= relative_threshold

        if self.state == InteractionState.IDLE:
            self.pickup_counter = 0
            self.release_counter = 0
            self.near_counter = self.near_counter + 1 if hand_near else 0
            if self.near_counter >= self.config.near_frames:
                self.state = InteractionState.NEAR_OBJECT

        elif self.state == InteractionState.NEAR_OBJECT:
            if hand_far:
                self.state = InteractionState.IDLE
                self.near_counter = 0
                self.pickup_counter = 0
            elif measured and hand_moving and object_moving and moving_together:
                self.pickup_counter += 1
                if self.pickup_counter >= self.config.pickup_frames:
                    self.state = InteractionState.PICKED_UP
                    self.picked_up_counter = 0
                    self.pickup_counter = 0
            else:
                self.pickup_counter = max(0, self.pickup_counter - 1)

        elif self.state == InteractionState.PICKED_UP:
            self.picked_up_counter += 1
            if self.picked_up_counter >= self.config.picked_up_frames:
                self.state = InteractionState.CARRYING

        elif self.state == InteractionState.CARRYING:
            self.release_counter = self.release_counter + 1 if hand_far else 0
            if self.release_counter >= self.config.release_frames:
                self.state = InteractionState.RELEASED
                self.release_counter = 0
                self.stable_counter = 0

        elif self.state == InteractionState.RELEASED:
            self.stable_counter = self.stable_counter + 1 if object_movement < movement_threshold else 0
            if self.stable_counter >= self.config.stable_frames:
                self.state = InteractionState.IDLE
                self.stable_counter = 0
                self.near_counter = 0

        self._previous_box = box if measured else self._previous_box
        self._previous_hand = closest_hand or self._previous_hand
        return self._result(
            closest_hand,
            hand_distance,
            object_movement,
            hand_movement,
            relative_movement,
        )

    def identity_locked(self) -> bool:
        return self.state in {
            InteractionState.PICKED_UP,
            InteractionState.CARRYING,
            InteractionState.RELEASED,
        }

    def _closest_hand(
        self, box: BBox, wrists: Iterable[Point]
    ) -> tuple[Optional[Point], Optional[float]]:
        best_hand: Optional[Point] = None
        best_distance: Optional[float] = None
        for wrist in wrists:
            distance = point_to_box_distance(wrist, box)
            if best_distance is None or distance < best_distance:
                best_hand = wrist
                best_distance = distance
        return best_hand, best_distance

    def _result(
        self,
        closest_hand: Optional[Point],
        hand_distance: Optional[float],
        object_movement: float,
        hand_movement: float,
        relative_movement: float,
    ) -> InteractionResult:
        return InteractionResult(
            self.state,
            closest_hand,
            hand_distance,
            object_movement,
            hand_movement,
            relative_movement,
            self.pickup_counter,
            self.release_counter,
        )


def detections_from_yolo_result(result: object) -> list[Detection]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return []

    boxes = boxes_obj.xyxy.cpu().numpy()
    confidences = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else [1.0] * len(boxes)
    ids = (
        boxes_obj.id.cpu().numpy().astype(int)
        if boxes_obj.id is not None
        else [None] * len(boxes)
    )
    return [
        Detection(tuple(float(value) for value in box), float(confidence), yolo_id)
        for box, confidence, yolo_id in zip(boxes, confidences, ids)
    ]


def wrists_from_pose_result(result: object, min_confidence: float = 0.2) -> list[Point]:
    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
        return []

    xy = keypoints_obj.xy.cpu().numpy()
    conf = keypoints_obj.conf.cpu().numpy() if keypoints_obj.conf is not None else None
    wrists: list[Point] = []
    for person_index, person in enumerate(xy):
        for keypoint_index in (9, 10):
            point = person[keypoint_index]
            if point[0] <= 0 or point[1] <= 0:
                continue
            if conf is not None and conf[person_index][keypoint_index] < min_confidence:
                continue
            wrists.append((float(point[0]), float(point[1])))
    return wrists


def run_webcam(args: argparse.Namespace) -> None:
    import cv2
    from ultralytics import YOLO

    object_model = YOLO(args.object_model)
    pose_model = YOLO(args.pose_model)
    tracker = SingleBottleTracker(
        TrackerConfig(
            min_confidence=args.confidence,
            max_area_fraction=args.max_area_fraction,
        )
    )
    interaction = InteractionMachine()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_width, frame_height)
        object_result = object_model.track(
            frame,
            persist=True,
            tracker=args.tracker_config,
            conf=args.confidence,
            classes=[39],
            verbose=False,
        )[0]
        pose_result = pose_model.track(
            frame,
            persist=True,
            tracker=args.tracker_config,
            conf=args.confidence,
            verbose=False,
        )[0]

        track = tracker.update(detections_from_yolo_result(object_result), frame_size)
        wrists = wrists_from_pose_result(pose_result)
        state = interaction.update(track.measured, track.box, wrists, frame_size)
        tracker.set_locked(interaction.identity_locked())
        _draw_debug(frame, track, state, wrists)

        cv2.imshow("SIH26174 - single Bottle 1 monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            tracker.reset_for_new_scene()
            interaction = InteractionMachine()

    cap.release()
    cv2.destroyAllWindows()


def _draw_debug(
    frame: object,
    track: TrackResult,
    state: InteractionResult,
    wrists: Iterable[Point],
) -> None:
    import cv2

    for wrist in wrists:
        cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 5, (255, 255, 255), -1)

    if track.box is None:
        cv2.putText(
            frame,
            f"Acquiring target: {track.acquiring_hits}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        return

    x1, y1, x2, y2 = track.box
    color = (0, 255, 0) if track.measured else (0, 180, 255)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    label = f"Bottle 1: {state.state.value}"
    if not track.measured:
        label += f" LOST {track.lost_frames}"
    cv2.putText(
        frame,
        label,
        (int(x1), max(24, int(y1) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    debug = [
        f"Candidates: {track.candidate_count}",
        f"Hand: {'--' if state.hand_distance is None else f'{state.hand_distance:.1f}px'}",
        f"Obj move: {state.object_movement:.1f}px",
        f"Hand move: {state.hand_movement:.1f}px",
        f"Relative: {state.relative_movement:.1f}px",
        f"Pickup: {state.pickup_counter}",
        f"Release: {state.release_counter}",
    ]
    for index, text in enumerate(debug):
        cv2.putText(
            frame,
            text,
            (int(x1), int(y2) + 22 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )

    if state.closest_hand is not None:
        hand = (int(state.closest_hand[0]), int(state.closest_hand[1]))
        target_center = center(track.box)
        cv2.circle(frame, hand, 7, (0, 255, 0), -1)
        cv2.line(frame, hand, (int(target_center[0]), int(target_center[1])), (255, 255, 255), 2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-target SIH 26174 bottle monitor")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--object-model", default="yolo11n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker-config", default="custom_bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--max-area-fraction", type=float, default=0.18)
    return parser


def main() -> None:
    run_webcam(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
