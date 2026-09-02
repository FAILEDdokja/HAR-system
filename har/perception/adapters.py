"""Duck-typed adapters from ultralytics result objects to frozen contracts.

These use ``getattr`` on purpose: they never import ultralytics, so they can be
exercised with plain stand-in objects in tests and they keep cv2/torch out of
the import graph of everything else in the package.

Moved from ``bottle_monitor.py`` (commit ``19c5436``).
"""

from __future__ import annotations

from typing import Any

from har.contracts import Detection, Wrist

__all__ = ["detections_from_yolo_result", "wrists_from_pose_result", "COCO_WRIST_KEYPOINTS"]

#: COCO keypoint indices: 9 = left wrist, 10 = right wrist.
COCO_WRIST_KEYPOINTS: tuple[tuple[int, str], ...] = ((9, "left"), (10, "right"))


def detections_from_yolo_result(result: Any, label_for_class: Any = None) -> list[Detection]:
    """Convert one ultralytics detection result into ``Detection`` objects.

    ``label_for_class`` is an optional ``{int: str}`` map (e.g. ``model.names``)
    so detections carry a protocol label instead of a bare class index.
    """
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return []

    boxes = boxes_obj.xyxy.cpu().numpy()
    confidences = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else [1.0] * len(boxes)
    classes = boxes_obj.cls.cpu().numpy().astype(int) if boxes_obj.cls is not None else [-1] * len(boxes)
    ids = (
        boxes_obj.id.cpu().numpy().astype(int)
        if boxes_obj.id is not None
        else [None] * len(boxes)
    )
    detections: list[Detection] = []
    for box, confidence, class_id, track_id in zip(boxes, confidences, classes, ids):
        label = str(class_id)
        if label_for_class is not None:
            label = str(label_for_class.get(int(class_id), class_id))
        detections.append(
            Detection(
                box=tuple(float(value) for value in box),  # type: ignore[arg-type]
                confidence=float(confidence),
                label=label,
                track_id=None if track_id is None else int(track_id),
            )
        )
    return detections


def wrists_from_pose_result(
    result: Any, min_confidence: float = 0.2
) -> list[Wrist]:
    """Convert one ultralytics pose result into ``Wrist`` objects."""
    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
        return []

    xy = keypoints_obj.xy.cpu().numpy()
    conf = keypoints_obj.conf.cpu().numpy() if keypoints_obj.conf is not None else None
    wrists: list[Wrist] = []
    highest_index = max(index for index, _ in COCO_WRIST_KEYPOINTS)
    for person_index, person in enumerate(xy):
        # ultralytics reports "nobody detected" as a placeholder row with no
        # keypoints at all (keypoints.xy shaped (1, 0, 2) next to empty boxes),
        # not as zero rows — indexing wrist 9 into it raises IndexError and
        # takes the whole frame loop down on the first empty camera frame.
        if len(person) <= highest_index:
            continue
        for keypoint_index, side in COCO_WRIST_KEYPOINTS:
            point = person[keypoint_index]
            if point[0] <= 0 or point[1] <= 0:
                continue
            if conf is not None and conf[person_index][keypoint_index] < min_confidence:
                continue
            wrists.append(
                Wrist(
                    point=(float(point[0]), float(point[1])),
                    confidence=float(conf[person_index][keypoint_index]) if conf is not None else 1.0,
                    side=side,
                    person_id=int(person_index),
                )
            )
    return wrists
