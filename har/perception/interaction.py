"""Hand-object interaction FSM.

Moved from ``bottle_monitor.InteractionMachine`` (commit ``19c5436``). The
state arithmetic is unchanged; it is now driven by a ``label`` so a registry of
machines can run one per protocol object, and its state values are mirrored by
``har.contracts.HandObjectState``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from har.contracts import BBox, Point
from har.perception.geometry import center, distance, point_to_box_distance

__all__ = ["InteractionState", "InteractionConfig", "InteractionResult", "InteractionMachine"]


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
    label: str
    state: InteractionState
    closest_hand: Optional[Point]
    hand_distance: Optional[float]
    object_movement: float
    hand_movement: float
    relative_movement: float
    pickup_counter: int
    release_counter: int


class InteractionMachine:
    """IDLE -> NEAR_OBJECT -> PICKED_UP -> CARRYING -> RELEASED -> IDLE.

    A pickup is only declared when the hand is near, the object is moving, the
    hand is moving, and the two move *together* (small relative movement). That
    last condition is what separates "picked it up" from "waved a hand past it".
    """

    def __init__(self, label: str = "object", config: InteractionConfig | None = None) -> None:
        self.label = label
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
            distance(center(self._previous_box), center(box))
            if self._previous_box and measured
            else 0.0
        )
        hand_movement = (
            distance(self._previous_hand, closest_hand)
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
            wrist_distance = point_to_box_distance(wrist, box)
            if best_distance is None or wrist_distance < best_distance:
                best_hand = wrist
                best_distance = wrist_distance
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
            self.label,
            self.state,
            closest_hand,
            hand_distance,
            object_movement,
            hand_movement,
            relative_movement,
            self.pickup_counter,
            self.release_counter,
        )
