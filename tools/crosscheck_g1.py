#!/usr/bin/env python3
"""B5 / Gate G1 — cross-check the perception stack against the validator.

Plan step B5 ("done when"): B4's ``PerceptionStack`` output on synthetic
protocol footage is accepted by A4's ``SequenceValidator`` with no contract
errors and produces 7 ``COMPLETED`` (the live demo build drops the
SAMPLE_TRANSFER/vial step).  Person C's
``tools/make_synthetic_video.py`` (C4) never landed, so the committed
``demo/*.mp4`` files — Person A's scripted stand-ins, rendered in exactly the
layout the colour detector is designed for — serve as the synthetic footage.

The full chain exercised per frame::

    mp4 frame -> ColorDetector (config/colours.yaml)
              -> TrackerRegistry + InteractionMachine (PerceptionStack)
              -> FrameEvidence -> SequenceValidator -> StepEvent

Wrists come from an HSV stand-in: the rendered videos draw hands as coloured
rings, which no pose network can see.  The contracts are duck-typed for
exactly this swap (plan §4); on live footage ``WristExtractor`` (B3) slots in
unchanged.

Pass criteria per run (from ``demo/ground_truth.json`` — the committed
expected logs, which encode the validator's one-shot violation design):

* ``correct``      7 COMPLETED in index order + PROTOCOL_COMPLETE, 0 violations
* ``skip``         exactly one OUT_OF_ORDER (EXTRACT_BLUE), no completion
* ``wrong_order``  exactly one OUT_OF_ORDER (EXTRACT_BLUE), no completion

Usage::

    .venv/bin/python -m tools.crosscheck_g1            # exits 0 on pass
    .venv/bin/python -m tools.crosscheck_g1 --verbose  # full event streams
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from har.contracts import Wrist  # noqa: E402
from har.perception.color_detector import ColorDetector, load_colour_config  # noqa: E402
from har.perception.interaction import InteractionConfig  # noqa: E402
from har.perception.perception import PerceptionStack  # noqa: E402
from har.protocol.spec import load_protocol  # noqa: E402
from har.protocol.validator import SequenceValidator  # noqa: E402

FRAME_SIZE = (640, 480)
FPS = 15.0
# Live demo build: the vial/SAMPLE_TRANSFER step is not part of the scored
# protocol, so the perception stack tracks the four scored props only.
LABELS = ("tray", "tray_lid", "red_box", "blue_box")

#: rack_roi of protocols/pts01.yaml at 640x480 — the demo footage's rack
#: position.  A venue/runtime setting, deliberately not in the shared yaml.
RACK_ROI = (52, 72, 588, 456)

# Hand ring colour in demo/build_dataset.py: BGR (92, 150, 230) -> HSV ~(12, 153, 230)
HAND_LO = (9, 110, 190)
HAND_HI = (16, 190, 255)

#: Interaction thresholds retuned for this 15 fps footage (B6-style pass):
#: interpolated motion is ~5 px/frame, under the webcam-tuned default
#: movement threshold (0.012 x 800 px diagonal = 9.6 px/frame).
INTERACTION = InteractionConfig(
    movement_fraction=0.004,
    near_frames=2,
    pickup_frames=3,
    picked_up_frames=2,
    release_frames=4,
    stable_frames=4,
)


class HandStandIn:
    """Duck-typed WristExtractor for the rendered demo footage."""

    def __init__(self) -> None:
        self._last: list[Wrist] = []
        self.person_present = True  # the scripted operator never leaves frame

    def wrists(self, frame, frame_index: int) -> list[Wrist]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HAND_LO, HAND_HI)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            if cv2.contourArea(contour) < 60:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            points.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
        if not points:
            # B3 rule: a frame with no fresh hands repeats the previous result,
            # never an empty list — an empty list reads as "hands vanished".
            return self._last
        points.sort()
        wrists = [
            Wrist(point=p, confidence=0.99, side="left" if i == 0 else "right", person_id=0)
            for i, p in enumerate(points[:2])
        ]
        self._last = wrists
        return wrists


def build_stack() -> PerceptionStack:
    ranges, options = load_colour_config(REPO / "config" / "colours.yaml")
    detector = ColorDetector(
        ranges,
        roi=RACK_ROI,
        median_window=int(options.get("median_window", 5)),
        min_area=int(options.get("min_area", 400)),
    )
    return PerceptionStack(
        detector, HandStandIn(), LABELS, FRAME_SIZE, interaction_config=INTERACTION
    )


def run(video: Path, protocol_path: Path) -> list:
    spec = load_protocol(protocol_path, FRAME_SIZE)
    validator = SequenceValidator(spec)
    stack = build_stack()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    events = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        evidence = stack.process(frame, frame_index, frame_index / FPS)
        events.extend(validator.update(evidence))
        frame_index += 1
    capture.release()
    return events


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="B5 / G1 perception-vs-validator cross-check")
    parser.add_argument("--protocol", default=REPO / "protocols" / "pts01.yaml", type=Path)
    parser.add_argument("--demo", default=REPO / "demo", type=Path)
    parser.add_argument("--verbose", action="store_true", help="print every event")
    args = parser.parse_args(argv)

    ok_all = True
    for run_id in ("correct", "skip", "wrong_order"):
        events = run(args.demo / f"{run_id}.mp4", args.protocol)
        completed = [e.step_id for e in events if e.event == "COMPLETED"]
        violations = [(e.event, e.step_id) for e in events if e.status == "VIOLATION"]
        complete = any(e.event == "PROTOCOL_COMPLETE" for e in events)

        if args.verbose:
            print(f"\n=== {run_id} ===")
            for e in events:
                print(f"  t={e.t_rel:7.3f}  f={e.frame_index:4d}  {e.event:17s} {e.status:11s} {e.step_id}")

        if run_id == "correct":
            ok = len(completed) == 7 and complete and not violations
            detail = f"{len(completed)} COMPLETED, complete={complete}, violations={violations}"
        else:
            ok = (
                violations == [("OUT_OF_ORDER", "EXTRACT_BLUE")]
                and not complete
                and len(completed) < 7
            )
            detail = f"violations={violations}, complete={complete}"
        ok_all &= ok
        print(f"{run_id:<12} {'PASS' if ok else 'FAIL'}  {detail}")

    print(f"\nGATE G1 (B5 cross-check): {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
