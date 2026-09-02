#!/usr/bin/env python3
"""B8 — the rotation demo: rotate the whole rig 90° mid-run, still validate.

Plan step B8 ("done when"): rotate the whole rig 90° mid-run and the sequence
still validates.  This is the honest, cheap version of the optional
"orientation-agnostic" challenge — we track relative to the payload rack
(B7's ``RackFrame`` homography), not to gravity.

How it works, per frame::

    mp4 frame --(after t_rot: cv2.rotate 90 CW)--> camera frame
    ColorDetector(camera frame)  ->  Detections in camera space
    RackFrame.to_rack(box)       ->  Detections in RACK space
    TrackerRegistry + FSMs       ->  FrameEvidence (rack space, rack_ready)
    SequenceValidator            ->  StepEvents

Because trackers, interaction FSMs, zones and predicates all operate in rack
space, the 90° flip mid-run is invisible downstream: boxes and wrists are
continuous across the rotation instant even though every camera pixel moved.

The four fiducials are the frame corners of the rendered rig, mapped
analytically under the applied rotation — the stand-in for the ArUco /
taped-corner detection B7 specifies for live footage (no camera on this
host; the demo mp4s are the committed stand-in dataset).

Usage::

    .venv/bin/python -m tools.rotation_demo                 # rotate at 7.0 s
    .venv/bin/python -m tools.rotation_demo --rotate-at 5
    .venv/bin/python -m tools.rotation_demo --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from har.contracts import Detection, Wrist  # noqa: E402
from har.perception.color_detector import ColorDetector, load_colour_config  # noqa: E402
from har.perception.rack import RackFrame  # noqa: E402
from har.perception.perception import PerceptionStack  # noqa: E402
from tools.crosscheck_g1 import (  # noqa: E402
    FPS,
    FRAME_SIZE,
    INTERACTION,
    LABELS,
    RACK_ROI,
    HandStandIn,
)
from har.protocol.spec import load_protocol  # noqa: E402
from har.protocol.validator import SequenceValidator  # noqa: E402

WIDTH, HEIGHT = FRAME_SIZE

#: Frame-corner fiducials (top_left, top_right, bottom_right, bottom_left).
UPRIGHT_FIDUCIALS = ((0.0, 0.0), (WIDTH, 0.0), (WIDTH, HEIGHT), (0.0, HEIGHT))
#: The same physical corners as seen after a 90° clockwise camera rotation:
#: (x, y) -> (HEIGHT - y, x) for cv2.ROTATE_90_CLOCKWISE.
ROTATED_FIDUCIALS = tuple((HEIGHT - y, x) for x, y in UPRIGHT_FIDUCIALS)
#: rack_roi rectangle in the rotated camera frame.
ROTATED_ROI = (HEIGHT - RACK_ROI[3], RACK_ROI[0], HEIGHT - RACK_ROI[1], RACK_ROI[2])


class RackSpaceDetector:
    """Wrap the colour detector: camera-space hits -> rack-space detections."""

    def __init__(self, detector: ColorDetector, rack: RackFrame) -> None:
        self._detector = detector
        self._rack = rack

    @property
    def backend(self) -> str:
        return f"{self._detector.backend}+rack"

    def detect(self, frame) -> list[Detection]:
        out = []
        for det in self._detector.detect(frame):
            box = self._rack.to_rack(det.box)
            out.append(Detection(box=box, confidence=det.confidence, label=det.label))
        return out


class RackSpaceHands(HandStandIn):
    """Wrist stand-in whose points are expressed in rack space."""

    def __init__(self, rack: RackFrame) -> None:
        super().__init__()
        self._rack = rack

    def wrists(self, frame, frame_index: int) -> list[Wrist]:
        raw = super().wrists(frame, frame_index)
        return [
            Wrist(
                point=self._rack.to_rack_point(w.point),
                confidence=w.confidence,
                side=w.side,
                person_id=w.person_id,
            )
            for w in raw
        ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="B8 rotation demo")
    parser.add_argument("--video", default=REPO / "demo" / "correct.mp4", type=Path)
    parser.add_argument("--protocol", default=REPO / "protocols" / "pts01.yaml", type=Path)
    parser.add_argument("--rotate-at", type=float, default=7.0, metavar="SECONDS",
                        help="rotate the rig 90 degrees clockwise at this t_rel (default 7.0)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    spec = load_protocol(args.protocol, FRAME_SIZE)
    validator = SequenceValidator(spec)

    ranges, options = load_colour_config(REPO / "config" / "colours.yaml")
    colour = ColorDetector(
        ranges,
        roi=RACK_ROI,
        # No temporal smoothing inside the detector: it works in camera space,
        # and blending boxes across the rotation instant would corrupt the
        # rack-space stream.  The trackers smooth in rack space instead.
        median_window=1,
        min_area=int(options.get("min_area", 400)),
    )
    rack = RackFrame(UPRIGHT_FIDUCIALS, FRAME_SIZE)
    stack = PerceptionStack(
        RackSpaceDetector(colour, rack),
        RackSpaceHands(rack),
        LABELS,
        FRAME_SIZE,  # rack space size — evidence geometry is rack-relative
        interaction_config=INTERACTION,
    )
    stack.rack = rack  # evidence rows carry rack_ready=True

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    events = []
    rotated = False
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t_rel = frame_index / FPS
        if t_rel >= args.rotate_at:
            if not rotated:
                rotated = True
                # The rig (camera) physically rotates: re-home the rack frame
                # on the fiducials as now seen, and point the detector's ROI
                # at the rack's new position in the camera frame.
                rack.update(ROTATED_FIDUCIALS)
                colour.roi = ROTATED_ROI
                if args.verbose:
                    print(f"--- rig rotated 90 degrees CW at t={t_rel:.3f}s (frame {frame_index})")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        evidence = stack.process(frame, frame_index, t_rel)
        assert evidence.rack_ready, "rack frame must stay homed through the rotation"
        events.extend(validator.update(evidence))
        frame_index += 1
    capture.release()

    completed = [e.step_id for e in events if e.event == "COMPLETED"]
    violations = [(e.event, e.step_id) for e in events if e.status == "VIOLATION"]
    complete = any(e.event == "PROTOCOL_COMPLETE" for e in events)

    if args.verbose:
        for e in events:
            print(f"  t={e.t_rel:7.3f}  f={e.frame_index:4d}  {e.event:17s} {e.status:11s} {e.step_id}")

    ok = len(completed) == 8 and complete and not violations
    print(
        f"rotation at t={args.rotate_at:.1f}s: {len(completed)} COMPLETED, "
        f"PROTOCOL_COMPLETE={complete}, violations={violations}"
    )
    print(f"B8 ROTATION DEMO: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
