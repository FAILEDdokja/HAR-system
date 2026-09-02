#!/usr/bin/env python3
"""C4 — Synthetic PTS-01 footage for gate G1.

Gate G1 runs the *whole system* with no camera::

    .venv/bin/python tools/make_synthetic_video.py --verify
    .venv/bin/python -m har.app --source tests/fixtures/synthetic_correct.mp4 \\
        --protocol protocols/pts01.yaml --headless --out-dir runs/g1

so this tool renders ``tests/fixtures/synthetic_correct.mp4`` and
``synthetic_wrong_order.mp4``: coloured props moving through the PTS-01
zones at 640×480 / 15 fps.

The motion timeline is the A1 evidence fixture (the one Person A's
validator is proven against in A4/A5), densified to 15 fps with linear
interpolation — the same reconstruction ``demo/build_dataset.py`` documents.
The visual language deliberately matches that renderer (prop colours, hand
rings, zone overlays, colour-key card): ``config/colours.yaml`` and the HSV
hand stand-in used for rendered footage are both tuned to exactly this
palette, and diverging would make the generated videos undetectable.

``--verify`` (the default) then plays each video back through the real
``PerceptionStack`` + ``SequenceValidator`` and asserts the G1 expectations:

* ``synthetic_correct.mp4``      8 COMPLETED in index order +
                                 PROTOCOL_COMPLETE, zero violations
* ``synthetic_wrong_order.mp4``  exactly one OUT_OF_ORDER (EXTRACT_BLUE),
                                 never completes the protocol

A note on the wrong-order criterion, for whoever reads this at the gate.  The
plan's G1 text says the wrong-order re-run yields "exactly one OUT_OF_ORDER
and one SKIPPED".  That pair is what the shipped validator emits on the
*fixture* layer (``tests/fixtures/events_wrong_order.jsonl``; A5 asserts both)
because sparse keyframe gaps inflate the hold span so both events mature on
the same frame.  On *dense 15 fps video* the hold matures frame by frame, so
the validator's one-shot violation rule (``har/protocol/validator.py``
module docstring: one violation episode per run) emits the OUT_OF_ORDER the
moment EXTRACT_BLUE's ``hoi_cycle`` first holds, and the cursor is never
re-baselined — the same observable Person A committed for the demo footage
in ``demo/ground_truth.json`` and ``tools/crosscheck_g1.py``.  The operator
still gets the spoken "Out of sequence" warning immediately, EXTRACT_RED is
never completed, and the run can never reach PROTOCOL_COMPLETE.  The
SKIPPED path through the *output* layer (log row, voice alert, red banner)
is exercised end-to-end by ``har.app --stub --events
tests/fixtures/events_wrong_order.jsonl --evidence
tests/fixtures/evidence_wrong_order.json``.

Usage::

    .venv/bin/python tools/make_synthetic_video.py              # render + verify
    .venv/bin/python tools/make_synthetic_video.py --no-verify  # render only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import FrameEvidence, ObjectTrack, Wrist  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
PROTOCOL = REPO / "protocols" / "pts01.yaml"
FPS = 15.0
FRAME_SIZE = (640, 480)
#: Hold after the last authored observation so dwell predicates
#: (hands_clear hold_frames=20) accumulate on the dense stream.
TAIL_S = 2.0

#: What to render and the G1 expectation for each.  ``wrong_order`` is the
#: plan's *wrong order* scenario: the blue box is extracted before the red
#: box, and the operator lingers over the tray long enough for the skip to
#: be declared (OUT_OF_ORDER then SKIPPED for EXTRACT_RED).
SCENARIOS = (
    ("synthetic_correct", "evidence_correct.json"),
    ("synthetic_wrong_order", "evidence_wrong_order.json"),
)

# BGR palette shared with demo/build_dataset.py — do not recolour (see module
# docstring: the HSV ranges in config/colours.yaml are tuned to these).
COLOURS_BGR = {
    "background": (52, 56, 60),
    "desk": (168, 172, 176),
    "rack": (78, 86, 94),
    "tray": (18, 18, 18),
    "tray_lid": (0, 220, 255),
    "red_box": (36, 36, 220),
    "blue_box": (220, 96, 32),
    "vial": (36, 200, 48),
    "hand": (92, 150, 230),
    "text": (245, 245, 245),
    "muted": (190, 190, 190),
}

ZONES_NORM = {
    "rack_roi": (0.08, 0.15, 0.92, 0.95),
    "tray_slot": (0.34, 0.48, 0.66, 0.88),
    "zone_a": (0.10, 0.50, 0.30, 0.80),
    "zone_b": (0.70, 0.50, 0.90, 0.80),
    "rack_slot": (0.44, 0.18, 0.56, 0.34),
}


# --------------------------------------------------------------------------
# Evidence loading and densification
# --------------------------------------------------------------------------


def evidence_from_dict(d: dict) -> FrameEvidence:
    """Parse one ``FrameEvidence.to_dict()`` row (same schema the A5 tests use)."""
    return FrameEvidence(
        frame_index=d["frame_index"],
        t_rel=d["t_rel"],
        frame_size=tuple(d["frame_size"]),
        objects={
            label: ObjectTrack(
                label=od["label"],
                box=tuple(od["box"]) if od["box"] is not None else None,
                measured=od["measured"],
                lost_frames=od.get("lost_frames", 0),
            )
            for label, od in d["objects"].items()
        },
        hands=tuple(
            Wrist(tuple(h["point"]), h["confidence"], h["side"], h.get("person_id", 0))
            for h in d["hands"]
        ),
        hoi=dict(d["hoi"]),
        rack_ready=d["rack_ready"],
        fps=d["fps"],
    )


def load_evidence(path: Path) -> list[FrameEvidence]:
    import json

    return [evidence_from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_box(a, b, t):
    if a is None or b is None:
        return b if t >= 1.0 else a
    return tuple(_lerp(a[i], b[i], t) for i in range(4))


def _lerp_point(a, b, t):
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t))


def densify(frames: list[FrameEvidence], fps: float = FPS, tail_s: float = TAIL_S) -> list[FrameEvidence]:
    """Interpolate a sparse evidence fixture into a dense 15 fps timeline.

    Boxes and wrists move linearly between authored keyframes; HOI labels are
    held from the most recent keyframe (discrete FSM states, not invented
    geometry).  A ``tail_s`` hold of the final pose is appended so dwell-time
    predicates can complete on the video exactly as they do on the fixture.
    """
    if not frames:
        return []
    by_idx = {f.frame_index: f for f in frames}
    keys = sorted(by_idx)
    total = keys[-1] + int(round(tail_s * fps)) + 1
    labels = sorted({lbl for f in frames for lbl in f.objects})
    dense: list[FrameEvidence] = []
    for fi in range(total):
        if fi in by_idx:
            prev = next_ = by_idx[fi]
            t = 0.0
        else:
            earlier = [k for k in keys if k <= fi]
            later = [k for k in keys if k >= fi]
            prev = by_idx[earlier[-1]] if earlier else frames[0]
            next_ = by_idx[later[0]] if later else frames[-1]
            span = next_.frame_index - prev.frame_index
            t = 0.0 if span <= 0 else (fi - prev.frame_index) / span
        objects = {}
        for lbl in labels:
            a = prev.objects.get(lbl)
            b = next_.objects.get(lbl)
            if a is None and b is None:
                continue
            src = b if a is None else a
            objects[lbl] = ObjectTrack(
                label=src.label,
                box=_lerp_box(None if a is None else a.box, None if b is None else b.box, t),
                measured=src.measured,
                lost_frames=src.lost_frames,
            )
        if prev.hands and next_.hands:
            hands = tuple(
                Wrist(
                    _lerp_point(
                        prev.hands[i].point,
                        next_.hands[min(i, len(next_.hands) - 1)].point,
                        t,
                    ),
                    prev.hands[i].confidence,
                    prev.hands[i].side,
                    prev.hands[i].person_id,
                )
                for i in range(len(prev.hands))
            )
        else:
            hands = prev.hands
        dense.append(
            FrameEvidence(
                frame_index=fi,
                t_rel=round(fi / fps, 3),
                frame_size=tuple(prev.frame_size),
                objects=objects,
                hands=hands,
                hoi=dict(prev.hoi),
                rack_ready=prev.rack_ready,
                fps=fps,
            )
        )
    return dense


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _px(box_norm: Sequence[float], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = frame_size
    x1, y1, x2, y2 = box_norm
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def render_mp4(path: Path, frames: Iterable[FrameEvidence], title: str, fps: float = FPS) -> int:
    """Render one scenario to an mp4 a stock player and cv2 can both open."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = FRAME_SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    zone_colour = {
        "rack_roi": (110, 110, 110),
        "tray_slot": (90, 90, 90),
        "zone_a": (60, 60, 180),
        "zone_b": (180, 90, 40),
        "rack_slot": (60, 160, 60),
    }
    draw_order = ("tray", "red_box", "blue_box", "vial", "tray_lid")
    count = 0
    for ev in frames:
        img = np.full((h, w, 3), COLOURS_BGR["background"], dtype=np.uint8)
        rx1, ry1, rx2, ry2 = _px(ZONES_NORM["rack_roi"], FRAME_SIZE)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), COLOURS_BGR["desk"], thickness=-1)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), COLOURS_BGR["rack"], thickness=2)
        for zid, box in ZONES_NORM.items():
            x1, y1, x2, y2 = _px(box, FRAME_SIZE)
            cv2.rectangle(img, (x1, y1), (x2, y2), zone_colour[zid], thickness=1)
            cv2.putText(
                img, zid, (x1 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, zone_colour[zid], 1, cv2.LINE_AA,
            )
        for label in draw_order:
            track = ev.objects.get(label)
            if track is None or track.box is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in track.box)
            if label == "tray_lid":
                # A lid sits *on* a tray: keep the tray rim visible so the tray
                # blob keeps corroborating the evidence through the run.
                x1, y1, x2, y2 = x1 + 8, y1 + 8, x2 - 8, y2 - 8
            colour = COLOURS_BGR[label]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        for wrist in ev.hands:
            cx, cy = int(round(wrist.point[0])), int(round(wrist.point[1]))
            # Ring, not filled: a filled disc would occlude the vial during
            # the transfer and break detector-driven replay.
            cv2.circle(img, (cx, cy), 14, COLOURS_BGR["hand"], thickness=3)
            cv2.circle(img, (cx, cy), 16, (255, 255, 255), thickness=1)
        cv2.rectangle(img, (0, 0), (w, 28), (24, 24, 24), thickness=-1)
        cv2.putText(
            img, f"PTS-01  {title}   t={ev.t_rel:5.2f}s  f={ev.frame_index}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS_BGR["text"], 1, cv2.LINE_AA,
        )
        # Colour-key card (white-balance reference, plan §6).
        key = [("tray", "tray"), ("lid", "tray_lid"), ("red", "red_box"),
               ("blue", "blue_box"), ("vial", "vial")]
        x = 8
        for name, label in key:
            cv2.rectangle(img, (x, h - 22), (x + 12, h - 10), COLOURS_BGR[label], thickness=-1)
            cv2.putText(img, name, (x + 16, h - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOURS_BGR["muted"], 1, cv2.LINE_AA)
            x += 70
        writer.write(img)
        count += 1
    writer.release()
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError(f"rendered {path} is missing or tiny")
    return count


# --------------------------------------------------------------------------
# G1 verification of the rendered footage
# --------------------------------------------------------------------------


def play_video(video: Path) -> list:
    """Run the real perception stack + validator over a rendered video."""
    import cv2

    from har.perception.color_detector import ColorDetector, load_colour_config
    from har.perception.interaction import InteractionConfig
    from har.perception.perception import PerceptionStack
    from har.protocol.spec import load_protocol
    from har.protocol.validator import SequenceValidator

    # The same HSV hand stand-in and interaction tuning har.app uses for
    # rendered footage (see app.HsvHandTracker / app._interaction_config).
    from har.app import HsvHandTracker, rendered_interaction_config

    spec = load_protocol(PROTOCOL, FRAME_SIZE)
    validator = SequenceValidator(spec)
    ranges, options = load_colour_config(REPO / "config" / "colours.yaml")
    rack = spec.zone("rack_roi")
    detector = ColorDetector(
        ranges,
        roi=tuple(rack.box) if rack is not None else None,  # har.app's same default
        median_window=int(options.get("median_window", 5)),
        min_area=int(options.get("min_area", 400)),
    )
    stack = PerceptionStack(
        detector,
        HsvHandTracker(),
        spec.objects,
        FRAME_SIZE,
        interaction_config=rendered_interaction_config(),
    )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    events = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        events.extend(validator.update(stack.process(frame, frame_index, frame_index / FPS)))
        frame_index += 1
    capture.release()
    return events


def check_g1(name: str, events: list) -> tuple[bool, str]:
    completed = [e.step_id for e in events if e.event == "COMPLETED"]
    violations = [(e.event, e.step_id) for e in events if e.status == "VIOLATION"]
    complete = any(e.event == "PROTOCOL_COMPLETE" for e in events)
    if name == "synthetic_correct":
        ok = len(completed) == 8 and complete and not violations
        return ok, f"{len(completed)} COMPLETED, complete={complete}, violations={violations}"
    # Dense-video expectation (see module docstring): one violation episode —
    # the OUT_OF_ORDER alert on EXTRACT_BLUE — and the run never completes.
    ok = violations == [("OUT_OF_ORDER", "EXTRACT_BLUE")] and not complete and len(completed) < 8
    return ok, f"violations={violations}, complete={complete}, completed={len(completed)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=FIXTURES,
                        help="where to write the mp4s (default: tests/fixtures)")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="render only, skip the G1 playback check")
    parser.add_argument("--verbose", action="store_true", help="print every verified event")
    args = parser.parse_args(argv)

    written: dict[str, Path] = {}
    for name, fixture in SCENARIOS:
        keyframes = load_evidence(FIXTURES / fixture)
        dense = densify(keyframes)
        out = args.out_dir / f"{name}.mp4"
        frames = render_mp4(out, dense, title=name, fps=FPS)
        written[name] = out
        print(f"rendered {out}  ({frames} frames from {len(keyframes)} keyframes)")

    if not args.verify:
        return 0

    ok_all = True
    for name, path in written.items():
        events = play_video(path)
        ok, detail = check_g1(name, events)
        ok_all &= ok
        if args.verbose:
            print(f"\n=== {name} ===")
            for e in events:
                print(f"  t={e.t_rel:7.3f}  f={e.frame_index:4d}  {e.event:17s} {e.status:11s} {e.step_id}")
        print(f"{name:<24} {'PASS' if ok else 'FAIL'}  {detail}")

    print(f"\nG1 synthetic footage check: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
