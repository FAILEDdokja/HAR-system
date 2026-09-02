#!/usr/bin/env python3
"""Build the A8 evaluation dataset (Track A — Cognition).

A8 (``docs/DEVELOPMENT_PLAN.md`` §5) asks for three recorded protocol runs
plus a hand annotation of the true start and end time of every step.  That
dataset is what the problem statement means by "our own data": it is used
for evaluation (A9), never for training.

This host has no camera and Track C's recorder (C7) is not on the branch
yet, so the three ``demo/*.mp4`` files are *scripted stand-ins* for webcam
captures.  They replay the A1 evidence fixtures as coloured props moving
through the PTS-01 zones (black tray, yellow lid, red box, blue box, green
vial) at 15 fps / 640×480, which is the same layout the colour detector
is designed to see.  ``--source demo/correct.mp4`` is therefore a real
video file a stock player and ``cv2.VideoCapture`` can both open; once
C5 lands, replaying it is the webcam-fail fallback in §9.

The companion ``*_evidence.json`` files are the frame-accurate
``FrameEvidence`` recording of the same timeline, so A9 can score the
validator without waiting on perception.

Usage (from the repo root, with PyYAML; OpenCV only needed to render)::

    .venv/bin/python demo/build_dataset.py           # evidence + ground truth
    .venv/bin/python demo/build_dataset.py --render  # also write the mp4s
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import CONTRACT_VERSION, FrameEvidence, ObjectTrack, Wrist  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
DEMO = REPO / "demo"
PROTOCOL = REPO / "protocols" / "pts01.yaml"
FPS = 15.0
FRAME_SIZE = (640, 480)
TAIL_S = 2.0  # extra hold after the last authored observation (step 8 is 20 frames)

# Colours are BGR (OpenCV).  Chosen to be trivially separable in HSV so the
# videos remain a valid input for Track B's colour detector once it lands.
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

# Operator-true step windows, hand-annotated from the scripted actions
# (the A1 keyframes).  These are *not* copied from the validator: A9
# measures latency as (validator COMPLETED time) - (annotated t_end).
# Each entry is (t_start, t_end, operator_outcome).  ``None`` times mean
# the operator never performed that step.
OPERATOR_TIMES = {
    "correct": {
        "PRESENT_TRAY": (0.0, 1.0, "COMPLETED"),
        "OPEN_TRAY": (1.0, 2.0, "COMPLETED"),
        "EXTRACT_RED": (2.0, 4.0, "COMPLETED"),
        "VERIFY_RED_PLACED": (4.0, 6.0, "COMPLETED"),
        "EXTRACT_BLUE": (6.0, 8.0, "COMPLETED"),
        "VERIFY_BLUE_PLACED": (8.0, 10.0, "COMPLETED"),
        "SAMPLE_TRANSFER": (10.0, 12.0, "COMPLETED"),
        "STOW_AND_CLOSE": (12.0, 14.5, "COMPLETED"),
    },
    "skip": {
        "PRESENT_TRAY": (0.0, 1.0, "COMPLETED"),
        "OPEN_TRAY": (1.0, 2.0, "COMPLETED"),
        "EXTRACT_RED": (None, None, "SKIPPED"),
        "VERIFY_RED_PLACED": (None, None, "NOT_PERFORMED"),
        "EXTRACT_BLUE": (2.0, 4.0, "COMPLETED"),
        "VERIFY_BLUE_PLACED": (4.0, 6.0, "COMPLETED"),
        "SAMPLE_TRANSFER": (6.0, 7.0, "COMPLETED"),
        "STOW_AND_CLOSE": (7.0, 9.5, "COMPLETED"),
    },
    "wrong_order": {
        "PRESENT_TRAY": (0.0, 1.0, "COMPLETED"),
        "OPEN_TRAY": (1.0, 2.0, "COMPLETED"),
        "EXTRACT_RED": (4.0, 6.0, "COMPLETED"),  # performed *after* blue
        "VERIFY_RED_PLACED": (None, None, "NOT_PERFORMED"),
        "EXTRACT_BLUE": (2.0, 4.0, "COMPLETED"),  # first action after the lid
        "VERIFY_BLUE_PLACED": (None, None, "NOT_PERFORMED"),
        "SAMPLE_TRANSFER": (6.0, 8.0, "COMPLETED"),
        "STOW_AND_CLOSE": (8.0, 10.0, "COMPLETED"),
    },
}

RUNS = (
    ("correct", "evidence_correct.json", "all eight steps in index order"),
    ("skip", "evidence_skip.json", "red box omitted; blue placed first"),
    ("wrong_order", "evidence_wrong_order.json", "blue placed before red"),
)

STEP_META = (
    ("PRESENT_TRAY", 1),
    ("OPEN_TRAY", 2),
    ("EXTRACT_RED", 3),
    ("VERIFY_RED_PLACED", 4),
    ("EXTRACT_BLUE", 5),
    ("VERIFY_BLUE_PLACED", 6),
    ("SAMPLE_TRANSFER", 7),
    ("STOW_AND_CLOSE", 8),
)

ZONES_NORM = {
    "rack_roi": (0.08, 0.15, 0.92, 0.95),
    "tray_slot": (0.34, 0.48, 0.66, 0.88),
    "zone_a": (0.10, 0.50, 0.30, 0.80),
    "zone_b": (0.70, 0.50, 0.90, 0.80),
    "rack_slot": (0.44, 0.18, 0.56, 0.34),
}


def evidence_from_dict(d: dict) -> FrameEvidence:
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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_box(a, b, t):
    if a is None or b is None:
        return b if t >= 1.0 else a
    return tuple(_lerp(a[i], b[i], t) for i in range(4))


def _lerp_point(a, b, t):
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t))


def densify(frames: list[FrameEvidence], fps: float = FPS, tail_s: float = TAIL_S) -> list[FrameEvidence]:
    """Turn a sparse A1 fixture into a 15 fps recording.

    Geometry (boxes, wrists) is linearly interpolated between authored
    keyframes so the mp4 shows continuous motion.  HOI labels are held
    from the most recent keyframe — they are discrete FSM states, not
    quantities we invented.  A short tail of the final pose is appended
    so dwell predicates (``hands_clear`` hold_frames=20) can actually
    accumulate on a dense stream.
    """
    if not frames:
        return []
    by_idx = {f.frame_index: f for f in frames}
    keys = sorted(by_idx)
    last = keys[-1]
    total = last + int(round(tail_s * fps)) + 1
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
            box_a = None if a is None else a.box
            box_b = None if b is None else b.box
            objects[lbl] = ObjectTrack(
                label=src.label,
                box=_lerp_box(box_a, box_b, t),
                measured=src.measured,
                lost_frames=src.lost_frames,
            )
        if prev.hands and next_.hands:
            hands = tuple(
                Wrist(
                    _lerp_point(prev.hands[i].point, next_.hands[min(i, len(next_.hands) - 1)].point, t),
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
                hoi=dict(prev.hoi),  # discrete; held from last keyframe
                rack_ready=prev.rack_ready,
                fps=fps,
            )
        )
    return dense


def replay(frames: list[FrameEvidence]):
    from har.protocol.spec import load_protocol
    from har.protocol.validator import SequenceValidator

    spec = load_protocol(PROTOCOL, FRAME_SIZE)
    validator = SequenceValidator(
        spec,
        start_time=datetime(2026, 9, 2, 14, 30, tzinfo=timezone.utc),
    )
    events = []
    for frame in frames:
        events.extend(validator.update(frame))
    return validator, events


def _outcome_for(step_id: str, events) -> str:
    kinds = {e.event for e in events if e.step_id == step_id}
    if "SKIPPED" in kinds:
        return "SKIPPED"
    if "COMPLETED" in kinds:
        return "COMPLETED"
    if "TIMEOUT" in kinds:
        return "TIMEOUT"
    if "STARTED" in kinds:
        return "INCOMPLETE"
    return "NOT_PERFORMED"


def _px(box_norm):
    w, h = FRAME_SIZE
    x1, y1, x2, y2 = box_norm
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def render_mp4(path: Path, frames: list[FrameEvidence], title: str) -> None:
    """Render one run.  OpenCV is imported here so the rest of A8 stays cv2-free."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")

    draw_order = ("tray", "red_box", "blue_box", "vial", "tray_lid")
    zone_colour = {
        "rack_roi": (110, 110, 110),
        "tray_slot": (90, 90, 90),
        "zone_a": (60, 60, 180),
        "zone_b": (180, 90, 40),
        "rack_slot": (60, 160, 60),
    }

    for ev in frames:
        img = np.full((h, w, 3), COLOURS_BGR["background"], dtype=np.uint8)
        rx1, ry1, rx2, ry2 = _px(ZONES_NORM["rack_roi"])
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), COLOURS_BGR["desk"], thickness=-1)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), COLOURS_BGR["rack"], thickness=2)
        for zid, box in ZONES_NORM.items():
            x1, y1, x2, y2 = _px(box)
            cv2.rectangle(img, (x1, y1), (x2, y2), zone_colour[zid], thickness=1)
            cv2.putText(
                img, zid, (x1 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, zone_colour[zid], 1, cv2.LINE_AA,
            )
        for label in draw_order:
            tr = ev.objects.get(label)
            if tr is None or tr.box is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in tr.box)
            if label == "tray_lid":
                # A lid sits *on* a tray: leave the tray's rim visible, exactly
                # as the evidence recording claims (tray measured=True from
                # frame 0). Without the inset the lid pixel-occludes the tray
                # and the colour detector cannot corroborate the evidence.
                x1, y1, x2, y2 = x1 + 8, y1 + 8, x2 - 8, y2 - 8
            colour = COLOURS_BGR[label]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        for wrist in ev.hands:
            cx, cy = int(round(wrist.point[0])), int(round(wrist.point[1]))
            # Ring, not a filled disc: a filled 14 px disc fully occludes the
            # 20 px vial while it is carried, which contradicts the evidence
            # (vial measured=True through the transfer) and breaks any
            # detector-driven replay of this footage.
            cv2.circle(img, (cx, cy), 14, COLOURS_BGR["hand"], thickness=3)
            cv2.circle(img, (cx, cy), 16, (255, 255, 255), thickness=1)
        cv2.rectangle(img, (0, 0), (w, 28), (24, 24, 24), thickness=-1)
        cv2.putText(
            img,
            f"PTS-01  {title}   t={ev.t_rel:5.2f}s  f={ev.frame_index}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS_BGR["text"], 1, cv2.LINE_AA,
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
    writer.release()
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError(f"rendered {path} is missing or tiny ({path.stat().st_size if path.exists() else 0} bytes)")


def build_ground_truth(built: dict) -> dict:
    runs = {}
    for run_id, _fixture, scenario in RUNS:
        frames: list[FrameEvidence] = built[run_id]["frames"]
        events = built[run_id]["events"]
        validator = built[run_id]["validator"]
        op = OPERATOR_TIMES[run_id]
        steps = []
        for step_id, index in STEP_META:
            t_start, t_end, operator_outcome = op[step_id]
            completed_ev = next((e for e in events if e.step_id == step_id and e.event == "COMPLETED"), None)
            started_ev = next((e for e in events if e.step_id == step_id and e.event == "STARTED"), None)
            skipped_ev = next((e for e in events if e.step_id == step_id and e.event == "SKIPPED"), None)
            ooo_ev = next((e for e in events if e.step_id == step_id and e.event == "OUT_OF_ORDER"), None)
            steps.append({
                "step_id": step_id,
                "index": index,
                "t_start": t_start,
                "t_end": t_end,
                "frame_start": None if t_start is None else int(round(t_start * FPS)),
                "frame_end": None if t_end is None else int(round(t_end * FPS)),
                "operator_outcome": operator_outcome,
                "validator_outcome": _outcome_for(step_id, events),
                "validator_started_s": None if started_ev is None else round(started_ev.t_rel, 3),
                "validator_completed_s": None if completed_ev is None else round(completed_ev.t_rel, 3),
                "validator_skipped_s": None if skipped_ev is None else round(skipped_ev.t_rel, 3),
                "validator_out_of_order_s": None if ooo_ev is None else round(ooo_ev.t_rel, 3),
            })
        runs[run_id] = {
            "video": f"demo/{run_id}.mp4",
            "evidence": f"demo/{run_id}_evidence.json",
            "scenario": scenario,
            "duration_s": round(frames[-1].t_rel, 3),
            "frame_count": len(frames),
            "fps": FPS,
            "frame_size": list(FRAME_SIZE),
            "steps": steps,
            "expected": {
                "completed": list(validator.completed_steps),
                "skipped": list(getattr(validator, "status")().skipped),
                "violations": list(validator.violations),
                "protocol_complete": bool(validator.finished),
                "events": [
                    {
                        "t_rel": round(e.t_rel, 3),
                        "frame_index": e.frame_index,
                        "step_id": e.step_id,
                        "step_index": e.step_index,
                        "event": e.event,
                        "status": e.status,
                        "message": e.message,
                    }
                    for e in events
                ],
            },
        }
    return {
        "protocol_id": "PTS-01",
        "protocol_version": "1.0.0",
        "contract_version": CONTRACT_VERSION,
        "annotator": "Person A",
        "annotated_at": "2026-09-02",
        "fps": FPS,
        "frame_size": list(FRAME_SIZE),
        "notes": (
            "Hand-annotated true start/end of each PTS-01 step, in seconds from "
            "the start of the recording (t_rel).  t_start/t_end describe the "
            "operator's actions, not the validator's hold delay — A9 scores "
            "latency as (COMPLETED t_rel) - t_end.  The mp4 files are scripted "
            "stand-ins for webcam captures: this host has no camera and C7's "
            "recorder is not on the branch.  They share the evidence timeline, "
            "so `--source demo/correct.mp4` and a validator replay of "
            "demo/correct_evidence.json produce the same step log."
        ),
        "runs": runs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the A8 demo dataset")
    ap.add_argument("--render", action="store_true", help="also write demo/*.mp4 (needs OpenCV)")
    ap.add_argument("--out-dir", type=Path, default=DEMO)
    args = ap.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    built = {}
    for run_id, fixture_name, scenario in RUNS:
        raw = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
        sparse = [evidence_from_dict(d) for d in raw]
        frames = densify(sparse)
        validator, events = replay(frames)
        evidence_path = out / f"{run_id}_evidence.json"
        evidence_path.write_text(
            json.dumps([f.to_dict() for f in frames], indent=2),
            encoding="utf-8",
        )
        built[run_id] = {
            "frames": frames,
            "events": events,
            "validator": validator,
            "scenario": scenario,
        }
        completed = sum(1 for e in events if e.event == "COMPLETED")
        skipped = sum(1 for e in events if e.event == "SKIPPED")
        ooo = sum(1 for e in events if e.event == "OUT_OF_ORDER")
        print(
            f"{run_id:<12} {len(frames):4d} frames  "
            f"COMPLETED={completed} SKIPPED={skipped} OUT_OF_ORDER={ooo}  "
            f"finished={validator.finished}  -> {evidence_path.name}"
        )
        if args.render:
            mp4 = out / f"{run_id}.mp4"
            render_mp4(mp4, frames, scenario)
            print(f"             wrote {mp4} ({mp4.stat().st_size} bytes)")

    gt = build_ground_truth(built)
    gt_path = out / "ground_truth.json"
    gt_path.write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
    print(f"ground truth -> {gt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
