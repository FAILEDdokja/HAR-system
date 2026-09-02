"""B1 - measure what the hardware can do, so everything downstream is sized.

Times ``models/yolo11n-pose.pt`` alone (the shipped design: one model serves
wrists AND the person gate) against the legacy two-model plan (pose + a full
detection pass with ``yolo11n.pt``), at several ``imgsz`` values. The output
is a markdown table meant to be pasted into ``docs/PERF.md``.

Usage (on the demo machine, with ultralytics installed)::

    python -m tools.probe_fps                     # synthetic frames, 480/640
    python -m tools.probe_fps --frames 200
    python -m tools.probe_fps --source 0          # live camera instead
    python -m tools.probe_fps --source demo.mp4   # a recording instead

Exits non-zero with an actionable message when torch/ultralytics is missing -
run it on a machine that has them, never on the bare test interpreter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

POSE_WEIGHTS = REPO_ROOT / "models" / "yolo11n-pose.pt"
DETECT_WEIGHTS = REPO_ROOT / "models" / "yolo11n.pt"


def _open_frames(source: str | None, frame_size: tuple[int, int]):
    """Yield BGR frames from a camera/file, or synthetic noise frames."""
    if source is None:
        height, width = frame_size
        rng = np.random.default_rng(7)
        while True:
            yield rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    import cv2

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open source {source!r}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def _time_pass(model, frames, imgsz: int, measured_frames: int, warmup: int = 10) -> dict:
    iterator = iter(frames)
    count = 0
    start = None
    target = warmup + measured_frames
    for frame in iterator:
        model.predict(frame, imgsz=imgsz, conf=0.45, verbose=False)
        count += 1
        if count == warmup:
            start = time.perf_counter()
        if count >= target:
            break
    elapsed = time.perf_counter() - start if start else 0.0
    measured = max(1, min(measured_frames, count - warmup))
    ms = elapsed / measured * 1000.0
    return {"imgsz": imgsz, "ms": ms, "fps": 1000.0 / ms if ms else 0.0, "frames": measured}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=str(POSE_WEIGHTS))
    parser.add_argument("--detector-weights", default=str(DETECT_WEIGHTS),
                        help="second model for the legacy two-model comparison")
    parser.add_argument("--source", default=None, help="camera index or video file")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[480, 640])
    parser.add_argument("--frames", type=int, default=100, help="measured frames per row")
    parser.add_argument("--frame-size", type=int, nargs=2, default=(640, 480),
                        metavar=("WIDTH", "HEIGHT"), help="synthetic frame size")
    parser.add_argument("--no-compare", action="store_true",
                        help="skip the two-model legacy comparison")
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed in this interpreter. Run this probe "
            "on the demo machine: .venv/bin/python -m tools.probe_fps",
            file=sys.stderr,
        )
        return 1

    pose = YOLO(args.weights)
    rows = []
    for imgsz in args.imgsz:
        frames = _open_frames(args.source, tuple(args.frame_size))
        rows.append({"what": f"pose only (person gate, 1 model)", **_time_pass(pose, frames, imgsz, args.frames)})

    if not args.no_compare:
        try:
            detector = YOLO(args.detector_weights)
        except Exception as exc:  # pragma: no cover
            print(f"skipping two-model comparison: {exc}", file=sys.stderr)
            detector = None
        if detector is not None:
            for imgsz in args.imgsz:
                frames = _open_frames(args.source, tuple(args.frame_size))

                def both(frame, _imgsz=imgsz):
                    detector.predict(frame, imgsz=_imgsz, conf=0.45, verbose=False)
                    pose.predict(frame, imgsz=_imgsz, conf=0.45, verbose=False)

                start = None
                count = 0
                target = 10 + args.frames
                for frame in frames:
                    both(frame)
                    count += 1
                    if count == 10:
                        start = time.perf_counter()
                    if count >= target:
                        break
                elapsed = time.perf_counter() - start if start else 0.0
                measured = max(1, min(args.frames, count - 10))
                ms = elapsed / measured * 1000.0
                rows.append(
                    {
                        "what": "pose + yolo11n (legacy, 2 models)",
                        "imgsz": imgsz,
                        "ms": ms,
                        "fps": 1000.0 / ms if ms else 0.0,
                        "frames": measured,
                    }
                )

    print("\n| Configuration | imgsz | ms / frame | FPS | measured frames |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['what']} | {row['imgsz']} | {row['ms']:.1f} | "
            f"{row['fps']:.1f} | {row['frames']} |"
        )
    print(
        "\nPaste the table into docs/PERF.md, filling in the machine, the "
        "date and the thread count used."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
