"""CLI entrypoint and composition root (C5).

``har.app`` is the only module in Track C that may import Person A's
``SequenceValidator`` and Person B's ``PerceptionStack``; everything in
``har.out`` and ``har.ui`` receives plain contract data.  It wires::

    frame source -> PerceptionStack -> FrameEvidence -> SequenceValidator
        -> StepEvent -> { JsonlEventLog, OfflineSpeaker, EventRing (GUI) }
    frame source -> draw_hud -> { VideoRecorder, MjpegStreamer }

Run modes (plan §1.4: *a demo that cannot fail*):

* **Live camera:**  ``--source 0`` — pose wrists from ``yolo11n-pose.pt`` (the
  model doubles as the person gate, §2), colour detector for the five props.
* **File replay:**  ``--source demo/correct.mp4`` — the entire demo replays
  from a recording if the venue webcam misbehaves.  Rendered footage (the
  shipped demo videos and C4's synthetic ones) draws hands as orange rings no
  pose network can see, so file sources default to an HSV ring stand-in
  (``--wrists hsv``); real recordings should pass ``--wrists pose``.
* **Stub:**  ``--stub`` replays the A1 evidence fixture through a
  ``StubValidator`` that re-emits the canned ``events_correct.jsonl`` — the
  whole output layer runs with no camera, no cv2 and no perception at all.
  This is how C5 was built before A4/B4 landed.

Heavy dependencies (cv2, numpy, ultralytics, PyYAML, flask, pyttsx3) are
imported lazily inside functions, so ``python -m har.app --help`` and the
stub path work in a bare interpreter, and the C-suite stays importable for
the 28-test dependency-free baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO = Path(__file__).resolve().parents[1]

from har.contracts import CONTRACT_VERSION, StepEvent, UiStatus

__all__ = [
    "build_arg_parser",
    "main",
    "HsvHandTracker",
    "NullWrists",
    "StubPerception",
    "StubValidator",
    "EventRing",
    "rendered_interaction_config",
]

DEFAULT_PROTOCOL = REPO / "protocols" / "pts01.yaml"
DEFAULT_COLOURS = REPO / "config" / "colours.yaml"
DEFAULT_POSE_WEIGHTS = REPO / "models" / "yolo11n-pose.pt"
DEFAULT_YOLO_WEIGHTS = REPO / "models" / "yolo11n.pt"
DEFAULT_EVIDENCE_FIXTURE = REPO / "tests" / "fixtures" / "evidence_correct.json"
DEFAULT_EVENTS_FIXTURE = REPO / "tests" / "fixtures" / "events_correct.jsonl"
FALLBACK_FPS = 15.0

# Hand-ring colour of the rendered footage (BGR 92,150,230) in OpenCV HSV.
HAND_HSV_LO = (9, 110, 190)
HAND_HSV_HI = (16, 190, 255)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Wrist backends (duck-typed contracts.WristExtractor seam, plan §4)
# --------------------------------------------------------------------------


class HsvHandTracker:
    """Wrist stand-in for *rendered* footage (demo and synthetic videos).

    The stand-in videos draw the operator's hands as orange rings that no
    pose network can see, so live pose is replaced by an HSV blob detector
    over exactly that ring colour.  Follows the B3 rule: a frame with no
    visible hands repeats the previous result rather than reporting empty
    ("hands vanished" corrupts the interaction FSM).
    """

    def __init__(self, lo: tuple[int, int, int] = HAND_HSV_LO, hi: tuple[int, int, int] = HAND_HSV_HI) -> None:
        self.lo = lo
        self.hi = hi
        self._last: list = []
        #: The scripted operator never leaves frame, so the perception
        #: stack's person gate stays open for rendered footage.
        self.person_present = True

    def wrists(self, frame: Any, frame_index: int) -> list:
        import cv2

        from har.contracts import Wrist

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            if cv2.contourArea(contour) < 60:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                points.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
        if not points:
            return list(self._last)
        points.sort()
        self._last = [
            Wrist(point=p, confidence=0.99, side="left" if i == 0 else "right", person_id=0)
            for i, p in enumerate(points[:2])
        ]
        return list(self._last)

    def reset(self) -> None:
        self._last = []


class NullWrists:
    """No wrist signal at all.  Colour detection still runs (the person gate
    is held open), but manipulation steps cannot be observed — the CLI warns
    loudly when this backend is in force."""

    person_present = True

    def wrists(self, frame: Any, frame_index: int) -> list:
        return []

    def reset(self) -> None:
        pass


class _ImgszModel:
    """Add an ``imgsz`` argument to a duck-typed ultralytics model.

    ``WristExtractor`` (Person B's file) calls ``predict(frame, conf=...,
    verbose=False)``; the CLI's ``--imgsz`` latency lever is applied here, at
    the composition root, without touching the perception package.
    """

    def __init__(self, model: Any, imgsz: int) -> None:
        self._model = model
        self._imgsz = int(imgsz)

    def predict(self, frame: Any, conf: float = 0.35, verbose: bool = False, **kw: Any) -> Any:
        return self._model.predict(frame, conf=conf, verbose=verbose, imgsz=self._imgsz, **kw)


def rendered_interaction_config():
    """Interaction thresholds for rendered (15 fps, interpolated) footage.

    Live-webcam defaults (``InteractionConfig``) expect ~10 px/frame motion;
    the interpolated demo footage moves ~5 px/frame, so the pickup detector
    never fires.  These are the same B6-style retuned values
    ``tools/crosscheck_g1.py`` was validated with against the demo footage,
    centralised here so C4's ``--verify`` and the CLI share one definition.
    """
    from har.perception.interaction import InteractionConfig

    return InteractionConfig(
        movement_fraction=0.004,
        near_frames=2,
        pickup_frames=3,
        picked_up_frames=2,
        release_frames=4,
        stable_frames=4,
    )


# --------------------------------------------------------------------------
# Stub components (C5: build the output layer before A4/B4 exist)
# --------------------------------------------------------------------------


class StubPerception:
    """Replays a recorded ``FrameEvidence`` sequence (the A1 fixture)."""

    def __init__(self, frames: Sequence, labels: Sequence[str] = ()) -> None:
        self._frames = list(frames)
        if not self._frames:
            raise ValueError("StubPerception needs at least one evidence frame")
        self.fps = self._frames[0].fps

    def __len__(self) -> int:
        return len(self._frames)

    def process(self, frame: Any, frame_index: int, t_rel: float):
        return self._frames[min(frame_index, len(self._frames) - 1)]

    def reset(self) -> None:
        pass


class StubValidator:
    """Re-emits canned ``StepEvent`` rows in ``t_rel`` order.

    Stands in for ``SequenceValidator`` behind ``--stub``: same interface
    (``update`` / ``status`` / ``finished``), driven by the fixture clock so
    the log, voice and GUI all see a faithful replay.
    """

    def __init__(self, events: Sequence[StepEvent], spec: Any) -> None:
        self._pending = deque(sorted(events, key=lambda e: (e.t_rel, e.frame_index)))
        self._spec = spec
        self._steps = spec.steps
        self._emitted: list[StepEvent] = []
        self._t_rel = 0.0
        self._fps = 0.0
        self._cursor = 0

    def update(self, evidence) -> list[StepEvent]:
        out: list[StepEvent] = []
        self._t_rel, self._fps = evidence.t_rel, evidence.fps
        while self._pending and self._pending[0].t_rel <= evidence.t_rel + 1e-9:
            out.append(self._pending.popleft())
        self._emitted.extend(out)
        return out

    def drain(self) -> list[StepEvent]:
        """Emit everything still pending, at end of replay.

        The canned events fixture was authored on a slightly longer timeline
        than the evidence fixture, so a naive replay stops early; draining
        when the evidence runs out keeps the stub faithful to the canned log.
        """
        out = list(self._pending)
        self._pending.clear()
        self._emitted.extend(out)
        return out

    def status(self) -> UiStatus:
        started = self._steps[0]
        completed, skipped, violations = [], [], []
        last_alert = ""
        state = "NOT_STARTED"
        for e in self._emitted:
            state = "IN_PROGRESS"
            if e.event == "STARTED":
                step = self._spec.step(e.step_id)
                if step is not None:
                    started = step
            elif e.event == "COMPLETED":
                completed.append(e.step_id)
            elif e.event == "SKIPPED":
                skipped.append(e.step_id)
            if e.status == "VIOLATION":
                if e.step_id not in violations:
                    violations.append(e.step_id)
                last_alert = e.message
            if e.event == "PROTOCOL_COMPLETE":
                state = "COMPLETE"
        next_step = self._steps[started.index] if started.index < len(self._steps) else None
        return UiStatus(
            protocol_id=self._spec.protocol_id,
            protocol_title=self._spec.title,
            current_step_id=started.step_id,
            current_step_index=started.index,
            next_step_id=next_step.step_id if next_step else "",
            next_instruction=next_step.instruction if next_step else "",
            completed=tuple(completed),
            skipped=tuple(skipped),
            violations=tuple(violations),
            state=state,
            t_rel=self._t_rel,
            fps=self._fps,
            last_alert=last_alert,
            contract_version=CONTRACT_VERSION,
        )

    def reset(self) -> None:
        raise NotImplementedError("stub replays once")

    @property
    def current(self):
        return None

    @property
    def completed_steps(self) -> tuple[str, ...]:
        return tuple(e.step_id for e in self._emitted if e.event == "COMPLETED")

    @property
    def violations(self) -> tuple[str, ...]:
        return self.status().violations

    @property
    def finished(self) -> bool:
        return not self._pending


# --------------------------------------------------------------------------
# Small runtime helpers
# --------------------------------------------------------------------------


class EventRing:
    """Thread-safe recent-event buffer; the GUI's log tail (C9) reads it."""

    def __init__(self, capacity: int = 300) -> None:
        self._lock = threading.Lock()
        self._events: deque[StepEvent] = deque(maxlen=capacity)

    def push(self, event: StepEvent) -> None:
        with self._lock:
            self._events.append(event)

    def tail(self, n: int) -> list[StepEvent]:
        with self._lock:
            if n >= len(self._events):
                return list(self._events)
            return list(self._events)[-n:]


def _open_source(source: str) -> tuple[Any, bool]:
    """Open a camera index or a video file.  Returns (capture, is_camera)."""
    import cv2

    is_camera = source.isdigit()
    capture = cv2.VideoCapture(int(source) if is_camera else source)
    if not capture.isOpened():
        hint = " (replay instead: --source demo/correct.mp4)" if is_camera else ""
        raise SystemExit(f"cannot open video source {source!r}{hint}")
    return capture, is_camera


def _source_props(capture: Any) -> tuple[tuple[int, int], float]:
    import cv2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_size = (width, height) if width > 0 and height > 0 else (640, 480)
    if not 1.0 <= fps <= 240.0:
        fps = FALLBACK_FPS
    return frame_size, fps


def _rewind(capture: Any) -> None:
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="har.app",
        description="SIH26174 — AI Human Activity Recognition for on-board BAS experiments (offline).",
    )
    parser.add_argument("--source", default="0", metavar="0|PATH",
                        help="camera index or video file (default: 0)")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL), metavar="PATH",
                        help="protocol yaml (default: protocols/pts01.yaml)")
    parser.add_argument("--detector", choices=("color", "yolo"), default="color",
                        help="detector backend (default: color)")
    parser.add_argument("--colours", default=str(DEFAULT_COLOURS), metavar="PATH",
                        help="HSV colour ranges (default: config/colours.yaml)")
    parser.add_argument("--out-dir", default="runs/latest", metavar="DIR",
                        help="run artefacts directory (default: runs/latest)")
    parser.add_argument("--headless", action="store_true",
                        help="no GUI, no window; exit at end of source")
    parser.add_argument("--no-voice", action="store_true", help="disable TTS")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                        help="write recordings/run_<ts>.mp4 (default: off)")
    parser.add_argument("--stream-host", default=None, metavar="HOST",
                        help="GUI/MJPEG bind host (default: 0.0.0.0; forces the server on even --headless)")
    parser.add_argument("--stream-port", default=None, type=int, metavar="PORT",
                        help="GUI/MJPEG port (default: 8080)")
    parser.add_argument("--pose-every-n", default=1, type=int, metavar="N",
                        help="run pose on every Nth frame (default: 1)")
    parser.add_argument("--imgsz", default=480, type=int, metavar="N",
                        help="pose inference image size (default: 480)")
    parser.add_argument("--conf", default=0.45, type=float, metavar="F",
                        help="pose detection confidence (default: 0.45)")
    parser.add_argument("--max-frames", default=0, type=int, metavar="N",
                        help="stop after N frames (0 = no limit)")
    parser.add_argument("--loop", action="store_true",
                        help="rewind a file source at EOF and re-validate each pass "
                        "(continuous demo replay; the event log stays append-only)")
    parser.add_argument("--contract", action="store_true",
                        help="print CONTRACT_VERSION and exit 0")
    # Composition-root extras (Track C owns the CLI surface, Appendix A):
    parser.add_argument("--wrists", choices=("auto", "pose", "hsv", "none"), default="auto",
                        help="wrist backend. auto: pose for a camera, hsv for a file "
                        "(the shipped demo/synthetic footage is rendered; use "
                        "--wrists pose for real recordings)")
    parser.add_argument("--weights", default=str(DEFAULT_POSE_WEIGHTS), metavar="PATH",
                        help="pose weights for --wrists pose (default: models/yolo11n-pose.pt)")
    parser.add_argument("--stub", action="store_true",
                        help="replay the A1 fixtures through stub perception + validator "
                        "(no camera, no cv2, no model)")
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE_FIXTURE), metavar="PATH",
                        help="evidence fixture for --stub (default: evidence_correct.json)")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_FIXTURE), metavar="PATH",
                        help="canned events for --stub (default: events_correct.jsonl)")
    return parser


# --------------------------------------------------------------------------
# Builders (lazy heavy imports live here)
# --------------------------------------------------------------------------


def _build_detector(args: argparse.Namespace, spec: Any):
    if args.detector == "yolo":
        return _build_yolo_detector(args, spec)
    from har.perception.color_detector import ColorDetector, load_colour_config

    ranges, options = load_colour_config(args.colours)
    roi = options.get("roi")
    if roi is None:
        # Default the search to the rack work envelope: it keeps the dark
        # background outside the rack from masquerading as the black tray and
        # it is the region the protocol operates in (plan B2).
        zone = spec.zone("rack_roi")
        roi = tuple(zone.box) if zone is not None else None
    return ColorDetector(
        ranges,
        roi=roi,
        median_window=int(options.get("median_window", 5)),
        min_area=int(options.get("min_area", 400)),
    )


def _build_yolo_detector(args: argparse.Namespace, spec: Any):
    """Ultralytics-backed detector — the seam §11 fine-tune will drop into.

    Stock COCO weights have no box/crate/tray/vial classes (§2), so this
    backend is only useful with a fine-tuned model whose class names are the
    protocol labels; it maps model classes 1:1 and filters to the protocol.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "--detector yolo needs ultralytics (pip install ultralytics==8.2.100)"
        ) from exc
    from har.perception.adapters import detections_from_yolo_result

    weights = str(DEFAULT_YOLO_WEIGHTS)
    model = YOLO(weights)
    names = getattr(model, "names", {}) or {}
    wanted = set(spec.objects)
    if not wanted.intersection(str(n) for n in names.values()):
        print(
            f"warning: {Path(weights).name} classes do not include any PTS-01 labels; "
            "--detector yolo will see nothing until a fine-tuned model is supplied (plan §11)",
            file=sys.stderr,
        )

    class _YoloDetector:
        backend = f"yolo({Path(weights).name})"

        def detect(self, frame):
            result = model.predict(frame, conf=args.conf, verbose=False)[0]
            return [d for d in detections_from_yolo_result(result, lambda c: names.get(c, str(c)))
                    if d.label in wanted]

    return _YoloDetector()


def _build_wrists(args: argparse.Namespace, is_camera: bool):
    choice = args.wrists
    if choice == "auto":
        choice = "pose" if is_camera else "hsv"
    if choice == "pose":
        try:
            from har.perception.pose import WristExtractor

            return WristExtractor(
                args.weights,
                conf=args.conf,
                every_n_frames=args.pose_every_n,
                model=_ImgszModel(WristExtractor._load_model(args.weights), args.imgsz),
            ), choice
        except Exception as exc:
            print(
                f"warning: pose unavailable ({exc}); falling back to --wrists none "
                "(manipulation steps will not be observed)",
                file=sys.stderr,
            )
            return NullWrists(), "none"
    if choice == "hsv":
        return HsvHandTracker(), choice
    return NullWrists(), choice


def _start_web_server(streamer, status_provider, log_tail, spec, host: str, port: int):
    """Start the C9 GUI in a daemon thread.  Returns (server, url) or (None, reason)."""
    try:
        from werkzeug.serving import make_server

        from har.ui import web
    except ImportError as exc:
        return None, f"GUI unavailable ({exc}); continuing without it"
    web.bind_protocol(spec)
    app = web.create_app(streamer, status_provider, log_tail)
    try:
        server = make_server(host, port, app, threaded=True)
    except OSError as exc:
        return None, f"cannot bind {host}:{port} ({exc}); continuing without the GUI"
    thread = threading.Thread(target=server.serve_forever, name="har-gui", daemon=True)
    thread.start()
    return server, f"http://{host}:{port}"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.contract:
        print(CONTRACT_VERSION)
        return 0

    from har.protocol.spec import load_protocol

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_iso = _now_iso()

    capture = None
    if args.stub:
        from tools.make_synthetic_video import load_evidence
        from tools.replay_events import load_events

        frames = load_evidence(Path(args.evidence))
        frame_size = tuple(int(v) for v in frames[0].frame_size)
        nominal_fps = frames[0].fps or FALLBACK_FPS
        spec = load_protocol(args.protocol, frame_size)
        perception = StubPerception(frames)
        validator: Any = StubValidator(load_events(Path(args.events)), spec)
        detector_backend = "stub"
        wrist_backend = "stub"
        total_frames = len(frames)
        is_camera = False
    else:
        capture, is_camera = _open_source(args.source)
        frame_size, nominal_fps = _source_props(capture)
        spec = load_protocol(args.protocol, frame_size)
        from har.perception.perception import PerceptionStack

        detector = _build_detector(args, spec)
        wrists, wrist_backend = _build_wrists(args, is_camera)
        interaction_config = rendered_interaction_config() if wrist_backend == "hsv" else None
        perception = PerceptionStack(
            detector,
            wrists,
            spec.objects,
            frame_size,
            interaction_config=interaction_config,
        )
        validator = _make_validator(spec)
        detector_backend = getattr(detector, "backend", args.detector)
        total_frames = 0

    # ---- sinks ------------------------------------------------------
    from har.out.eventlog import JsonlEventLog
    from har.out.speaker import OfflineSpeaker

    log = JsonlEventLog(out_dir / "events.jsonl", out_dir / "events.csv")
    speaker = OfflineSpeaker(enabled=not args.no_voice)
    ring = EventRing()

    def handle_event(event: StepEvent) -> None:
        ring.push(event)
        log.emit(event)
        step = spec.step(event.step_id)
        if event.event == "STARTED" and step is not None and step.voice_prompt:
            speaker.say(step.voice_prompt, priority=0)
        elif event.status == "VIOLATION":
            speaker.say(event.message or f"Violation on step {event.step_index}", priority=1)
        elif event.event == "PROTOCOL_COMPLETE":
            speaker.say(event.message, priority=1)
        print(f"  t={event.t_rel:7.3f}  f={event.frame_index:5d}  {event.event:17s} "
              f"{event.status:11s} {event.step_id}  {event.message}")

    recorder = None
    recording_path = None
    if args.record:
        from har.out.recorder import VideoRecorder

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_path = out_dir / "recordings" / f"run_{ts}.mp4"
        recorder = VideoRecorder(recording_path, frame_size, fps=nominal_fps)

    from har.out.streamer import MjpegStreamer

    stream_host = args.stream_host or "0.0.0.0"
    stream_port = int(args.stream_port) if args.stream_port is not None else 8080
    streamer = MjpegStreamer(stream_host, stream_port)
    server = None
    server_note = "disabled"
    want_server = (not args.headless) or args.stream_host is not None or args.stream_port is not None
    if want_server:
        server, note = _start_web_server(streamer, validator.status, ring.tail, spec,
                                         stream_host, stream_port)
        server_note = note

    voice_note = "off (--no-voice)"
    if speaker.enabled:
        if speaker.wait_ready(timeout=3.0):
            voice_note = "on"
        else:
            voice_note = f"UNAVAILABLE ({speaker.init_error}); alerts stay on the banner — " \
                         "install espeak-ng or pass --no-voice"
            print(f"warning: TTS initialisation failed: {speaker.init_error}", file=sys.stderr)

    print(f"SIH26174 HAR — contract {CONTRACT_VERSION}")
    print(f"  source    : {'stub fixture' if args.stub else args.source}{' (camera)' if capture and is_camera else ''}")
    print(f"  protocol  : {spec.protocol_id} v{spec.version} ({len(spec.steps)} steps)")
    print(f"  detector  : {detector_backend}   wrists: {wrist_backend}")
    print(f"  out dir   : {out_dir}")
    print(f"  voice     : {voice_note}")
    print(f"  record    : {recording_path or 'off'}")
    print(f"  gui/stream: {server_note}")

    # ---- frame loop -------------------------------------------------
    frame_count = 0
    pass_number = 0
    t0 = time.monotonic()
    exit_reason = "end of source"
    frame_limit = args.max_frames or (total_frames if args.stub else 0)
    try:
        while True:
            if frame_limit and frame_count >= frame_limit:
                exit_reason = (f"--max-frames {args.max_frames}" if args.max_frames
                               else "stub fixture replayed")
                break
            if capture is not None:
                ok, frame = capture.read()
                if not ok:
                    if is_camera:
                        print("warning: dropped camera frame; retrying", file=sys.stderr)
                        time.sleep(0.01)
                        continue
                    if args.loop and not args.stub:
                        # New pass over the same footage: rewind and start a
                        # fresh validation run (log stays append-only).
                        _rewind(capture)
                        perception.reset()
                        validator.reset()
                        pass_number += 1
                        print(f"--- pass {pass_number + 1}: source rewound, run reset ---")
                        continue
                    break
                t_rel = time.monotonic() - t0 if is_camera else frame_count / nominal_fps
            else:
                frame = None
                t_rel = 0.0  # the stub evidence rows carry their own t_rel
            evidence = perception.process(frame, frame_count, t_rel)
            events = validator.update(evidence)
            for event in events:
                handle_event(event)
            if frame is not None:
                try:
                    from har.ui.overlay import draw_hud

                    draw_hud(frame, validator.status(), evidence)
                except ImportError:
                    if recorder is None and streamer is not None and frame_count == 0:
                        print("warning: overlay unavailable (no cv2); streams/recordings carry raw frames",
                              file=sys.stderr)
                except Exception as exc:  # never let cosmetics kill the run
                    if frame_count == 0:
                        print(f"warning: overlay failed ({exc}); continuing without the HUD",
                              file=sys.stderr)
                if recorder is not None:
                    recorder.write(frame)
                streamer.publish(frame)
            frame_count += 1
    except KeyboardInterrupt:
        exit_reason = "interrupted"
    if args.stub and hasattr(validator, "drain"):
        for event in validator.drain():
            handle_event(event)
    print(f"\nrun finished: {exit_reason}; {frame_count} frames")

    # ---- shutdown ---------------------------------------------------
    if server is not None:
        server.shutdown()
    streamer.shutdown()
    if recorder is not None:
        recorder.close()
        print(f"recording: {recording_path}")
    speaker.stop()
    log.close()
    if capture is not None:
        capture.release()

    summary = _summarise(out_dir)
    meta = {
        "source": "stub" if args.stub else args.source,
        "protocol_id": spec.protocol_id,
        "protocol_version": spec.version,
        "contract_version": CONTRACT_VERSION,
        "detector_backend": detector_backend,
        "wrist_backend": wrist_backend,
        "fps": nominal_fps,
        "frame_count": frame_count,
        "started_at": started_iso,
        "ended_at": _now_iso(),
        "exit_reason": exit_reason,
        "headless": bool(args.headless),
        "voice": "off" if not speaker.enabled else ("on" if speaker.available else "unavailable"),
        "recording": str(recording_path) if recording_path else None,
        "gui": server_note if server is not None else None,
        "events": summary,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"artefacts: {out_dir}/events.jsonl, events.csv, meta.json"
          + (f", {recording_path}" if recording_path else ""))
    return 0


def _make_validator(spec: Any):
    from har.protocol.validator import SequenceValidator

    return SequenceValidator(spec)


def _summarise(out_dir: Path) -> dict[str, int]:
    """Event-kind tally of the run's log, for meta.json and the console."""
    counts: dict[str, int] = {}
    try:
        from tools.replay_events import iter_events

        for event in iter_events(out_dir / "events.jsonl"):
            counts[event.event] = counts.get(event.event, 0) + 1
    except Exception:
        pass
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
