# PERF — perception timing budget (Person B)

Owner: Person B. This is **not** `docs/METRICS.md` (that is A's file).
Everything downstream — frame loop pacing, `--pose-every-n` defaults, the
streamer's latest-frame-only policy — is sized from the numbers here.

> **Status: probe tool landed and local synthetic baseline measured.**
> Re-run this on the actual demo laptop with `--source 0` before rehearsal,
> because live camera drivers and CPU governor settings can move these numbers.
>
> ```bash
> .venv/bin/python -m tools.probe_fps --frames 200
> ```

## Method

`tools/probe_fps.py` times end-to-end `model.predict()` calls after a 10-frame
warm-up, reporting ms/frame and FPS, for:

1. **pose only** — `yolo11n-pose.pt` alone. This is the shipped design: §2 of
   the plan drops `yolo11n.pt` because the pose head already detects the
   `person` class, so one inference pass serves both wrist extraction and the
   operator-in-frame gate.
2. **pose + yolo11n (legacy)** — the two-model plan we rejected, kept as a
   baseline to quantify the win.

Sources: synthetic frames by default, or `--source 0` / a video file.

## Results

Machine: Windows, Intel64 Family 6 Model 186 Stepping 2, 12 logical CPUs,
15.6 GiB RAM, CPU-only torch 2.14.0, date: 2026-09-02, ultralytics 8.4.138.
Source: synthetic 640x480 BGR frames, 10 warm-up frames, 50 measured frames.

| Configuration | imgsz | ms / frame | FPS | measured frames |
|---|---|---|---|---|
| pose only (person gate, 1 model) | 480 | 54.6 | 18.3 | 50 |
| pose only (person gate, 1 model) | 640 | 54.2 | 18.4 | 50 |
| pose + yolo11n (legacy, 2 models) | 480 | 127.9 | 7.8 | 50 |
| pose + yolo11n (legacy, 2 models) | 640 | 140.2 | 7.1 | 50 |

### Re-probe: Linux sandbox (2026-09-03)

Machine: Linux, Intel Xeon @ 2.60 GHz (CPU-only torch 2.14.0),
ultralytics 8.4.138. Source: synthetic 640x480 BGR frames, 10 warm-up
frames, 50 measured frames (`python -m tools.probe_fps --frames 50`).

| Configuration | imgsz | ms / frame | FPS | measured frames |
|---|---|---|---|---|
| pose only (person gate, 1 model) | 480 | 64.7 | 15.4 | 50 |
| pose only (person gate, 1 model) | 640 | 100.3 | 10.0 | 50 |
| pose + yolo11n (legacy, 2 models) | 480 | 116.2 | 8.6 | 50 |
| pose + yolo11n (legacy, 2 models) | 640 | 177.6 | 5.6 | 50 |

Same shape as the Windows numbers: the single-pass design roughly doubles
the frame rate, and `imgsz=480` clears the 12 FPS floor on this weaker CPU
while 640 does not — which is exactly why `imgsz=480` is the first B6 lever.

## B9 — ONNX export and latency (done 2026-09-03)

`YOLO("models/yolo11n-pose.pt").export(format="onnx", imgsz=480, opset=17)`
succeeds (11.2 MB, `models/yolo11n-pose.onnx`; the file stays untracked —
`.gitignore` excludes `*.onnx`). End-to-end `model.predict()` on the same
Linux sandbox, 10 warm-up + 50 measured synthetic 640x480 frames:

| Backend | imgsz | ms / frame | FPS |
|---|---|---|---|
| `.pt` (torch CPU) | 480 | 61.1 | 16.4 |
| `.onnx` (ONNX Runtime CPU) | 480 | 54.7 | 18.3 |

~10 % latency win on CPU. Nice-to-have, not needed to clear the 12 FPS
floor; the `.pt` path stays the default.

## Demo-Laptop Retune Checklist

Run before the final demo:

```bash
.venv/bin/python -m tools.probe_fps --source 0 --frames 200
```

Acceptance for B6:

* `pose only` at `imgsz=480` sustains at least 12 FPS from the live camera.
* `config/colours.yaml` has a rack ROI filled in after the camera is fixed.
* All five labels (`tray`, `tray_lid`, `red_box`, `blue_box`, `vial`) stay
  stable for 60 seconds with the colour-key card in frame.

## What the budget must satisfy (B6)

* ≥ 12 FPS sustained on the demo hardware with all five labels stable for
  60 seconds. Levers, in order: `imgsz=480`, `conf=0.45`, pose every 2nd
  frame (`--pose-every-n 2`).
* The colour detector is *not* on the critical path of this table: HSV
  segmentation + largest-blob labelling costs ~1–2 ms at 640×480 with cv2
  (it runs every frame, pose does not). If the pose numbers above cannot
  reach 12 FPS, `--pose-every-n 2` is the first lever, not the detector.

## Person-gate note

When `PerceptionStack` sees no person (from the pose pass), it skips the
detector and hands the trackers an empty list — they coast and the interaction
FSMs hold. On frames where the operator steps out of frame this roughly
removes all detector + tracking cost; the gate is pass-through for extractor
implementations that do not report a person count.

## B5 / Gate G1 — perception-vs-validator cross-check (done 2026-09-03)

Person C's `tools/make_synthetic_video.py` (C4) never landed, so the gate ran
on the committed `demo/*.mp4` stand-ins instead (same PTS-01 layout the
colour detector is designed for). Full chain per frame:
`mp4 → ColorDetector → PerceptionStack → FrameEvidence → SequenceValidator`.
Wrists come from an HSV stand-in for the rendered hand rings (the contracts
are duck-typed for exactly this swap; `WristExtractor` slots in unchanged on
live footage).

```
$ .venv/bin/python -m tools.crosscheck_g1
correct      PASS  8 COMPLETED, complete=True, violations=[]
skip         PASS  violations=[('OUT_OF_ORDER', 'EXTRACT_BLUE')], complete=False
wrong_order  PASS  violations=[('OUT_OF_ORDER', 'EXTRACT_BLUE')], complete=False
GATE G1 (B5 cross-check): PASS
```

Fixes the cross-check forced (this is why B5 exists):

* `settled()` now refuses an object whose HOI state is still
  `PICKED_UP`/`CARRYING` — on live perception the FSM lags a few frames
  after a release, and without the guard VERIFY_* raised a false
  `OUT_OF_ORDER` on a perfectly correct run (pinned in
  `tests/test_predicates.py`).
* `demo/build_dataset.py` renders the lid inset on the tray and hands as
  rings, not filled discs — the old rendering pixel-occluded the tray and
  the carried vial, contradicting its own evidence recording.
* `config/colours.yaml` `min_area` 400 → 300: the rendered vial is a
  20×20 px blob (~361 px² contour area).

## B8 — rotation demo (done 2026-09-03)

`tools/rotation_demo.py` rotates the whole rig 90° clockwise mid-run
(`cv2.rotate` on the frame; fiducials re-homed via B7's `RackFrame`
homography; detector ROI re-pointed). Trackers, FSMs, zones and predicates
all operate in rack space, so the flip is invisible downstream:

```
$ .venv/bin/python -m tools.rotation_demo
rotation at t=7.0s: 8 COMPLETED, PROTOCOL_COMPLETE=True, violations=[]
B8 ROTATION DEMO: PASS
```

Also passes with the rotation at t = 2.5, 3.5, 5.0, 9.0, 11.0 and 13.0 s —
including mid-carry instants where the tracked object is in the operator's
hand at the moment every camera pixel moves.
