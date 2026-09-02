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

## Phase 3 / optional (B9)

ONNX export (`model.export(format="onnx")`) plus a CPU-vs-ONNX latency row
belongs here. Requires `onnxruntime` + `onnx` (commented out in
`requirements.txt`). First thing to cut if time runs short — it is optional.
