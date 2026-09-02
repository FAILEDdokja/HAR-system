# A8 — PTS-01 evaluation dataset

This directory is the custom dataset the problem statement asks for.
It is used for **evaluation** (step A9), never for training. There is no
labelling of bounding boxes and no fine-tune.

| File | What it is |
|---|---|
| `correct.mp4` | Scripted 15 fps / 640×480 run of all 8 steps in order |
| `skip.mp4` | Same layout; the red box is never moved |
| `wrong_order.mp4` | Blue box placed before the red box |
| `ground_truth.json` | Hand-annotated true start/end of every step |
| `*_evidence.json` | Frame-accurate `FrameEvidence` for the same timeline |
| `build_dataset.py` | Regenerator (Track A; OpenCV only required to re-render) |

## Why these are not webcam captures

A8's plan text says "use C7's recorder to capture three real runs". This
host has no camera and `har/out/recorder.py` is not on the branch yet, so
the three mp4 files are **scripted stand-ins**: coloured props (black tray,
yellow lid, red box, blue box, green vial) moving through the PTS-01
zones. They are real video files — a stock player and
`cv2.VideoCapture` both open them — and they are the `--source`
fallback if the venue webcam fails (`docs/DEVELOPMENT_PLAN.md` §9).

Rebuild:

```bash
.venv/bin/python demo/build_dataset.py --render
```

## Ground-truth schema

`runs.<id>.steps[]` carries the human annotation:

* `t_start` / `t_end` — operator-true times in seconds (`t_rel`)
* `operator_outcome` — `COMPLETED` / `SKIPPED` / `NOT_PERFORMED`
* `validator_*` — what `SequenceValidator` emitted on the recording
  (A9 diffs these against the annotation to get latency and
  false-alarm rate)

`runs.<id>.expected.events` is the full event log a replay of
`*_evidence.json` must reproduce. For the correct run that log is also
what `--source demo/correct.mp4` must write once Track C's CLI lands.

## Layout (pixels at 640×480)

Matches `protocols/pts01.yaml` (normalised boxes × frame size):

* rack envelope `[51, 72, 589, 456]`
* tray slot `[218, 230, 422, 422]`
* zone A (red) `[64, 240, 192, 384]`
* zone B (blue) `[448, 240, 576, 384]`
* rack slot `[282, 86, 358, 163]`
