# Deploying with Docker

The container packages the whole offline system — CLI, perception stack,
protocol, browser GUI, MJPEG stream, recorder and voice — behind one image and
one published port.  It is the recommended way to run SIH26174 anywhere except
a bare demo laptop, and it is the only path that survives "the venue machine
has a different Python".

```
docker build -t sih26174-har .          # CPU image, on the order of 2.5 GB
docker compose up -d                    # GUI + stream on http://localhost:8080
```

---

## What is in the image

| | |
|---|---|
| Base | `python:3.11-slim-bookworm`, multi-stage: the builder's virtualenv is the only thing copied forward |
| Python | `requirements.lock`, installed with `--no-deps` (see below) |
| OS packages | `espeak-ng` only — pyttsx3's offline Linux TTS driver |
| Code | `har/`, `protocols/`, `config/`, `models/` (12 MB of pretrained weights), `demo/` footage, `tests/` — 79 files, 17 MB of build context |
| User | `har`, uid 1001, no shell; `/data` is the artefact volume |
| Port | 8080 (monitoring GUI + MJPEG stream) |
| Entrypoint | `docker/entrypoint.sh` → `python -m har.app` |

No `libgl1`, no `libglib2.0-0`, no `libgomp1`.  The image installs
`opencv-python-headless` at the same pinned version as the lock's
`opencv-python` — nothing in `har/` or `tools/` opens a window, so the GUI
build's X11 dependencies are dead weight — and torch's manylinux wheel bundles
its own `libgomp.so.1` (`ldd torch/lib/libtorch_global_deps.so` resolves it
from inside the wheel).

### Why the build never runs a resolver

`requirements.lock` is a complete frozen set, so both install steps use
`--no-deps`: the image gets exactly the pinned versions, no resolver surprises,
and the headless-OpenCV substitution cannot be undone.  That last point is not
cosmetic — `ultralytics` declares `opencv-python>=4.6.0`, so a normal resolve
would pull the GUI build, and with it `libGL`, straight back in.

The set is derived from the lock at build time by
`tools/docker_requirements.py`, so there is one source of truth:

```bash
python tools/docker_requirements.py --check --channel cpu requirements.lock   # validate
python tools/docker_requirements.py --channel cpu requirements.lock           # print the set
```

`--channel cpu` (default) drops `torch`/`torchvision` — installed from the CPU
wheel index instead — plus the `nvidia-*`, `cuda-*` and `triton` wheels the
lock pins for Linux.  That is 3.3 GB of `nvidia/*` plus 0.9 GB of `triton`
installed, none of which a CPU box can use.  `--channel cuda` keeps them.

`docker/check_torch.py` runs during the build and fails it if the resolved
torch does not match the channel, which is what stops a stray
`--extra-index-url` from silently shipping the wrong 3 GB.

---

## Run it

### docker compose

```bash
cp .env.example .env                  # optional: every value has a default
docker compose up -d                  # demo replay + GUI on :8080
docker compose logs -f
docker compose down
```

The default service replays `demo/correct.mp4` in a loop with `--record`, which
is the demo that cannot fail: no camera, no GPU, no network.  Artefacts land in
`/data/latest` inside the container, which compose backs with the `har-runs`
named volume (see *Artefacts, volumes and permissions*) — there is no `./runs`
directory in your checkout, and on a Windows Docker Desktop host the named
volume is the safe default because it has no bind-mount ownership or
path-sharing requirements.

### docker run

```bash
docker run --rm -p 8080:8080 -v "$PWD/runs:/data" sih26174-har
```

`-v "$PWD/runs:/data"` is a Linux bind mount; `-v har-runs:/data` uses a Docker
named volume instead, which also works unchanged on Windows Docker Desktop.

Any arguments go straight to the CLI.  Anything whose first word is *not* a
flag is treated as a command to run instead of CLI arguments — the entrypoint
execs it directly, the way the postgres and mysql images do:

```bash
# replay a specific file, headless, no voice
docker run --rm -v "$PWD/runs:/data" sih26174-har \
    --source demo/wrong_order.mp4 --headless --no-voice

# one-shot gate G1 inside the image
docker run --rm -v "$PWD/runs:/data" sih26174-har \
    --source tests/fixtures/synthetic_correct.mp4 --headless --no-voice
grep PROTOCOL_COMPLETE runs/latest/events.jsonl

# the test suite, in the image (a command override, not CLI flags)
docker run --rm sih26174-har python -m unittest discover
```

### Live camera

```bash
docker run --rm -p 8080:8080 -v "$PWD/runs:/data" \
    --device /dev/video0:/dev/video0 sih26174-har --source 0 --record

# or, via compose:
docker compose --profile camera up -d har-camera
```

The entrypoint refuses to start with a camera index and no `/dev/video*`
device, and prints the two ways to fix it — otherwise the CLI would spin
forever printing `dropped camera frame; retrying`.  If the device is present
but permission is denied, run the container as a host user in the `video`
group (`HAR_UID`/`HAR_GID` in `.env`).

### GPU

```bash
docker compose -f docker-compose.yml -f compose.gpu.yml build
docker compose -f docker-compose.yml -f compose.gpu.yml up -d
```

Needs the NVIDIA driver and the NVIDIA Container Toolkit on the host.  The
override switches the build to `--build-arg TORCH_CHANNEL=cuda` with the
`cu130` wheel index, which matches the lock's `nvidia-*-cu13` pins.

---

## Artefacts, volumes and permissions

Everything a run produces goes to `--out-dir`, which the entrypoint points at
`$HAR_OUT_DIR` (default `/data/latest`) unless you pass your own:

```
/data/<run>/events.jsonl        one JSON object per step event, fsynced per event
/data/<run>/events.csv          the same rows, spreadsheet-shaped
/data/<run>/meta.json           run metadata + an event tally
/data/<run>/recordings/run_<ts>.mp4
```

`/data` is a **named volume** by default.  `docker-compose.yml` declares
`har-runs` and mounts it at `/data` via `HAR_DATA_DIR=har-runs`; Docker creates
and owns the volume, seeding a brand-new one from the image so `/data` starts
out owned by `har` (uid 1001).  It survives `docker compose down`, shows up in
`docker volume ls`, and is removed by `docker compose down -v`.  Run artefacts
land in `/data/<run>/` inside the volume, not in a `./runs` directory in your
checkout.

To keep artefacts as ordinary files on the host, opt back into a bind mount by
setting `HAR_DATA_DIR` to a path (relative paths need a leading `./`):

```bash
echo 'HAR_DATA_DIR=./runs' >> .env    # bind mount ./runs:/data
docker compose down && docker compose up -d
```

A bind mount keeps the *host's* ownership, so if the container cannot write to
it the entrypoint stops immediately and tells you which fix to apply:

```bash
chown 1001:1001 ./runs              # the image's uid, or
echo 'HAR_UID=1000' >> .env         # your own ids (id -u / id -g)
echo 'HAR_GID=1000' >> .env
```

### Windows Docker Desktop

On a Windows host, leave `HAR_DATA_DIR` at its `har-runs` default: a named
volume needs no Windows path, no sharing configuration and no ownership fixup,
and it is much faster than a bind mount across the WSL2 boundary.  If you must
use a bind mount of a Windows directory:

1. Create the directory from Windows first (e.g. `mkdir C:\har\runs`); a path
   that only exists inside the WSL2 VM will fail to mount.
2. Share it with the VM: Docker Desktop **Settings → Resources → File sharing**,
   add the drive (e.g. `C:\har`) and restart Docker Desktop if prompted.
3. Set `HAR_DATA_DIR` to the path as Docker presents it — `C:\har\runs` from
   PowerShell, or `/c/har/runs` from Git Bash — and set `HAR_UID`/`HAR_GID` to
   whatever owns that directory, because the host's ownership is preserved.

---

## Health

```bash
docker inspect --format '{{.State.Health.Status}}' <container>
docker compose ps                   # HEALTH column
```

`docker/healthcheck.sh` understands both shapes of healthy run: when the GUI is
up it requires `GET /status` to return 200 (that route renders the live
`UiStatus`, so a 200 proves the frame loop, the validator and the HTTP layer
are all still talking); for a `--headless` run, where nothing is listening, it
falls back to "the `har.app` process is still alive", scanned out of `/proc`
because `python:slim` has no `pgrep`.

---

## Stopping a run

`docker compose stop` / `docker stop` send SIGTERM.  `har.app` converts it into
the same clean shutdown path as Ctrl-C, which is what finalises the mp4 trailer
and closes the event log — without it a stopped container leaves an unplayable
recording and a truncated `events.jsonl`.  `meta.json` records
`"exit_reason": "interrupted"`.  `stop_grace_period: 30s` in the compose file
gives the recorder time before compose escalates to SIGKILL.

---

## Air-gapped deployment

The venue may have no network, and the image must not need one at runtime: the
weights, protocol, config and demo footage are all baked in, and nothing in the
frame loop makes a network call.

Build on a networked machine, ship the image, load it on the disconnected one:

```bash
# networked build box
docker build -t sih26174-har:cpu .
docker save sih26174-har:cpu | gzip > sih26174-har-cpu.tar.gz     # ~the image size

# venue machine, cable unplugged
gunzip -c sih26174-har-cpu.tar.gz | docker load
docker run --rm -p 8080:8080 -v "$PWD/runs:/data" sih26174-har:cpu
```

The size figures in this document are estimates from the installed payload
(`du` over the resolved site-packages, minus the CUDA wheels the cpu channel
drops), not `docker image ls` output — run that on your build box for the real
number.

`docker save`/`load` replaces `wheelhouse/` for container deployments; the
wheelhouse remains the offline path for a bare-metal install
(`wheelhouse/README.md`).

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TORCH_CHANNEL` | `cpu` | build arg: `cpu` or `cuda` |
| `TORCH_INDEX_URL` | `.../whl/cpu` | build arg: the torch wheel index |
| `HAR_HOST_PORT` | `8080` | host port for the GUI/stream |
| `HAR_DATA_DIR` | `har-runs` | Docker volume name (or a `./`- or `/`-prefixed host path for a bind mount) mounted at `/data` |
| `HAR_RUN_NAME` | `latest` | run directory under `/data` |
| `HAR_UID` / `HAR_GID` | `1001` | container uid:gid; only matters for bind mounts |
| `HAR_EXTRA_ARGS` | empty | extra CLI flags, word-split |
| `HAR_OUT_DIR` | `/data/latest` | in-container artefact directory |
| `HAR_STREAM_PORT` | `8080` | port the GUI binds and the healthcheck probes |

The entrypoint adds `--out-dir` and `--stream-port` only when you have not
passed them, and never adds `--stream-port` to a `--headless` run.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `is not writable by har (uid 1001)` | Bind-mount ownership — only reachable with `HAR_DATA_DIR=./…`; the `har-runs` named volume never hits this.  See *Artefacts, volumes and permissions*. |
| `no /dev/video* device is visible` | Camera not passed through.  `--device /dev/video0:/dev/video0` or `--profile camera`. |
| Voice silent, banner works | Expected without an audio device.  Pass an ALSA/Pulse device into the container, or add `--no-voice` to silence the warning. |
| `Address already in use` on 8080 | Another run holds the port.  Set `HAR_HOST_PORT`, or stop the other container. |
| Unhealthy but logs look fine | A `--headless` run has no HTTP server; the healthcheck then checks the process.  If it reports unhealthy, the frame loop died — read `docker compose logs`. |
| `/data` grows without bound | `--loop` is append-only by design.  Measured on the default demo command: the recording grows **25 MB/min** (640x480@15fps, mp4v) and `events.jsonl` about 0.06 MB/min — an hour of demo is ~1.5 GB.  Rotate `/data` yourself (`docker volume rm har-runs`, or delete the bind-mount directory), or drop `--loop --record` for a one-shot run. |

## Known gaps

* **Voice needs an audio device.**  `espeak-ng` is installed, but a container
  has no sound card by default; alerts fall back to the on-screen banner
  (deliverable D3's documented degradation path).
* **The GUI is a monitoring surface, not a hardened one.**  It is a Werkzeug
  dev server bound to `0.0.0.0` by design, for a LAN demo.  Do not expose 8080
  to the public internet; put a reverse proxy with auth in front if you must.
* **Single container, single camera.**  There is no orchestration story here
  beyond compose; one camera per container, one container per experiment.
