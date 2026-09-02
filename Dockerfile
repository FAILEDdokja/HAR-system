# syntax=docker/dockerfile:1
#
# SIH26174 — AI Human Activity Recognition for on-board BAS experiments.
#
#   docker build -t sih26174-har .                     # CPU image (default)
#   docker build --build-arg TORCH_CHANNEL=cuda \
#                --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
#                -t sih26174-har:cuda .
#
# Dependency resolution never runs at build time.  requirements.lock is a
# complete frozen set, so both install steps use --no-deps and the image gets
# exactly the pinned versions — no resolver, no surprise upgrades, no network
# beyond the two indexes named below.  That is also what makes the headless
# OpenCV substitution in tools/docker_requirements.py stick: ultralytics
# 8.2.100 declares "opencv-python>=4.6.0", so a normal resolve would pull the
# GUI build — and with it libGL — straight back in.

ARG PYTHON_VERSION=3.11
ARG TORCH_CHANNEL=cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# ----------------------------------------------------------------- builder --
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG TORCH_CHANNEL
ARG TORCH_INDEX_URL

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m venv /opt/venv

WORKDIR /build

# Only the inputs to dependency resolution, so editing application code never
# invalidates the multi-gigabyte install layer.  Paths are preserved (not
# flattened) because the RUN below addresses them the way the repo does.
COPY requirements.lock ./
COPY tools/docker_requirements.py tools/
COPY docker/check_torch.py docker/

RUN set -eux; \
    python tools/docker_requirements.py --check --channel "${TORCH_CHANNEL}" requirements.lock; \
    python tools/docker_requirements.py --channel "${TORCH_CHANNEL}" --torch-only requirements.lock > /tmp/torch.txt; \
    python tools/docker_requirements.py --channel "${TORCH_CHANNEL}" requirements.lock > /tmp/requirements.txt; \
    pip install --no-deps --index-url "${TORCH_INDEX_URL}" -r /tmp/torch.txt; \
    pip install --no-deps -r /tmp/requirements.txt; \
    python docker/check_torch.py "${TORCH_CHANNEL}"

# ----------------------------------------------------------------- runtime --
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# espeak-ng is pyttsx3's offline Linux TTS driver (deliverable D3).  Without an
# audio device the speaker disables itself and alerts stay on the GUI banner,
# so this is the only OS package the image needs: headless OpenCV has no libGL
# dependency and torch's manylinux wheel bundles libgomp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="SIH26174 HAR" \
      org.opencontainers.image.description="Offline human activity recognition for on-board BAS experiments: protocol tracking, voice alerts, timestamped event log, MJPEG stream and local recording." \
      org.opencontainers.image.source="https://github.com/KrSikchi/HAR-system" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HAR_OUT_DIR=/data/latest \
    HAR_STREAM_PORT=8080

# Run unprivileged; /data is the artefact volume (events.jsonl, events.csv,
# meta.json, recordings/).  docker-compose.yml maps HAR_UID/HAR_GID onto this
# account so bind mounts stay writable.
RUN useradd --create-home --uid 1001 --shell /usr/sbin/nologin har \
    && mkdir -p /data \
    && chown -R har:har /data

WORKDIR /app

COPY --from=builder --chown=har:har /opt/venv /opt/venv
COPY --chown=har:har . /app

USER har

# GUI + MJPEG stream (har/ui/web.py binds 0.0.0.0 by design).
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD ["/app/docker/healthcheck.sh"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default: the demo that cannot fail — replay the shipped footage in a loop,
# record it, and serve the monitoring GUI on 8080.  Override with any
# `python -m har.app` arguments, e.g. `--source 0` for a live camera.
CMD ["--source", "demo/correct.mp4", "--loop", "--record"]
