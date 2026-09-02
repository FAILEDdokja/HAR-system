#!/usr/bin/env bash
# SIH26174 container entrypoint.
#
# Three jobs, in order:
#
#   1. Point --out-dir at the artefact volume.  The CLI's own default is
#      ./runs/latest, which inside a container is writable, invisible from the
#      host and deleted with the container — the worst of all three worlds for
#      the event log that is deliverable D4.
#   2. Fail fast, with the fix printed, on the two mistakes a container makes
#      that a bare `python -m har.app` does not: a bind-mounted artefact
#      directory owned by the host user, and a camera index with no device
#      passed through (the CLI would otherwise spin forever printing
#      "dropped camera frame; retrying").
#   3. exec the CLI so it is PID 1 and receives `docker stop`'s SIGTERM
#      directly — har.app turns that into the same clean shutdown as Ctrl-C,
#      which is what finalises the recording and closes the event log.
#
# Environment (see .env.example):
#   HAR_OUT_DIR      artefact directory          (default /data/latest)
#   HAR_STREAM_PORT  GUI/MJPEG port to publish   (default 8080)
#   HAR_EXTRA_ARGS   extra CLI flags, word-split (default empty)
set -euo pipefail

OUT_DIR="${HAR_OUT_DIR:-/data/latest}"
STREAM_PORT="${HAR_STREAM_PORT:-8080}"

args=()
source_value="0"          # the CLI's own default is camera index 0
have_out_dir=0
have_stream_port=0
headless=0
previous=""

for arg in "$@"; do
    args+=("$arg")
    case "$arg" in
        --source=*) source_value="${arg#--source=}" ;;
        --out-dir|--out-dir=*) have_out_dir=1 ;;
        --stream-port|--stream-port=*) have_stream_port=1 ;;
        --headless) headless=1 ;;
    esac
    if [[ "$previous" == "--source" ]]; then
        source_value="$arg"
    fi
    previous="$arg"
done

# Escape hatch for flags this script knows nothing about.  Word-splitting is
# the point: HAR_EXTRA_ARGS="--imgsz 320 --conf 0.6".
if [[ -n "${HAR_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra=(${HAR_EXTRA_ARGS})
    args+=("${extra[@]}")
fi

# --- camera preflight ------------------------------------------------------
if [[ "$source_value" =~ ^[0-9]+$ ]] && ! compgen -G "/dev/video*" > /dev/null; then
    cat >&2 <<EOF
har: --source ${source_value} is a camera index, but no /dev/video* device is
har: visible inside this container.  Pass one through:
har:
har:   docker run --rm -p 8080:8080 --device /dev/video0:/dev/video0 sih26174-har
har:   docker compose --profile camera up          # see docker-compose.yml
har:
har: or replay a file instead: --source demo/correct.mp4
EOF
    exit 1
fi

# --- artefact directory ----------------------------------------------------
if [[ "$have_out_dir" == 0 ]]; then
    probe="$OUT_DIR"
    while [[ ! -d "$probe" && "$probe" != "/" ]]; do
        probe="$(dirname "$probe")"
    done
    if [[ ! -w "$probe" ]]; then
        cat >&2 <<EOF
har: ${OUT_DIR} is not writable by $(id -un) (uid $(id -u)).
har: A bind-mounted host directory keeps the host's ownership, so either
har:
har:   chown 1001:1001 <host directory>          # the image's uid, or
har:   set HAR_UID/HAR_GID in .env to your ids   # docker-compose.yml uses them
har:
har: or point HAR_OUT_DIR somewhere writable.
EOF
        exit 1
    fi
    mkdir -p "$OUT_DIR"
    args+=(--out-dir "$OUT_DIR")
fi

# --- GUI / MJPEG port ------------------------------------------------------
# --stream-port also switches the server on (har.app treats an explicit port
# as "serve even when --headless"), so it is only injected when a GUI is
# wanted; a headless one-shot run stays headless.
if [[ "$headless" == 0 && "$have_stream_port" == 0 ]]; then
    args+=(--stream-port "$STREAM_PORT")
fi

echo "har: python -m har.app ${args[*]}"
exec python -u -m har.app ${args[@]+"${args[@]}"}
