#!/usr/bin/env bash
# Container health probe.
#
# Two shapes of healthy run exist, and a probe that only understood one would
# report a false failure for the other:
#
#   * GUI/stream run — GET /status must answer 200.  That endpoint renders the
#     live UiStatus through har.ui.web, so a 200 proves the frame loop, the
#     validator and the HTTP layer are all still talking.
#   * Headless run — no server is listening at all, so health falls back to
#     "the CLI process is still alive" (scanned out of /proc: python:slim has
#     no pgrep).  A frame loop that died on a decode error is caught here.
#
# Exit codes are the contract Docker uses: 0 healthy, 1 unhealthy.
set -uo pipefail

exec python - "${HAR_STREAM_PORT:-8080}" <<'PY'
import glob
import os
import socket
import sys
import urllib.request

port = int(sys.argv[1])


def listening() -> bool:
    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def cli_alive() -> bool:
    me = os.getpid()
    for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
        pid = int(cmdline.split("/")[2])
        if pid == me:
            continue
        try:
            with open(cmdline, "rb") as handle:
                parts = handle.read().split(b"\0")
        except OSError:
            continue
        if b"har.app" in parts:
            return True
    return False


if listening():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as response:
            body = response.read()
    except Exception as exc:  # noqa: BLE001 - any failure means "not healthy"
        print(f"health: /status failed: {exc}", file=sys.stderr)
        sys.exit(1)
    if response.status != 200:
        print(f"health: /status returned {response.status}", file=sys.stderr)
        sys.exit(1)
    if not body.startswith(b"{"):
        print("health: /status did not return a JSON object", file=sys.stderr)
        sys.exit(1)
    print("health: gui /status 200")
    sys.exit(0)

if cli_alive():
    print(f"health: headless run alive (nothing listening on {port})")
    sys.exit(0)

print("health: no har.app process and nothing listening "
      f"on {port}", file=sys.stderr)
sys.exit(1)
PY
