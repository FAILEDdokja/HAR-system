"""Browser GUI (C9) — deliverable D6, and the visible half of D5's stream.

One Flask application factory, exactly per Appendix A::

    create_app(streamer, status_provider, log_tail) -> flask.Flask
    Routes: GET /   GET /stream   GET /status   GET /events?n=20

The page (``index.html`` — everything local, no CDN, no network: the system
must run in airplane mode) shows the live MJPEG video, the eight-step
checklist with the current step highlighted, a red violation banner, the
spoken next instruction, the live log tail and the FPS readout; it polls
``/status`` at 2 Hz and ``/events`` at 1 Hz.

Where the page gets the step list
---------------------------------
``UiStatus`` carries the *run* state (current/next/completed/skipped/
violations) but not the protocol's static step table, and
``create_app``'s signature is frozen — so the entrypoint calls
``bind_protocol(spec)`` (a Track-C-side, additive hook; no frozen signature
changes) before ``create_app``.  A bound spec is served as ``GET /protocol``
and the checklist renders the full table with real instructions; without it
the page degrades to a status-only view.  Nothing here imports perception or
protocol modules — the data is passed in, per the ownership rules.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from har.contracts import StepEvent, UiStatus

__all__ = ["create_app", "bind_protocol", "bind_reset"]

_INDEX = Path(__file__).with_name("index.html")

_protocol_lock = threading.Lock()
_protocol_spec: Any = None

_reset_lock = threading.Lock()
_reset_handler: Any = None


def bind_protocol(spec: Any) -> None:
    """Serve this ``ProtocolSpec`` at ``/protocol`` for the step checklist.

    Called once by the composition root (``har.app``) before ``create_app``;
    pass ``None`` to unbind.  Thread-safe for tests that build several apps.
    """
    global _protocol_spec
    with _protocol_lock:
        _protocol_spec = spec


def _bound_protocol() -> Any:
    with _protocol_lock:
        return _protocol_spec


def bind_reset(handler: Any) -> None:
    """Register the callable that a ``POST /reset`` invokes (manual restart).

    Additive hook used exactly like ``bind_protocol`` so the frozen
    ``create_app`` signature stays unchanged.  ``handler()`` must be safe to
    call from the GUI thread (typically it only flags the main loop) and
    should return a dict (merged into the ``/reset`` JSON response).
    """
    global _reset_handler
    with _reset_lock:
        _reset_handler = handler


def _bound_reset() -> Any:
    with _reset_lock:
        return _reset_handler


def _as_dict(status: Any) -> dict:
    if isinstance(status, UiStatus):
        return status.to_dict()
    if hasattr(status, "to_dict"):
        return status.to_dict()
    return dict(status)


def create_app(
    streamer: Any,
    status_provider: Callable[[], UiStatus],
    log_tail: Callable[[int], "list[StepEvent]"],
):
    """Build the monitoring GUI.  See module docstring for the routes."""
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:  # pragma: no cover - depends on machine
        raise RuntimeError("the browser GUI needs flask (pip install flask)") from exc

    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.get("/")
    def index():  # noqa: ANN202 - flask route
        html = _INDEX.read_text(encoding="utf-8")
        return Response(html, mimetype="text/html; charset=utf-8")

    @app.get("/stream")
    def stream():  # noqa: ANN202 - flask route
        def generate() -> Iterator[bytes]:
            yield from streamer.mjpeg_stream()

        return Response(
            generate(),
            mimetype=f"multipart/x-mixed-replace; boundary={streamer.boundary}",
        )

    @app.get("/status")
    def status():  # noqa: ANN202 - flask route
        return jsonify(_as_dict(status_provider()))

    @app.get("/events")
    def events():  # noqa: ANN202 - flask route
        try:
            n = int(request.args.get("n", "20"))
        except ValueError:
            n = 20
        n = max(0, min(n, 500))
        recent = log_tail(n)
        return jsonify([e.to_dict() if hasattr(e, "to_dict") else dict(e) for e in recent])

    @app.get("/protocol")
    def protocol():  # noqa: ANN202 - flask route
        spec = _bound_protocol()
        if spec is None:
            return jsonify({"error": "no protocol bound"}), 404
        return jsonify(spec.to_dict())

    @app.post("/reset")
    def reset():  # noqa: ANN202 - flask route
        """Manual restart: reset the experiment sequence back to step 1.

        The handler is bound by the composition root via :func:`bind_reset`.
        It does *not* restart the app, camera or models — the composition root
        only flags the frame loop, which resets the sequence between frames.
        """
        handler = _bound_reset()
        if handler is None:
            return jsonify({"ok": False, "error": "no reset handler bound"}), 503
        try:
            result = handler() or {}
        except Exception as exc:  # noqa: BLE001 - surface as a clean 500
            return jsonify({"ok": False, "error": str(exc)}), 500
        if isinstance(result, dict):
            body = dict(result)
        else:
            body = {"ok": True, "detail": str(result)}
        body.setdefault("ok", True)
        return jsonify(body)

    return app
