"""In-frame HUD overlay (C10) — the projector fallback for the browser GUI.

``draw_hud`` draws directly onto the BGR frame (in place, per Appendix A),
so everything downstream of it — the C7 recording and the C8 MJPEG stream —
carries the same status readout.  It renders, never computes: every value
comes out of the ``UiStatus`` snapshot Person A produces and the
``FrameEvidence`` Person B packed.

Layout (readable at 1280×720 from three metres: face-height text, thick
strokes, high contrast):

* top-left panel — protocol id/title, run state, current step, elapsed time;
* "NEXT:" instruction under it (what the operator should do);
* per-object tracked boxes, colour-coded by hand-object state, plus wrist
  markers;
* bottom strip — FPS (D1's evidence) and, on any violation, a red alert
  banner carrying the spoken message.

cv2 is imported lazily so the module loads in a bare interpreter.
"""

from __future__ import annotations

from typing import Any

from har.contracts import FrameEvidence, HandObjectState, UiStatus

__all__ = ["draw_hud"]

# Object box colours (BGR) by HOI state — green settled, amber in hand.
_STATE_COLOURS = {
    HandObjectState.IDLE.value: (90, 200, 90),
    HandObjectState.NEAR_OBJECT.value: (60, 200, 240),
    HandObjectState.PICKED_UP.value: (0, 190, 255),
    HandObjectState.CARRYING.value: (0, 190, 255),
    HandObjectState.RELEASED.value: (200, 160, 90),
}
_PANEL = (18, 18, 18)
_TEXT = (245, 245, 245)
_MUTED = (185, 185, 185)
_ACCENT = (90, 200, 240)
_ALERT_BG = (36, 36, 200)
_OK = (90, 200, 90)


def _scale(frame_shape: Any) -> float:
    """Text/line scale factor: 1.0 at 720p, smaller on tiny test frames."""
    return max(0.45, min(1.6, frame_shape[0] / 720.0))


def _fit(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def draw_hud(frame: Any, status: UiStatus, evidence: FrameEvidence) -> None:
    """Draw the status overlay onto ``frame`` in place."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    s = _scale(frame.shape)
    thick = max(1, round(2 * s))
    pad = round(10 * s)

    # ---- top-left status panel --------------------------------------
    state_colour = _OK if status.state == "COMPLETE" else (_ALERT_BG if status.violations else _ACCENT)
    lines = [
        (f"{status.protocol_id} — {status.protocol_title}", _TEXT),
        (
            f"{status.state}   step {status.current_step_index}: {status.current_step_id}",
            state_colour,
        ),
        (f"t = {status.t_rel:6.1f}s", _MUTED),
    ]
    y = pad
    x = pad
    sizes = [(0.6 * s, thick), (0.58 * s, thick), (0.5 * s, max(1, thick - 1))]
    rendered = []
    panel_w = 0
    line_h = 0
    for (text, colour), (fscale, fthick) in zip(lines, sizes):
        (tw, th), _ = cv2.getTextSize(text, font, fscale, fthick)
        rendered.append((text, colour, fscale, fthick, th))
        panel_w = max(panel_w, tw)
        line_h += th + round(8 * s)
    panel_h = line_h + 2 * pad
    cv2.rectangle(frame, (0, 0), (panel_w + 3 * pad, panel_h), _PANEL, thickness=-1)
    cv2.rectangle(frame, (0, 0), (panel_w + 3 * pad, panel_h), (70, 70, 70), thickness=1)
    for text, colour, fscale, fthick, th in rendered:
        y += th + round(6 * s)
        cv2.putText(frame, text, (x, y), font, fscale, colour, fthick, cv2.LINE_AA)
        y += round(2 * s)

    # ---- next instruction -------------------------------------------
    if status.state != "COMPLETE" and status.next_instruction:
        text = _fit(f"NEXT: {status.next_instruction}", int(w / (9.5 * s)))
        (tw, th), _ = cv2.getTextSize(text, font, 0.58 * s, thick)
        top = panel_h + round(6 * s)
        cv2.rectangle(frame, (0, top), (tw + 3 * pad, top + th + 2 * pad), (30, 44, 52), thickness=-1)
        cv2.putText(frame, text, (pad, top + th + pad - round(3 * s)),
                    font, 0.58 * s, (150, 220, 250), thick, cv2.LINE_AA)

    # ---- tracked objects + wrists ------------------------------------
    for label in sorted(evidence.objects):
        track = evidence.objects[label]
        if track.box is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in track.box)
        state = evidence.hoi_state(label)
        colour = _STATE_COLOURS.get(state, _OK)
        if not track.measured:
            colour = (110, 110, 110)  # coasting estimate: mute it
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
        caption = f"{label}: {state}" + ("" if track.measured else " (pred)")
        (tw, th), _ = cv2.getTextSize(caption, font, 0.42 * s, 1)
        cy = max(y1 - round(5 * s), th + 2)
        cv2.rectangle(frame, (x1, cy - th - 3), (x1 + tw + 6, cy + 3), _PANEL, thickness=-1)
        cv2.putText(frame, caption, (x1 + 3, cy - 1), font, 0.42 * s, colour, 1, cv2.LINE_AA)
    for wrist in evidence.hands:
        cx, cy = (int(round(v)) for v in wrist.point)
        cv2.circle(frame, (cx, cy), max(3, round(5 * s)), (60, 200, 240), thickness=-1)
        cv2.circle(frame, (cx, cy), max(4, round(7 * s)), _TEXT, thickness=1)

    # ---- bottom strip: FPS + violation banner ------------------------
    strip_h = round(34 * s)
    top = h - strip_h
    cv2.rectangle(frame, (0, top), (w, h), _PANEL, thickness=-1)
    fps_text = f"FPS {status.fps:5.1f}   f={evidence.frame_index}   contract {status.contract_version}"
    cv2.putText(frame, fps_text, (pad, top + strip_h - round(11 * s)),
                font, 0.5 * s, _MUTED, 1, cv2.LINE_AA)
    done = f"{len(status.completed)} completed - step {status.current_step_index}"
    (tw, _), _ = cv2.getTextSize(done, font, 0.5 * s, 1)
    cv2.putText(frame, done, (w - tw - pad, top + strip_h - round(11 * s)),
                font, 0.5 * s, _OK, 1, cv2.LINE_AA)

    if status.violations and status.last_alert:
        banner_h = round(40 * s)
        b_top = top - banner_h
        cv2.rectangle(frame, (0, b_top), (w, top), _ALERT_BG, thickness=-1)
        text = _fit(f"ALERT: {status.last_alert}", int(w / (9.5 * s)))
        (tw, th), _ = cv2.getTextSize(text, font, 0.6 * s, thick)
        cv2.putText(frame, text, (max(pad, (w - tw) // 2), b_top + (banner_h + th) // 2 - round(3 * s)),
                    font, 0.6 * s, _TEXT, thick, cv2.LINE_AA)
