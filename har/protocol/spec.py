"""Protocol loader and validation for SIH26174 experiment procedures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from har.contracts import BBox, ProtocolSpec, StepSpec, Zone

PREDICATE_VOCABULARY = frozenset({
    "object_stable",
    "object_left_zone",
    "hoi_cycle",
    "settled",
    "transfer",
    "hands_clear",
})

_ALLOWED_STEP_KEYS = frozenset({
    "step_id",
    "index",
    "title",
    "instruction",
    "predicate",
    "target",
    "zone",
    "requires",
    "hold_frames",
    "timeout_s",
    "voice_prompt",
    "voice_alert",
})


class ProtocolError(ValueError):
    """Raised when a protocol definition violates format or semantic invariants."""
    pass


def _resolve_zone_box(box_norm: Any, frame_size: tuple[int, int], zone_id: str) -> BBox:
    if not isinstance(box_norm, (list, tuple)) or len(box_norm) != 4:
        raise ProtocolError(f"Zone '{zone_id}' box must be a 4-element sequence [x1, y1, x2, y2]")
    try:
        x1_n, y1_n, x2_n, y2_n = [float(v) for v in box_norm]
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"Zone '{zone_id}' box values must be numbers") from exc

    for val, name in [(x1_n, "x1"), (y1_n, "y1"), (x2_n, "x2"), (y2_n, "y2")]:
        if not (0.0 <= val <= 1.0):
            raise ProtocolError(f"Zone '{zone_id}' coordinate {name}={val} is outside [0.0, 1.0]")

    if x1_n >= x2_n or y1_n >= y2_n:
        raise ProtocolError(f"Zone '{zone_id}' box must satisfy x1 < x2 and y1 < y2")

    width, height = frame_size
    return (
        round(x1_n * width, 2),
        round(y1_n * height, 2),
        round(x2_n * width, 2),
        round(y2_n * height, 2),
    )


def load_protocol(path: str | Path, frame_size: tuple[int, int]) -> ProtocolSpec:
    """Load and validate a YAML protocol file, resolving normalized coordinates to pixels.

    Args:
        path: Path to the YAML protocol file.
        frame_size: (width, height) in pixels.

    Returns:
        Validated ProtocolSpec with pixel-space Zone bounding boxes.

    Raises:
        ProtocolError: On syntax errors, unknown predicates, dangling dependencies,
                       undeclared targets/zones, duplicate IDs, or non-linear chains.
    """
    if not isinstance(frame_size, (tuple, list)) or len(frame_size) != 2:
        raise ProtocolError(f"frame_size must be a (width, height) tuple, got {frame_size}")
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ProtocolError(f"frame_size dimensions must be positive integers, got {frame_size}")

    file_path = Path(path)
    if not file_path.is_file():
        raise ProtocolError(f"Protocol file not found: {path}")

    try:
        content = file_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ProtocolError(f"YAML parsing error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol root must be a mapping, got {type(raw).__name__}")

    for req_field in ("protocol_id", "title", "version", "steps"):
        if req_field not in raw:
            raise ProtocolError(f"Missing required top-level field '{req_field}'")

    protocol_id = str(raw["protocol_id"])
    title = str(raw["title"])
    version = str(raw["version"])

    raw_objects = raw.get("objects", [])
    if not isinstance(raw_objects, list):
        raise ProtocolError("'objects' must be a list of string identifiers")
    objects = tuple(str(o) for o in raw_objects)
    if len(objects) != len(set(objects)):
        raise ProtocolError(f"Duplicate object names in objects declaration: {objects}")

    declared_objects = set(objects)

    raw_zones = raw.get("zones", [])
    if not isinstance(raw_zones, list):
        raise ProtocolError("'zones' must be a list of zone mappings")

    zones_list: list[Zone] = []
    zone_ids: set[str] = set()
    for z in raw_zones:
        if not isinstance(z, dict):
            raise ProtocolError(f"Zone entry must be a dictionary, got {type(z).__name__}")
        if "id" not in z or "box" not in z:
            raise ProtocolError(f"Zone definition missing 'id' or 'box': {z}")
        z_id = str(z["id"])
        if z_id in zone_ids:
            raise ProtocolError(f"Duplicate zone ID '{z_id}'")
        zone_ids.add(z_id)
        box_px = _resolve_zone_box(z["box"], frame_size, z_id)
        zones_list.append(Zone(id=z_id, box=box_px, label=str(z.get("label", ""))))

    raw_steps = raw["steps"]
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        raise ProtocolError("'steps' must be a non-empty list of step definitions")

    step_specs: list[StepSpec] = []
    step_ids: set[str] = set()
    seen_step_ids: list[str] = []

    for i, step in enumerate(raw_steps, start=1):
        if not isinstance(step, dict):
            raise ProtocolError(f"Step {i} must be a dictionary, got {type(step).__name__}")

        unknown_keys = set(step) - _ALLOWED_STEP_KEYS
        if unknown_keys:
            raise ProtocolError(f"Step {i} contains unknown keys: {sorted(unknown_keys)}")

        for required_key in ("step_id", "index", "title", "instruction", "predicate"):
            if required_key not in step:
                raise ProtocolError(f"Step {i} missing required key '{required_key}'")

        step_id = str(step["step_id"])
        if step_id in step_ids:
            raise ProtocolError(f"Duplicate step_id '{step_id}' at index {i}")
        step_ids.add(step_id)

        index = int(step["index"])
        if index != i:
            raise ProtocolError(f"Step '{step_id}' index {index} does not match 1-based order {i}")

        predicate_str = str(step["predicate"]).strip()
        match = re.match(r"^([a-z_]+)\(", predicate_str)
        if not match:
            raise ProtocolError(f"Step '{step_id}' has malformed predicate string: {predicate_str}")
        pred_name = match.group(1)
        if pred_name not in PREDICATE_VOCABULARY:
            raise ProtocolError(f"Step '{step_id}' uses unknown predicate '{pred_name}'")

        target = str(step.get("target", ""))
        if target and target not in declared_objects:
            raise ProtocolError(f"Step '{step_id}' references undeclared target object '{target}'")

        zone = str(step.get("zone", ""))
        if zone and zone not in zone_ids:
            raise ProtocolError(f"Step '{step_id}' references undeclared zone '{zone}'")

        raw_reqs = step.get("requires", [])
        if not isinstance(raw_reqs, (list, tuple)):
            raise ProtocolError(f"Step '{step_id}' requires must be a sequence")
        requires = tuple(str(r) for r in raw_reqs)

        # Check for dangling or forward references
        for req in requires:
            if req not in seen_step_ids:
                raise ProtocolError(f"Step '{step_id}' has dangling or forward dependency '{req}'")

        # Check linear dependency chain: step 1 requires [], step k requires [step_{k-1}]
        if i == 1:
            if len(requires) != 0:
                raise ProtocolError(f"First step '{step_id}' must have empty requires, got {requires}")
        else:
            prev_step_id = seen_step_ids[-1]
            if list(requires) != [prev_step_id]:
                raise ProtocolError(
                    f"Non-linear chain: step '{step_id}' (index {i}) must require exactly ['{prev_step_id}'], got {list(requires)}"
                )

        seen_step_ids.append(step_id)

        hold_frames = int(step.get("hold_frames", 1))
        if hold_frames <= 0:
            raise ProtocolError(f"Step '{step_id}' hold_frames must be > 0, got {hold_frames}")

        timeout_s = float(step.get("timeout_s", 60.0))
        if timeout_s <= 0.0:
            raise ProtocolError(f"Step '{step_id}' timeout_s must be > 0, got {timeout_s}")

        step_specs.append(StepSpec(
            step_id=step_id,
            index=index,
            title=str(step["title"]),
            instruction=str(step["instruction"]),
            predicate=predicate_str,
            target=target,
            zone=zone,
            requires=requires,
            hold_frames=hold_frames,
            timeout_s=timeout_s,
            voice_prompt=str(step.get("voice_prompt", "")),
            voice_alert=str(step.get("voice_alert", "")),
        ))

    return ProtocolSpec(
        protocol_id=protocol_id,
        title=title,
        version=version,
        steps=tuple(step_specs),
        zones=tuple(zones_list),
        objects=objects,
    )
