"""SIH26174 Protocol Cognition Package (Track A).

Importing predicate code must not require optional YAML support.  The protocol
loader is therefore exposed lazily: callers can still use
``from har.protocol import load_protocol`` when PyYAML is installed, while
``har.protocol.predicates`` remains importable in a bare interpreter.
"""

PREDICATE_VOCABULARY = frozenset({
    "object_stable",
    "object_left_zone",
    "hoi_cycle",
    "settled",
    "transfer",
    "hands_clear",
})


def __getattr__(name: str):
    if name in {"ProtocolError", "load_protocol"}:
        from har.protocol.spec import ProtocolError, load_protocol

        return {"ProtocolError": ProtocolError, "load_protocol": load_protocol}[name]
    if name == "SequenceValidator":
        from har.protocol.validator import SequenceValidator

        return SequenceValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProtocolError", "load_protocol", "SequenceValidator", "PREDICATE_VOCABULARY"]
