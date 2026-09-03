"""Validation of protocols/pts01.yaml.

This is a *config* test, not the loader - ``har/protocol/spec.py`` (Track A,
Phase 1) turns this file into ``contracts.ProtocolSpec``. What is asserted here
is the invariants the sequence validator silently depends on: contiguous
indices, no forward references, known predicates, resolvable targets and zones.
A broken protocol file otherwise shows up as a mysterious runtime mis-sequence.
"""

import re
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from har.contracts import StepSpec

PROTOCOL_FILE = Path(__file__).resolve().parents[1] / "protocols" / "pts01.yaml"

#: Frozen predicate vocabulary. Mirrors the header of protocols/pts01.yaml.
PREDICATE_VOCABULARY = {
    "object_stable",
    "object_left_zone",
    "hoi_cycle",
    "settled",
    "transfer",
    "hands_clear",
}


def step_spec_fields() -> set[str]:
    return {f for f in StepSpec.__dataclass_fields__ if f not in {"step_id", "index"}}


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class ProtocolConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = yaml.safe_load(PROTOCOL_FILE.read_text(encoding="utf-8"))
        cls.steps = cls.doc["steps"]
        cls.objects = set(cls.doc["objects"])
        cls.zones = {z["id"] for z in cls.doc["zones"]}

    def test_protocol_is_pts01_with_seven_ordered_steps(self):
        self.assertEqual("PTS-01", self.doc["protocol_id"])
        self.assertEqual(7, len(self.steps))
        self.assertEqual(list(range(1, 8)), [s["index"] for s in self.steps])

    def test_step_ids_are_unique(self):
        ids = [s["step_id"] for s in self.steps]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate step ids: {ids}")

    def test_every_predicate_is_in_the_frozen_vocabulary(self):
        for step in self.steps:
            name = re.match(r"^([a-z_]+)\(", step["predicate"])
            self.assertIsNotNone(name, f"unparsable predicate: {step['predicate']}")
            self.assertIn(name.group(1), PREDICATE_VOCABULARY,
                          f"{step['step_id']} uses unknown predicate {name.group(1)}")

    def test_targets_and_zones_resolve(self):
        for step in self.steps:
            if step.get("target"):
                self.assertIn(step["target"], self.objects, f"{step['step_id']} target")
            if step.get("zone"):
                self.assertIn(step["zone"], self.zones, f"{step['step_id']} zone")

    def test_no_forward_references_in_requires(self):
        """A step may only depend on steps that come before it."""
        seen: list[str] = []
        for step in self.steps:
            for dependency in step.get("requires", []):
                self.assertIn(dependency, seen,
                              f"{step['step_id']} requires later step {dependency}")
            seen.append(step["step_id"])

    def test_dependency_chain_is_linear_and_complete(self):
        """PTS-01 is a strict chain: skipping is a violation, never a branch."""
        for previous, step in zip(self.steps, self.steps[1:]):
            self.assertEqual([previous["step_id"]], list(step.get("requires", [])),
                             f"{step['step_id']} must require exactly {previous['step_id']}")

    def test_zones_are_normalised_and_well_formed(self):
        for zone in self.doc["zones"]:
            x1, y1, x2, y2 = zone["box"]
            for value in zone["box"]:
                self.assertGreaterEqual(value, 0.0, zone["id"])
                self.assertLessEqual(value, 1.0, zone["id"])
            self.assertLess(x1, x2, zone["id"])
            self.assertLess(y1, y2, zone["id"])

    def test_every_step_announces_itself_and_violations_speak(self):
        for step in self.steps:
            self.assertTrue(step.get("voice_prompt"), f"{step['step_id']} has no voice prompt")
            self.assertGreater(step.get("timeout_s", 0), 0, step["step_id"])
            self.assertGreater(step.get("hold_frames", 0), 0, step["step_id"])

    def test_step_keys_are_a_subset_of_the_contract(self):
        """A typo in the yaml must fail here, not silently in the validator."""
        allowed = step_spec_fields() | {"step_id", "index"}
        for step in self.steps:
            unknown = set(step) - allowed
            self.assertEqual(set(), unknown, f"{step['step_id']} has unknown keys {unknown}")


if __name__ == "__main__":
    unittest.main()
