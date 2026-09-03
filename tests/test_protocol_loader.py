"""Unit tests for the protocol loader (Track A, Step A2)."""

import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - bare interpreter (plan §3)
    yaml = None

from har.contracts import ProtocolSpec, StepSpec, Zone

if yaml is not None:
    from har.protocol.spec import ProtocolError, load_protocol

PTS01_PATH = Path(__file__).resolve().parents[1] / "protocols" / "pts01.yaml"


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class ProtocolLoaderTests(unittest.TestCase):
    def test_load_valid_pts01_protocol(self):
        spec = load_protocol(PTS01_PATH, frame_size=(640, 480))
        self.assertIsInstance(spec, ProtocolSpec)
        self.assertEqual("PTS-01", spec.protocol_id)
        self.assertEqual("Payload Tray Sorting & Sample Transfer", spec.title)
        self.assertEqual("1.0.0", spec.version)
        self.assertEqual(7, len(spec.steps))
        self.assertEqual(4, len(spec.zones))
        self.assertEqual(4, len(spec.objects))
        # Live demo build: no vial / rack slot / sample transfer.
        self.assertNotIn("vial", spec.objects)
        self.assertIsNone(spec.zone("rack_slot"))
        self.assertIsNone(spec.step("SAMPLE_TRANSFER"))
        self.assertEqual(
            ("tray", "tray_lid", "red_box", "blue_box"), spec.objects
        )
        # The final step follows straight on from VERIFY_BLUE_PLACED (the
        # deleted vial step used to sit between them).
        stow = spec.step("STOW_AND_CLOSE")
        self.assertIsNotNone(stow)
        self.assertEqual(7, stow.index)
        self.assertEqual(("VERIFY_BLUE_PLACED",), stow.requires)

        # Check Zone coordinate resolution to 640x480
        # rack_roi: [0.08, 0.15, 0.92, 0.95] -> [51.2, 72.0, 588.8, 456.0]
        rack_roi = spec.zone("rack_roi")
        self.assertIsNotNone(rack_roi)
        self.assertEqual((51.2, 72.0, 588.8, 456.0), rack_roi.box)

        # Check Zone coordinate resolution for 1280x720
        spec_hd = load_protocol(PTS01_PATH, frame_size=(1280, 720))
        rack_roi_hd = spec_hd.zone("rack_roi")
        self.assertEqual((102.4, 108.0, 1177.6, 684.0), rack_roi_hd.box)

    def test_step_spec_attributes(self):
        spec = load_protocol(PTS01_PATH, frame_size=(640, 480))
        step1 = spec.step("PRESENT_TRAY")
        self.assertIsNotNone(step1)
        self.assertEqual(1, step1.index)
        self.assertEqual("tray", step1.target)
        self.assertEqual("rack_roi", step1.zone)
        self.assertEqual(15, step1.hold_frames)
        self.assertEqual((), step1.requires)

        step3 = spec.step("EXTRACT_RED")
        self.assertIsNotNone(step3)
        self.assertEqual(("OPEN_TRAY",), step3.requires)
        self.assertTrue(step3.voice_alert)

    def _write_temp_yaml(self, data: dict) -> Path:
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(data, tf)
        tf.close()
        return Path(tf.name)

    def _valid_base_dict(self) -> dict:
        return yaml.safe_load(PTS01_PATH.read_text(encoding="utf-8"))

    def test_raises_on_missing_file(self):
        with self.assertRaises(ProtocolError):
            load_protocol("non_existent_file.yaml", (640, 480))

    def test_raises_on_invalid_frame_size(self):
        with self.assertRaises(ProtocolError):
            load_protocol(PTS01_PATH, (-640, 480))
        with self.assertRaises(ProtocolError):
            load_protocol(PTS01_PATH, (640, 0))

    def test_raises_on_unknown_predicate(self):
        data = self._valid_base_dict()
        data["steps"][0]["predicate"] = "unknown_predicate(tray)"
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("unknown predicate", str(ctx.exception))

    def test_raises_on_malformed_predicate_string(self):
        data = self._valid_base_dict()
        data["steps"][0]["predicate"] = "not_a_function_call"
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("malformed predicate", str(ctx.exception))

    def test_raises_on_undeclared_target(self):
        data = self._valid_base_dict()
        data["steps"][0]["target"] = "golden_snitch"
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("undeclared target", str(ctx.exception))

    def test_raises_on_undeclared_zone(self):
        data = self._valid_base_dict()
        data["steps"][0]["zone"] = "zone_z"
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("undeclared zone", str(ctx.exception))

    def test_raises_on_duplicate_step_id(self):
        data = self._valid_base_dict()
        data["steps"][1]["step_id"] = data["steps"][0]["step_id"]
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("Duplicate step_id", str(ctx.exception))

    def test_raises_on_duplicate_zone_id(self):
        data = self._valid_base_dict()
        data["zones"].append(dict(data["zones"][0]))
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("Duplicate zone ID", str(ctx.exception))

    def test_raises_on_dangling_requires(self):
        data = self._valid_base_dict()
        data["steps"][1]["requires"] = ["NON_EXISTENT_STEP"]
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("dangling or forward dependency", str(ctx.exception))

    def test_raises_on_forward_requires(self):
        data = self._valid_base_dict()
        data["steps"][0]["requires"] = ["OPEN_TRAY"]
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("dangling or forward dependency", str(ctx.exception))

    def test_raises_on_non_linear_chain(self):
        data = self._valid_base_dict()
        # Step 3 (index 3) should require step 2, let's make it require step 1
        data["steps"][2]["requires"] = [data["steps"][0]["step_id"]]
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("Non-linear chain", str(ctx.exception))

    def test_raises_on_invalid_zone_box(self):
        data = self._valid_base_dict()
        data["zones"][0]["box"] = [0.8, 0.2, 0.1, 0.9]  # x1 > x2
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("x1 < x2", str(ctx.exception))

        data = self._valid_base_dict()
        data["zones"][0]["box"] = [0.1, 0.2, 1.5, 0.9]  # > 1.0
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("outside [0.0, 1.0]", str(ctx.exception))

    def test_raises_on_unknown_step_keys(self):
        data = self._valid_base_dict()
        data["steps"][0]["unexpected_key"] = 123
        path = self._write_temp_yaml(data)
        with self.assertRaises(ProtocolError) as ctx:
            load_protocol(path, (640, 480))
        self.assertIn("unknown keys", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
