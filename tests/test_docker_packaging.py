"""Container packaging — the image is a deployment artefact, so its inputs get
the same treatment as the rest of the system.

The Dockerfile never runs a dependency resolver: it installs
``requirements.lock`` with ``--no-deps`` after filtering it through
``tools/docker_requirements.py``.  That makes the lock, the filter and the
Dockerfile a single contract, and these tests are the check on it — a lock
bump that reintroduces a CUDA wheel, or a ``.dockerignore`` line that drops
the model weights the pose backend loads, fails here rather than at the demo.

Standard library only, like the rest of the suite: a bare interpreter runs all
of it.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from tools.docker_requirements import (
    CUDA_ONLY_PREFIXES,
    GUI_TO_HEADLESS,
    TORCH_DISTRIBUTIONS,
    channel_problems,
    filter_pins,
    main as requirements_main,
    parse_pins,
    torch_pins,
)

REPO = Path(__file__).resolve().parents[1]
LOCK = (REPO / "requirements.lock").read_text(encoding="utf-8").splitlines()


def _load_module(path: Path, name: str):
    """Load a module by path (``docker/`` is not a package, and ``docker`` is
    also a PyPI distribution name — never import it by dotted name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(relative: str) -> str:
    path = REPO / relative
    if not path.exists():
        raise AssertionError(f"{relative} is missing")
    return path.read_text(encoding="utf-8")


def _names(lines):
    return {name for name, _ in parse_pins(lines)}


class RequirementsFilterTests(unittest.TestCase):
    def test_cpu_channel_drops_every_cuda_wheel(self):
        kept = _names(filter_pins(parse_pins(LOCK), "cpu"))
        cuda = {name for name in kept if name.startswith(CUDA_ONLY_PREFIXES)}
        self.assertEqual(set(), cuda)
        self.assertNotIn("triton", kept)

    def test_cpu_channel_drops_only_cuda_wheels_torch_and_the_gui_opencv(self):
        dropped = _names(LOCK) - _names(filter_pins(parse_pins(LOCK), "cpu"))
        for name in dropped:
            self.assertTrue(
                name in TORCH_DISTRIBUTIONS
                or name.startswith(CUDA_ONLY_PREFIXES)
                or name in GUI_TO_HEADLESS,
                f"{name} was dropped from the cpu set but is not a CUDA wheel, "
                f"torch, or the substituted GUI opencv build",
            )
        self.assertIn("torch", dropped)
        self.assertIn("opencv-python", dropped)

    def test_cuda_channel_keeps_the_nvidia_runtime_the_lock_pins(self):
        kept = _names(filter_pins(parse_pins(LOCK), "cuda"))
        self.assertIn("nvidia-cudnn-cu13", kept)
        self.assertIn("triton", kept)
        self.assertNotIn("torch", kept)  # always installed from the wheel index

    def test_top_level_pins_survive_both_channels(self):
        for channel in ("cpu", "cuda"):
            with self.subTest(channel=channel):
                kept = dict(parse_pins(filter_pins(parse_pins(LOCK), channel)))
                # 8.3.0 is the floor that can load the committed YOLO11 weights.
                self.assertEqual("ultralytics==8.3.0", kept.get("ultralytics"))
                self.assertEqual("numpy==1.26.4", kept.get("numpy"))
                self.assertEqual("PyYAML==6.0.3", kept.get("pyyaml"))
                self.assertEqual("pyttsx3==2.99", kept.get("pyttsx3"))
                self.assertEqual("Flask==3.1.3", kept.get("flask"))

    def test_opencv_is_swapped_for_the_headless_twin_at_the_same_version(self):
        """ultralytics requires "opencv-python>=4.6.0", so a resolving
        install would pull the GUI build (and libGL) back in; the image installs
        with --no-deps precisely so this substitution holds."""
        kept = dict(parse_pins(filter_pins(parse_pins(LOCK), "cpu")))
        self.assertEqual("opencv-python-headless==4.11.0.86", kept.get("opencv-python-headless"))
        self.assertNotIn("opencv-python", kept)
        for gui, headless in GUI_TO_HEADLESS.items():
            gui_version = dict(parse_pins(LOCK))[gui].split("==", 1)[1]
            self.assertEqual(f"{headless}=={gui_version}", kept[headless])

    def test_gui_opencv_opt_out_keeps_the_lock_unchanged(self):
        kept = _names(filter_pins(parse_pins(LOCK), "cpu", headless=False))
        self.assertIn("opencv-python", kept)
        self.assertNotIn("opencv-python-headless", kept)

    def test_torch_only_prints_exactly_the_torch_pins(self):
        self.assertEqual(["torch==2.14.0", "torchvision==0.29.0"], torch_pins(parse_pins(LOCK)))

    def test_check_mode_rejects_an_unpinned_requirements_file(self):
        problems = channel_problems((REPO / "requirements.txt").read_text(encoding="utf-8").splitlines(), "cpu")
        self.assertTrue(problems)
        self.assertTrue(any("no longer pins torch" in problem for problem in problems))

    def test_check_mode_accepts_the_lock(self):
        self.assertEqual([], channel_problems(LOCK, "cpu"))
        self.assertEqual([], channel_problems(LOCK, "cuda"))
        self.assertEqual(0, requirements_main(["--check", str(REPO / "requirements.lock")]))

    def test_unknown_channel_is_rejected(self):
        self.assertTrue(channel_problems(LOCK, "rocm"))


class TorchChannelGuardTests(unittest.TestCase):
    def setUp(self):
        self.check = _load_module(REPO / "docker" / "check_torch.py", "har_check_torch").check

    def test_cpu_channel_rejects_a_cuda_build(self):
        problem = self.check("cpu", "13.0", "2.14.0+cu130")
        self.assertIn("CUDA 13.0", problem)
        self.assertIn("download.pytorch.org/whl/cpu", problem)

    def test_cpu_channel_accepts_a_cpu_build(self):
        self.assertEqual("", self.check("cpu", None, "2.14.0+cpu"))

    def test_cuda_channel_rejects_a_cpu_build(self):
        self.assertIn("CPU-only", self.check("cuda", None, "2.14.0+cpu"))

    def test_cuda_channel_accepts_a_cuda_build(self):
        self.assertEqual("", self.check("cuda", "13.0", "2.14.0+cu130"))

    def test_unknown_channel_is_rejected(self):
        self.assertIn("unknown channel", self.check("rocm", None, "2.14.0"))


class DockerfileTests(unittest.TestCase):
    def setUp(self):
        self.text = _read("Dockerfile")

    def test_base_image_is_pinned_to_a_release_not_latest(self):
        bases = re.findall(r"^FROM\s+(\S+)", self.text, re.M)
        self.assertTrue(bases, "the Dockerfile has no FROM")
        for base in bases:
            self.assertNotIn(":latest", base)
            self.assertIn("slim-bookworm", base)

    def test_build_is_multi_stage_with_a_virtualenv_handoff(self):
        self.assertLessEqual(2, len(re.findall(r"^FROM\s", self.text, re.M)))
        self.assertIn("COPY --from=builder", self.text)

    def test_dependency_resolution_is_never_run_at_build_time(self):
        for install in re.findall(r"^\s*pip install.*$", self.text, re.M):
            self.assertIn("--no-deps", install, f"unlocked install: {install.strip()}")
        self.assertIn("tools/docker_requirements.py", self.text)
        self.assertIn("requirements.lock", self.text)

    def test_channel_guard_runs_during_the_build(self):
        self.assertIn("check_torch.py", self.text)

    def test_runtime_hardening(self):
        self.assertIn("USER har", self.text)
        self.assertIn("HEALTHCHECK", self.text)
        self.assertIn("EXPOSE 8080", self.text)
        self.assertIn('ENTRYPOINT ["/app/docker/entrypoint.sh"]', self.text)
        self.assertNotIn("USER root", self.text)

    def test_declared_port_matches_the_cli_default(self):
        self.assertIn('help="GUI/MJPEG port (default: 8080)"', _read("har/app.py"))


class DockerIgnoreTests(unittest.TestCase):
    def setUp(self):
        self.lines = [line.strip() for line in _read(".dockerignore").splitlines()]
        self.entries = {line for line in self.lines if line and not line.startswith("#")}

    def test_local_state_stays_out_of_the_context(self):
        for entry in (".git", ".venv", ".env", ".env.*", "runs/", "**/__pycache__/"):
            with self.subTest(entry=entry):
                self.assertIn(entry, self.entries)

    def test_runtime_inputs_are_not_excluded(self):
        """The default CMD replays demo/correct.mp4 and --wrists pose loads
        models/yolo11n-pose.pt; excluding either would break the image."""
        for needed in ("models", "demo", "protocols", "config", "har", "docker", "tests"):
            with self.subTest(needed=needed):
                self.assertNotIn(needed, self.entries)
                self.assertNotIn(f"{needed}/", self.entries)


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.text = _read("docker-compose.yml")
        self.gpu = _read("compose.gpu.yml")

    def test_service_publishes_the_gui_port_and_the_artefact_volume(self):
        self.assertIn('"${HAR_HOST_PORT:-8080}:8080"', self.text)
        self.assertIn(":/data", self.text)

    def test_shutdown_is_graceful_so_recordings_are_finalised(self):
        self.assertIn("stop_grace_period", self.text)
        self.assertIn("init: true", self.text)

    def test_container_user_is_configurable_for_bind_mounts(self):
        self.assertIn('user: "${HAR_UID:-1001}:${HAR_GID:-1001}"', self.text)

    def test_restart_policy_survives_a_demo_machine_reboot(self):
        self.assertIn("restart: unless-stopped", self.text)

    def test_camera_service_is_behind_a_profile_and_passes_the_device(self):
        self.assertIn('profiles: ["camera"]', self.text)
        self.assertIn("/dev/video0:/dev/video0", self.text)

    def test_gpu_override_switches_the_channel_and_reserves_the_device(self):
        self.assertIn("TORCH_CHANNEL: cuda", self.gpu)
        self.assertIn("driver: nvidia", self.gpu)
        self.assertIn("capabilities: [gpu]", self.gpu)

    def test_every_env_default_is_documented(self):
        example = _read(".env.example")
        for variable in sorted(set(re.findall(r"\$\{([A-Z_]+)", self.text))):
            with self.subTest(variable=variable):
                self.assertIn(variable, example)


class ScriptTests(unittest.TestCase):
    def test_shell_scripts_are_syntactically_valid(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash on this machine")
        for script in ("docker/entrypoint.sh", "docker/healthcheck.sh"):
            with self.subTest(script=script):
                result = subprocess.run([bash, "-n", str(REPO / script)],
                                        capture_output=True, text=True)
                self.assertEqual(0, result.returncode, result.stderr)

    def test_shell_scripts_are_executable_and_shebanged(self):
        for script in ("docker/entrypoint.sh", "docker/healthcheck.sh"):
            with self.subTest(script=script):
                path = REPO / script
                self.assertTrue(path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash"))
                self.assertTrue(path.stat().st_mode & 0o111, f"{script} is not executable")

    def test_entrypoint_targets_the_cli_and_the_artefact_volume(self):
        text = _read("docker/entrypoint.sh")
        self.assertIn("exec python -u -m har.app", text)
        self.assertIn("--out-dir", text)
        self.assertIn("/data", text)

    def test_healthcheck_probes_the_gui_status_route(self):
        text = _read("docker/healthcheck.sh")
        self.assertIn("/status", text)
        self.assertIn("har.app", text)
        self.assertIn("/status", _read("har/ui/web.py"))

    def test_sigterm_shuts_the_run_down_like_ctrl_c(self):
        """`docker stop` sends SIGTERM; the app must finalise the recording and
        close the event log rather than dying mid-write."""
        from har.app import _sigterm_to_interrupt

        with self.assertRaises(KeyboardInterrupt):
            _sigterm_to_interrupt(15, None)
        self.assertIn("signal.signal(signal.SIGTERM, _sigterm_to_interrupt)", _read("har/app.py"))


class RepoHygieneTests(unittest.TestCase):
    def test_env_file_is_not_committed(self):
        self.assertIn(".env", _read(".gitignore"))


if __name__ == "__main__":
    unittest.main()
