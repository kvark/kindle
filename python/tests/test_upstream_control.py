import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import run_upstream_control as control  # noqa: E402


@pytest.fixture
def configured_upstream(tmp_path, monkeypatch):
    original = "defaults:\n  env:\n    atari100k: {clip_reward: False}\n"
    atari = ("    with self.LOCK:\n"
             "      self.ale.setLoggerMode(ale_py.LoggerMode.Error)\n"
             "      self.ale.setInt(b'random_seed', self.rng.integers(0, 2 ** 31))\n"
             "      self.ale.loadROM(path)\n\n"
             "    self.ale.setFloat('repeat_action_probability', 0.25 if sticky else 0.0)\n")
    changed = ["dreamerv3/configs.yaml", "embodied/envs/atari.py"]
    for name, value in zip(changed, (control.configured_source(original), control.compatible_atari_source(atari))):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def git(source, *args):
        return {
            ("rev-parse", "HEAD"): control.REVISION + "\n",
            ("ls-files", "--others", "--exclude-standard"): "",
            ("diff", "--name-only", "HEAD"): "\n".join(changed),
            ("show", f"{control.REVISION}:dreamerv3/configs.yaml"): original,
            ("show", f"{control.REVISION}:embodied/envs/atari.py"): atari,
            ("diff", "HEAD", "--", *changed): "declared diff",
        }[args]

    monkeypatch.setattr(control, "git", git)
    return tmp_path


def test_upstream_accepts_only_the_declared_config_patch(configured_upstream):
    assert control.validate_source(configured_upstream) == "declared diff"
    config = configured_upstream / "dreamerv3/configs.yaml"
    config.write_text(config.read_text().replace("noops: 0", "noops: 30"))
    with pytest.raises(ValueError, match="PUBLISHED_CONFIG"):
        control.validate_source(configured_upstream)


def test_upstream_rejects_a_different_revision(monkeypatch) -> None:
    monkeypatch.setattr(control, "git", lambda *args: "wrong revision")
    with pytest.raises(ValueError, match="upstream must be at"):
        control.validate_source(Path("/unused"))


def test_upstream_rejects_untracked_source(monkeypatch) -> None:
    def git(source, *args):
        if args == ("rev-parse", "HEAD"):
            return control.REVISION + "\n"
        assert args == ("ls-files", "--others", "--exclude-standard")
        return "dreamerv3/untracked_module.py\n"

    monkeypatch.setattr(control, "git", git)
    with pytest.raises(ValueError, match="untracked source"):
        control.validate_source(Path("/unused"))


def test_upstream_accepts_only_exact_ale_corrections(configured_upstream):
    assert control.validate_source(configured_upstream) == "declared diff"
    path = configured_upstream / "embodied/envs/atari.py"
    source = path.read_text()
    assert source.index("setFloat") < source.index("loadROM")
    for changed in (source.replace("2 ** 31", "2 ** 30"), source.replace("0.25", "0.50")):
        path.write_text(changed)
        with pytest.raises(ValueError, match="exact ALE corrections"):
            control.validate_source(configured_upstream)


def test_upstream_rejects_uncorrected_legacy_wrapper(monkeypatch):
    def git(source, *args):
        return {
            ("rev-parse", "HEAD"): control.REVISION,
            ("ls-files", "--others", "--exclude-standard"): "",
            ("diff", "--name-only", "HEAD"): "dreamerv3/configs.yaml\n",
        }[args]
    monkeypatch.setattr(control, "git", git)
    with pytest.raises(ValueError, match="require exactly"):
        control.validate_source(Path("/unused"))


def test_seed_compatibility_refuses_an_unknown_or_repeated_initialization():
    old = "self.ale.setInt(b'random_seed', self.rng.integers(0, 2 ** 31))"
    for source in ("", old + old):
        with pytest.raises(ValueError, match="unrecognized pinned ALE"):
            control.compatible_atari_source(source)


@pytest.mark.parametrize("sticky", [False, True])
def test_corrected_wrapper_materializes_requested_stickiness(sticky):
    ale_py = pytest.importorskip("ale_py")
    import numpy as np
    import random
    import threading
    from ale_py import roms

    # Same initialization order as the pinned wrapper; no learner/imported source.
    source = """class Probe:
  LOCK = threading.Lock()
  def __init__(self, path, sticky):
    self.ale = ale_py.ALEInterface()
    self.rng = np.random.default_rng(7301)
    with self.LOCK:
      self.ale.setLoggerMode(ale_py.LoggerMode.Error)
      self.ale.setInt(b'random_seed', self.rng.integers(0, 2 ** 31))
      self.ale.loadROM(path)
    self.ale.setFloat('repeat_action_probability', 0.25 if sticky else 0.0)
"""
    namespace = dict(ale_py=ale_py, np=np, threading=threading)
    exec(control.compatible_atari_source(source), namespace)
    actual = namespace["Probe"](roms.get_rom_path("pong"), sticky).ale
    reference = ale_py.ALEInterface()
    reference.setLoggerMode(ale_py.LoggerMode.Error)
    reference.setInt("random_seed", int(np.random.default_rng(7301).integers(0, 2**31)))
    reference.setFloat("repeat_action_probability", .25 if sticky else 0)
    reference.loadROM(roms.get_rom_path("pong"))
    actual.reset_game()
    reference.reset_game()
    rng = random.Random(7301)
    for _ in range(64):
        action = rng.randrange(18)
        assert actual.act(action) == reference.act(action)
        np.testing.assert_array_equal(actual.getScreenRGB(), reference.getScreenRGB())


@pytest.mark.parametrize("outcome,expected", [(None, 0), (0, 0), (7, 7), ("failure", 1)])
def test_control_runs_in_guarded_process_without_gpu_telemetry(tmp_path, monkeypatch, outcome, expected):
    source, logdir = tmp_path / "source", tmp_path / "result"
    source.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--source", str(source), "--logdir", str(logdir),
                                    "--steps", "1000", "--num-envs", "6"])
    monkeypatch.setattr(control, "validate_source", lambda _: "declared diff")
    monkeypatch.setattr(control.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **k: pytest.fail("spawned a child"))
    pid = os.getpid()
    calls = []

    def run(path, *, run_name):
        assert os.getpid() == pid and Path.cwd() == source
        assert path == str(source / "dreamerv3/main.py") and run_name == "__main__"
        args = sys.argv
        for key, value in (("--run.usage.nvsmi", "False"), ("--run.usage.gputil", "False"),
                           ("--run.debug", "True"), ("--jax.compute_dtype", "float32"), ("--run.envs", "6")):
            assert args.count(key) == 1 and args[args.index(key) + 1] == value
        calls.append(path)
        print("native stdout")
        print("native stderr", file=sys.stderr)
        if outcome is not None:
            raise SystemExit(outcome)

    monkeypatch.setattr(control.runpy, "run_path", run)
    with pytest.raises(SystemExit) as result:
        control.main()
    assert result.value.code == expected and len(calls) == 1
    manifest = json.loads((logdir / "reference-manifest.json").read_text())
    assert manifest["exit_code"] == expected
    assert manifest["status"] == ("complete" if expected == 0 else "failed")
    assert (logdir / "RUN_COMPLETE").exists() == (expected == 0)
    assert "native stdout\nnative stderr\n" in (logdir / "console.log").read_text()


def test_control_records_unhandled_failure_without_success_marker(tmp_path, monkeypatch):
    source, logdir = tmp_path / "source", tmp_path / "result"
    source.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--source", str(source), "--logdir", str(logdir)])
    monkeypatch.setattr(control, "validate_source", lambda _: "declared diff")
    monkeypatch.setattr(control.importlib.metadata, "distributions", lambda: [])

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    monkeypatch.setattr(control.runpy, "run_path", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        control.main()
    manifest = json.loads((logdir / "reference-manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["exit_code"] == 1
    assert not (logdir / "RUN_COMPLETE").exists()
