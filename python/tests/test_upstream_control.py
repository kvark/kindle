import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import run_upstream_control as control  # noqa: E402


def test_upstream_accepts_only_the_declared_config_patch(tmp_path, monkeypatch) -> None:
    original = "defaults:\n  env:\n    atari100k: {clip_reward: False}\n"
    directory = tmp_path / "dreamerv3"
    directory.mkdir()
    config = directory / "configs.yaml"
    config.write_text(control.configured_source(original))

    def git(source, *args):
        return {
            ("rev-parse", "HEAD"): control.REVISION + "\n",
            ("ls-files", "--others", "--exclude-standard"): "",
            ("diff", "--name-only", "HEAD"): "dreamerv3/configs.yaml\n",
            ("show", f"{control.REVISION}:dreamerv3/configs.yaml"): original,
            ("diff", "HEAD", "--", "dreamerv3/configs.yaml"): "declared diff",
        }[args]

    monkeypatch.setattr(control, "git", git)
    assert control.validate_source(tmp_path) == "declared diff"
    config.write_text(control.configured_source(original).replace("noops: 0", "noops: 30"))
    with pytest.raises(ValueError, match="PUBLISHED_CONFIG"):
        control.validate_source(tmp_path)


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
