import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from run_gpu_queue import run


def specification(tmp_path):
    binary = tmp_path / "native"
    binary.write_bytes(b"fixture")
    return dict(host=dict(monitoring="host-only", boot_id="fixture", driver="fixture", poll_seconds=2),
                jobs=[dict(name=name, command=[str(binary)], executable_sha256=hashlib.sha256(b"fixture").hexdigest(),
                           timeout_seconds=10, result=str(tmp_path / (name + "-result.json"))) for name in ("one", "two")])


def test_queue_is_serial_and_stops_before_successor_on_failure(tmp_path):
    called = []
    def fail(command, check, env):
        called.append(command)
        assert command[2] == "run" and check
        raise subprocess.CalledProcessError(1, command)
    with pytest.raises(subprocess.CalledProcessError):
        run(specification(tmp_path), tmp_path / "queue", execute=fail)
    assert len(called) == 1
    assert not (tmp_path / "queue/two.json").exists()


def test_queue_requires_guard_and_expected_native_result(tmp_path):
    spec = specification(tmp_path)
    completed = []
    def success(command, check, env):
        guard = Path(command[3])
        guard.mkdir()
        assert len(completed) == (0 if guard.name == "one" else 1)
        (guard / "result.json").write_text(json.dumps(dict(host_guard_passed=True, child_exit_code=0, unfinished_children=[])))
        (tmp_path / (guard.name + "-result.json")).write_text('{"status":"complete"}')
        completed.append(guard.name)
    run(spec, tmp_path / "queue", execute=success)
    assert completed == ["one", "two"]
    assert json.loads((tmp_path / "queue/result.json").read_text())["completed"] == completed
    with pytest.raises(FileExistsError):
        run(spec, tmp_path / "different-queue", execute=success)


def test_queue_refuses_changed_binary_before_spawn(tmp_path):
    spec = specification(tmp_path)
    Path(spec["jobs"][0]["command"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="executable changed"):
        run(spec, tmp_path / "queue", execute=lambda *a, **kw: pytest.fail("spawned"))


def test_queue_rejects_incomplete_native_jsonl_without_running_successor(tmp_path):
    spec = specification(tmp_path)
    spec["jobs"][0].update(result_format="jsonl_last", expected=dict(event="run_end", reason="budget_complete"),
                           minimum=dict(learner_updates=10), environment={"PYTHONPATH": "declared-overlay"})
    calls = []
    def incomplete(command, check, env):
        calls.append(command)
        assert env["PYTHONPATH"] == "declared-overlay"
        guard = Path(command[3])
        guard.mkdir()
        (guard / "result.json").write_text(json.dumps(dict(host_guard_passed=True, child_exit_code=0, unfinished_children=[])))
        Path(spec["jobs"][0]["result"]).write_text('{"event":"run_start"}\n{"event":"run_end","reason":"interrupted","learner_updates":1}\n')
    with pytest.raises(RuntimeError, match="reason"):
        run(spec, tmp_path / "queue", execute=incomplete)
    assert len(calls) == 1


def test_queue_never_follows_a_failed_prerequisite(tmp_path):
    spec = specification(tmp_path)
    state = tmp_path / "prior-guard.json"
    state.write_text('{"host_guard_passed":false,"child_exit_code":101,"unfinished_children":[]}')
    spec["prerequisites"] = [dict(guard_result=str(state), result="not-read", wait_seconds=1, expected={})]
    with pytest.raises(RuntimeError, match="failed prerequisite"):
        run(spec, tmp_path / "queue", execute=lambda *a, **kw: pytest.fail("spawned"))
