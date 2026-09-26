"""CPU-only host guard fixtures. No native Kindle import, GPU query or workload."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import gpu_host_guard as host

retained = host.retained
BOOT = "4f5152d1-e5fd-46cf-a0c4-06534c430d26"
DRIVER = "580.178.04"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    boot, driver = tmp_path / "boot", tmp_path / "driver"
    boot.write_text(BOOT)
    driver.write_text(DRIVER)
    monkeypatch.setattr(retained, "BOOT", boot)
    monkeypatch.setattr(retained, "DRIVER", driver)
    def forbidden(*_args, **_kwargs):
        pytest.fail("NVML or legacy run path must never be reached")
    monkeypatch.setattr(retained, "check_health", forbidden)
    monkeypatch.setattr(retained, "run", forbidden)
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    calls = dict(kernel=0, snapshot=0)
    controls = dict(fault_at=None, change_at=None, empty=False, unreadable=False, interrupted=False)
    def probe(evidence, name, command, **_kwargs):
        assert command[0] == "journalctl" and name.startswith("kernel")
        calls["kernel"] += 1
        if controls["unreadable"]:
            raise retained.GuardError("journal unreadable")
        if controls["interrupted"]:
            raise InterruptedError("host guard interrupted")
        if controls["change_at"] == calls["kernel"]:
            driver.write_text("changed")
        message = "NVRM: Xid (PCI:0000:01:00): 62, CPU fixture" if controls["fault_at"] == calls["kernel"] else "CPU fixture ordinary record"
        row = dict(_BOOT_ID=BOOT.replace("-", ""), __CURSOR=f'cursor-{calls["kernel"]}',
                   __MONOTONIC_TIMESTAMP="100000", MESSAGE=message)
        text = "" if controls["empty"] else json.dumps(row) + "\n"
        evidence.event("probe", name=name, stdout=text, stderr="", receipt=dict(command=command, exit_code=0, error=None))
        return text
    monkeypatch.setattr(host.HostEvidence, "probe", probe)
    monkeypatch.setattr(host.HostEvidence, "snapshot", lambda _self: calls.__setitem__("snapshot", calls["snapshot"] + 1))
    return tmp_path, calls, controls


def declare(path, code="print('CPU fixture')", **changes):
    config = dict(monitoring="host-only", boot_id=BOOT, driver=DRIVER,
                  command=[sys.executable, "-B", "-c", code], executable_sha256=retained.fingerprint(sys.executable),
                  timeout_seconds=2, poll_seconds=.01)
    config.update(changes)
    path.write_text(json.dumps(dict(host_guard=config)))
    return path


def execute(environment, **changes):
    path, _calls, _controls = environment
    return host.run(path / "evidence", declare(path / "declaration.json", **changes))


def test_cpu_child_passes_without_telemetry_or_legacy_guard(environment):
    path, calls, _controls = environment
    result = execute(environment)
    assert result["host_guard_passed"] and result["child_exit_code"] == 0
    assert result["guard_nvml_queries"] == 0 and result["gpu_telemetry"] == "unmeasured"
    assert result["hardware_qualified"] is False and not result["unfinished_children"]
    assert calls["kernel"] >= 2 and calls["snapshot"] == 0
    assert host.audit(path / "evidence")["host_guard_passed"]
    assert "guard_passed" not in result  # Cannot masquerade as the historical telemetry guard.
    with pytest.raises(FileExistsError):
        host.run(path / "evidence", path / "declaration.json")


@pytest.mark.parametrize("control,value", [("fault_at", 1), ("empty", True), ("unreadable", True),
                                         ("change_at", 1), ("interrupted", True)])
def test_bad_baseline_never_spawns(environment, control, value):
    path, calls, controls = environment
    controls[control] = value
    result = execute(environment)
    assert not result["host_guard_passed"] and not result["child_spawned"]
    assert not (path / "evidence/child.stdout").exists()
    assert calls == dict(kernel=1, snapshot=1)
    assert not host.audit(path / "evidence")["host_guard_passed"]


@pytest.mark.parametrize("identity,value", [("BOOT", "new-boot"), ("DRIVER", "595.91.07")])
def test_changed_identity_refuses_before_journal(environment, identity, value):
    _path, calls, _controls = environment
    getattr(retained, identity).write_text(value)
    result = execute(environment)
    assert result["reason"] == "boot or loaded driver changed"
    assert calls == dict(kernel=0, snapshot=1) and not result["child_spawned"]


@pytest.mark.parametrize("control", ["fault_at", "change_at"])
def test_live_fault_reaps_only_direct_child(environment, control):
    path, calls, controls = environment
    controls[control] = 2
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = execute(environment, code="import time; time.sleep(30)")
        assert result["child_spawned"] and result["child_exit_code"] is not None
        assert not result["host_guard_passed"] and not result["unfinished_children"]
        assert calls == dict(kernel=2, snapshot=1) and unrelated.poll() is None
        assert not host.audit(path / "evidence")["host_guard_passed"]
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_cpu_timeout_is_bounded_and_not_retried(environment):
    path, calls, _controls = environment
    started = time.monotonic()
    result = execute(environment, code="import time; time.sleep(30)", timeout_seconds=.03)
    assert time.monotonic() - started < 3
    assert result["reason"] == "job time budget exceeded" and result["child_exit_code"] is not None
    assert calls["snapshot"] == 1 and not result["unfinished_children"]
    assert (path / "evidence/events.jsonl").read_text().count('"event": "child_spawned"') == 1


def test_native_failure_remains_a_failure(environment):
    result = execute(environment, code="raise SystemExit(17)")
    assert result["child_exit_code"] == 17 and result["reason"] == "native job exited 17"
    assert not result["host_guard_passed"]


def test_postflight_fault_cannot_be_a_pass(environment, monkeypatch):
    _path, _calls, controls = environment
    controls["fault_at"] = 2
    class Completed:
        pid, returncode = 123, 0
        def poll(self):
            return 0
    monkeypatch.setattr(host.subprocess, "Popen", lambda *a, **kw: Completed())
    result = execute(environment)
    assert result["child_exit_code"] == 0 and not result["host_guard_passed"]
    assert "kernel fault" in result["reason"]


def test_unkillable_cpu_fake_stays_unfinished(environment, monkeypatch):
    path, _calls, _controls = environment
    class Blocked:
        pid, returncode = 123, None
        signals = []
        def poll(self):
            return None
        def send_signal(self, value):
            self.signals.append(value)
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("CPU fake", timeout)
    process = Blocked()
    monkeypatch.setattr(host.subprocess, "Popen", lambda *a, **kw: process)
    result = execute(environment, timeout_seconds=.001)
    assert process.signals == [host.signal.SIGTERM, host.signal.SIGKILL]
    assert result["child_exit_code"] is None and not result["host_guard_passed"]
    assert result["unfinished_children"] == [dict(pid=123, files=["child.stdout", "child.stderr"])]
    assert set(host.audit(path / "evidence")["unsealed_files"]) == {"child.stdout", "child.stderr"}


@pytest.mark.parametrize("changes", [dict(monitoring="nvml"), dict(boot_id="wrong"), dict(driver=""),
    dict(command=[]), dict(command=["relative-executable"]), dict(command=[sys.executable, 1]),
    dict(command=["/usr/bin/nvidia-smi"]), dict(command=[sys.executable, "\0"]),
    dict(executable_sha256="0" * 64), dict(timeout_seconds=0), dict(timeout_seconds=True),
    dict(timeout_seconds=float("nan")), dict(timeout_seconds=172801), dict(poll_seconds=0),
    dict(poll_seconds=5.01), dict(poll_seconds=float("inf"))])
def test_bad_declaration_refuses_before_outputs_or_probes(environment, changes):
    path, calls, _controls = environment
    with pytest.raises((ValueError, TypeError)):
        execute(environment, **changes)
    assert calls == dict(kernel=0, snapshot=0) and not (path / "evidence").exists()


@pytest.mark.parametrize("text", ['{"old_declaration":true}', '{"host_guard":null}',
    '{"host_guard":{},"host_guard":{}}'])
def test_legacy_or_ambiguous_declaration_cannot_be_reused(environment, text):
    path, calls, _controls = environment
    declaration = path / "old.json"
    declaration.write_text(text)
    with pytest.raises((ValueError, KeyError)):
        host.run(path / "evidence", declaration)
    assert calls == dict(kernel=0, snapshot=0) and not (path / "evidence").exists()


@pytest.mark.parametrize("command", [["nvidia-smi"], ["/usr/bin/nvidia-smi"], ["python3"], []])
def test_real_probe_dispatch_rejects_non_host_tools(tmp_path, monkeypatch, command):
    monkeypatch.setattr(retained.Evidence, "probe", lambda *a, **kw: pytest.fail("forbidden dispatch"))
    evidence = host.HostEvidence(tmp_path / "probe")
    with pytest.raises(retained.GuardError, match="non-host"):
        evidence.probe("fixture", command)


def test_snapshot_uses_only_the_host_allowlist(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(retained.Evidence, "probe", lambda self, name, command, **kw: commands.append(command) or "")
    evidence = host.HostEvidence(tmp_path / "snapshot")
    evidence.snapshot()
    assert len(commands) == 6 and all(command[0] in {"journalctl", "lspci", "lsmod", "modinfo", "ps"} for command in commands)


def test_audit_detects_tampering(environment):
    path, _calls, _controls = environment
    execute(environment)
    (path / "evidence/result.json").write_text("{}")
    with pytest.raises(retained.GuardError, match="changed"):
        host.audit(path / "evidence")


def test_import_does_not_load_nvml():
    code = f"""
import ctypes, sys
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected library load')
ctypes.CDLL = forbidden
sys.path.insert(0, {str(EXAMPLES)!r})
import gpu_host_guard
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=5)
