"""CPU-only guard checks: no NVIDIA device or native Kindle import."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "gpu_guard.py"
SPEC = importlib.util.spec_from_file_location("gpu_guard", SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)
UUID = "GPU-6869e50d-83aa-bec7-6169-adc413f49b32"
DRIVER = "595.91.07"
BOOT = "80351da1-04bb-4547-aab9-0b538ca01418"
HEALTH = f"{UUID}, 00000000:01:00.0, {DRIVER}, None, 0, 16303, 2, 15841, 462, 35, 31.5\n"


def journal(message="ordinary kernel message", boot=BOOT, cursor="cursor-1"):
    return json.dumps({"_BOOT_ID": boot.replace("-", ""), "__CURSOR": cursor,
                       "__MONOTONIC_TIMESTAMP": "100000", "MESSAGE": message}) + "\n"


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    boot = tmp_path / "boot-id"
    boot.write_text(BOOT)
    driver = tmp_path / "driver"
    driver.write_text(DRIVER)
    monkeypatch.setattr(guard, "BOOT", boot)
    monkeypatch.setattr(guard, "DRIVER", driver)
    return guard.Evidence(tmp_path / "evidence")


def test_health_requires_the_actual_uuid_driver_and_direct_free_memory():
    actual = guard.parse_health(HEALTH, UUID, DRIVER, 2048)
    assert actual["memory.free"] == 15841
    assert actual["memory.reserved"] == 462
    assert actual["temperature.gpu"] == "35"


@pytest.mark.parametrize("index,value", [
    (0, "GPU-wrong"), (2, "old-driver"), (3, "Reset"), (3, "Reboot"),
    (3, "[N/A]"), (4, "[N/A]"), (4, "nan"), (4, "inf"), (4, "-1"),
    (4, "101"), (5, "0"), (6, "999999"), (7, "2047"), (7, "[N/A]"),
    (8, "nan"), (8, "[N/A]"),
])
def test_bad_status_is_never_healthy(index, value):
    fields = HEALTH.strip().split(", ")
    fields[index] = value
    with pytest.raises(guard.GuardError):
        guard.parse_health(", ".join(fields), UUID, DRIVER, 2048)


@pytest.mark.parametrize("text", ["", "No devices were found\n", HEALTH + HEALTH, "a,b,c\n"])
def test_missing_or_extra_gpu_is_rejected(text):
    with pytest.raises(guard.GuardError):
        guard.parse_health(text, UUID, DRIVER, 2048)


def test_optional_telemetry_is_not_fabricated():
    text = HEALTH.replace("35, 31.5", "[N/A], [N/A]")
    assert guard.parse_health(text, UUID, DRIVER, 2048)["power.draw"] == "[N/A]"


@pytest.mark.parametrize("text", ["not-json", "{}", "null", "[]",
    journal(boot="wrong"), journal(cursor=""), journal(message=[255])])
def test_journal_is_fail_closed(text):
    with pytest.raises(guard.GuardError):
        guard.parse_journal(text, BOOT, 100)


def test_journal_limit_and_empty_delta():
    with pytest.raises(guard.GuardError, match="coverage"):
        guard.parse_journal(journal(), BOOT, 1)
    assert guard.parse_journal("", BOOT, 100) == []


@pytest.mark.parametrize("message", [
    "NVRM: Xid (PCI:0000:01:00): 62, payload",
    "NVRM: Xid (PCI:0000:01:00): 154, GPU Reset Required",
    "NVRM: Xid (PCI:0000:01:00): 109, CTX SWITCH TIMEOUT",
    "NVRM: Received signal from GSP that PMU has halted.",
    "NVRM: GPU 0000:01:00.0: RmInitAdapter failed! (0x62:0x40:2168)",
    "NVRM: GPU is probably locked!", "GSP task watchdog timeout",
    "NVRM: API mismatch: the client has the wrong version",
])
def test_first_kernel_fault_is_retained(evidence, monkeypatch, message):
    monkeypatch.setattr(evidence, "probe", lambda *a, **kw: journal(message))
    with pytest.raises(guard.GuardError, match="kernel fault"):
        guard.check_kernel(evidence, BOOT)
    event = json.loads((evidence.root / "events.jsonl").read_text())
    assert event["record"]["MESSAGE"] == message


def test_empty_baseline_is_not_a_clean_boot(evidence, monkeypatch):
    monkeypatch.setattr(evidence, "probe", lambda *a, **kw: "")
    with pytest.raises(guard.GuardError, match="baseline"):
        guard.check_kernel(evidence, BOOT)


def test_kernel_delta_uses_the_previous_cursor(evidence, monkeypatch):
    commands = []
    def probe(_name, command, **_kw):
        commands.append(command)
        return journal(cursor="cursor-2")
    monkeypatch.setattr(evidence, "probe", probe)
    assert guard.check_kernel(evidence, BOOT, "cursor-1") == "cursor-2"
    assert f"--boot={BOOT.replace('-', '')}" in commands[0]
    assert BOOT not in commands[0]  # --boot has an optional argument: bind it with '='.
    assert "--after-cursor=cursor-1" in commands[0]


@pytest.mark.parametrize("name,value", [("BOOT", "new-boot"), ("DRIVER", "new-driver")])
def test_changed_host_refuses_before_nvml(evidence, monkeypatch, name, value):
    getattr(guard, name).write_text(value)
    monkeypatch.setattr(evidence, "probe", lambda *a, **kw: pytest.fail("must not query NVML"))
    with pytest.raises(guard.GuardError, match="changed"):
        guard.check_health(evidence, BOOT, UUID, DRIVER, 2048)


def fake_host(evidence, monkeypatch, *, kernel_fault_at=None, nvml_fault_at=None):
    calls = {"kernel": 0, "nvml": 0, "snapshot": 0}
    def probe(name, _command, **_kw):
        key = "kernel" if name.startswith("kernel") else "nvml"
        calls[key] += 1
        if key == "kernel":
            if calls[key] == kernel_fault_at:
                return journal("NVRM: Xid (PCI:0000:01:00): 62, first fault")
            return journal(cursor=f"cursor-{calls[key]}")
        return HEALTH.replace(", None,", ", Reset,") if calls[key] == nvml_fault_at else HEALTH
    monkeypatch.setattr(evidence, "probe", probe)
    monkeypatch.setattr(evidence, "snapshot", lambda: calls.__setitem__("snapshot", calls["snapshot"] + 1))
    declaration = evidence.root.parent / "declaration.json"
    declaration.write_text('{"CPU_fixture": true}')
    return calls, declaration


def test_failed_preflight_never_spawns_the_cpu_sentinel(evidence, monkeypatch):
    calls, declaration = fake_host(evidence, monkeypatch, kernel_fault_at=1)
    sentinel = evidence.root / "should-not-exist"
    result = guard.run(evidence, [sys.executable, "-c", f"open({str(sentinel)!r}, 'x').close()"],
                       UUID, DRIVER, declaration, 1)
    assert not sentinel.exists()
    assert not result["child_spawned"] and not result["guard_passed"]
    assert calls == {"kernel": 1, "nvml": 0, "snapshot": 1}
    assert guard.audit(evidence.root)["verified_pins"] >= 3


@pytest.mark.parametrize("fault", ["kernel", "nvml"])
def test_live_fault_stops_only_our_child(evidence, monkeypatch, fault):
    calls, declaration = fake_host(evidence, monkeypatch,
        kernel_fault_at=3 if fault == "kernel" else None,
        nvml_fault_at=2 if fault == "nvml" else None)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        result = guard.run(evidence, [sys.executable, "-c", "import time; time.sleep(60)"],
                           UUID, DRIVER, declaration, 2, interval=0.01)
        assert result["child_spawned"] and not result["guard_passed"]
        assert result["child_exit_code"] is not None
        assert result["unfinished_children"] == []
        assert unrelated.poll() is None
        assert calls["snapshot"] == 1
        if fault == "kernel":
            assert calls["nvml"] == 1  # Never query a device again after its Xid.
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_healthy_cpu_job_and_postflight(evidence, monkeypatch):
    calls, declaration = fake_host(evidence, monkeypatch)
    result = guard.run(evidence, [sys.executable, "-c", "print('CPU fixture')"],
                       UUID, DRIVER, declaration, 2, interval=0.01)
    assert result["guard_passed"] and result["child_exit_code"] == 0
    assert calls["snapshot"] == 0 and calls["nvml"] >= 2
    assert guard.audit(evidence.root)["unsealed_files"] == []


def test_failed_cpu_job_captures_and_does_not_retry(evidence, monkeypatch):
    calls, declaration = fake_host(evidence, monkeypatch)
    result = guard.run(evidence, [sys.executable, "-c", "raise SystemExit(17)"],
                       UUID, DRIVER, declaration, 2, interval=0.01)
    assert result["child_exit_code"] == 17 and not result["guard_passed"]
    assert calls["snapshot"] == 1
    assert (evidence.root / "events.jsonl").read_text().count('"event": "child_spawned"') == 1
    with pytest.raises(FileExistsError):
        guard.Evidence(evidence.root)


def test_job_timeout_is_bounded(evidence, monkeypatch):
    _, declaration = fake_host(evidence, monkeypatch)
    result = guard.run(evidence, [sys.executable, "-c", "import time; time.sleep(60)"],
                       UUID, DRIVER, declaration, 0.03, interval=0.01)
    assert result["reason"] == "job time budget exceeded"
    assert result["child_exit_code"] is not None


@pytest.mark.parametrize("command,kwargs", [
    ([sys.executable, "-c", "raise SystemExit(7)"], {}),
    ([sys.executable, "-c", "import time; time.sleep(60)"], {"timeout": 0.05}),
    ([sys.executable, "-c", "print('x'*10000)"], {"limit": 100}),
    (["/definitely/not/a/program"], {}),
])
def test_probe_failures_leave_receipts(evidence, command, kwargs):
    with pytest.raises(guard.GuardError):
        evidence.probe("fixture", command, **kwargs)
    receipt = json.loads(next(evidence.root.glob("*-fixture.json")).read_text())
    assert receipt["error"] or receipt["exit_code"] != 0


def test_successful_polling_does_not_accumulate_files(evidence):
    for _ in range(3):
        assert evidence.probe("nvml", [sys.executable, "-c", "print('fabricated')"]).strip() == "fabricated"
    assert sorted(p.name for p in evidence.root.iterdir()) == ["events.jsonl", "start.json"]
    rows = [json.loads(line) for line in (evidence.root / "events.jsonl").read_text().splitlines()]
    assert [row["stdout"] for row in rows] == ["fabricated\n"] * 3


def test_audit_detects_changed_evidence(evidence):
    evidence.seal({"fixture": True})
    guard.audit(evidence.root)
    (evidence.root / "result.json").write_text("changed")
    with pytest.raises(guard.GuardError, match="changed"):
        guard.audit(evidence.root)


def test_snapshot_uses_only_host_read_commands(evidence, monkeypatch):
    commands = []
    monkeypatch.setattr(evidence, "probe", lambda name, command, **kw: commands.append(command) or "")
    evidence.snapshot()
    assert len(commands) == 6
    assert all(c[0] in {"journalctl", "lspci", "lsmod", "modinfo", "ps"} for c in commands)


def test_postflight_fault_is_not_a_successful_job(evidence, monkeypatch):
    _, declaration = fake_host(evidence, monkeypatch, kernel_fault_at=3)
    class CompletedChild:
        pid = 123
        returncode = 0

        def poll(self):
            return 0
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: CompletedChild())
    result = guard.run(evidence, [sys.executable, "-c", "pass"], UUID, DRIVER, declaration, 2)
    assert result["child_exit_code"] == 0
    assert not result["guard_passed"]
    assert "kernel fault" in result["reason"]


def test_unkillable_child_is_reported_not_waited_on_forever():
    class BlockedChild:
        signals = []

        def poll(self):
            return None

        def send_signal(self, value):
            self.signals.append(value)

        def wait(self, timeout):
            assert timeout == 2
            raise subprocess.TimeoutExpired("CPU fixture", timeout)
    child = BlockedChild()
    assert guard.stop_child(child) is None
    assert child.signals == [guard.signal.SIGTERM, guard.signal.SIGKILL]


def test_completed_child_is_never_signalled():
    class CompletedChild:
        returncode = 0

        def poll(self):
            return 0

        def send_signal(self, _sig):
            pytest.fail("must not signal an exited child")
    assert guard.stop_child(CompletedChild()) == 0


@pytest.mark.parametrize("timeout,interval,minimum", [(0, 1, 2048), (float("nan"), 1, 2048),
    (180000, 1, 2048), (1, 0, 2048), (1, 6, 2048), (1, 1, 2047)])
def test_invalid_budgets_refuse_before_any_probe(evidence, monkeypatch, timeout, interval, minimum):
    monkeypatch.setattr(evidence, "probe", lambda *a, **kw: pytest.fail("must refuse first"))
    with pytest.raises(ValueError):
        guard.run(evidence, [sys.executable], UUID, DRIVER, SOURCE, timeout, minimum, interval)


def test_unlisted_evidence_is_detected(evidence):
    evidence.seal({"fixture": True})
    (evidence.root / "unexpected").write_text("unaccounted")
    with pytest.raises(guard.GuardError, match="file set"):
        guard.audit(evidence.root)
