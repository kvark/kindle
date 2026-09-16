"""CPU fakes only. Never load NVML, import Kindle or open a GPU device."""

import ctypes as ct
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import gpu_guard as guard
import gpu_health as reader
from gpu_health_client import HealthClient

UUID, DRIVER, PCI = "GPU-cpu-fixture", "580.178.04", "00000000:01:00.0"
MIB = 1024 * 1024


class Function:
    def __init__(self, library, name, function):
        self.library, self.name, self.function = library, name, function

    def __call__(self, *args):
        self.library.calls.append(self.name)
        if self.library.trace:
            with self.library.trace.open("a") as target:
                target.write(self.name + "\n")
        if self.name in self.library.errors:
            return self.library.errors[self.name]
        self.function(*args)
        return 0


class FakeLibrary:
    def __init__(self, trace=None):
        self.calls, self.errors, self.trace = [], {}, trace
        self.driver, self.uuid, self.pci = DRIVER, UUID, PCI
        self.field = dict(fieldId=230, scopeId=0, timestamp=1789536600000000,
                          latencyUsec=25, valueType=1, nvmlReturn=0)
        self.recovery, self.utilization, self.handle = 0, 17, 1
        self.memory = dict(version=reader.MEMORY_VERSION, total=16303 * MIB + 511,
                           free=15835 * MIB + 128, used=4 * MIB, reserved=464 * MIB)
        for name in dir(type(self)):
            if name.startswith("nvml"):
                setattr(self, name, Function(self, name, getattr(self, name)))

    def nvmlInit_v2(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlSystemGetDriverVersion(self, output, size):
        assert size >= len(self.driver) + 1
        output.value = self.driver.encode()

    def nvmlDeviceGetHandleByUUID(self, uuid, output):
        assert uuid == UUID.encode()
        ct.cast(output, ct.POINTER(ct.c_void_p)).contents.value = self.handle

    def nvmlDeviceGetUUID(self, _device, output, size):
        assert size >= len(self.uuid) + 1
        output.value = self.uuid.encode()

    def nvmlDeviceGetPciInfo_v3(self, _device, output):
        ct.cast(output, ct.POINTER(reader.PciInfo)).contents.busId = self.pci.encode()

    def nvmlDeviceGetFieldValues(self, _device, count, output):
        field = ct.cast(output, ct.POINTER(reader.FieldValue)).contents
        assert count == 1 and field.fieldId == 230 and field.scopeId == 0
        for name, value in self.field.items():
            setattr(field, name, value)
        member = reader.INTEGER_FIELDS.get(field.valueType, "dVal")
        setattr(field.value, member, self.recovery)

    def nvmlDeviceGetMemoryInfo_v2(self, _device, output):
        memory = ct.cast(output, ct.POINTER(reader.Memory)).contents
        assert memory.version == reader.MEMORY_VERSION
        for name, value in self.memory.items():
            setattr(memory, name, value)

    def nvmlDeviceGetUtilizationRates(self, _device, output):
        ct.cast(output, ct.POINTER(reader.Utilization)).contents.gpu = self.utilization

    def nvmlDeviceGetTemperature(self, _device, sensor, output):
        assert sensor == 0
        ct.cast(output, ct.POINTER(ct.c_uint)).contents.value = 30

    def nvmlDeviceGetPowerUsage(self, _device, output):
        ct.cast(output, ct.POINTER(ct.c_uint)).contents.value = 25020


def session(library=None):
    return reader.NvmlSession(library or FakeLibrary(), UUID, DRIVER, PCI)


def sample():
    source = session()
    try:
        return source.sample()
    finally:
        source.close()


def test_import_does_not_load_nvml():
    code = f"""
import ctypes, sys
def forbidden(*args, **kwargs):
    raise AssertionError('library load on import')
ctypes.CDLL = forbidden
sys.path.insert(0, {str(EXAMPLES)!r})
import gpu_health, gpu_health_client
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=10)


def test_local_header_abi_without_nvml_linkage(tmp_path):
    binary = tmp_path / "abi"
    source = Path(__file__).parent / "fixtures/gpu_health_abi.c"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
                   check=True, timeout=20)
    needed = subprocess.run(["readelf", "-d", str(binary)], capture_output=True, text=True, check=True).stdout
    assert "nvidia" not in needed.lower() and "nvml" not in needed.lower()
    actual = json.loads(subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout)
    expected = dict(memory_version=reader.MEMORY_VERSION, integers=list(reader.INTEGER_FIELDS))
    for name, struct in (("nvmlPciInfo_t", reader.PciInfo), ("nvmlMemory_v2_t", reader.Memory),
                         ("nvmlUtilization_t", reader.Utilization), ("nvmlValue_t", reader.Value),
                         ("nvmlFieldValue_t", reader.FieldValue)):
        expected[name] = ct.sizeof(struct)
        expected.update({name + "." + field: getattr(struct, field).offset for field, _ in struct._fields_})
    assert actual == expected


@pytest.mark.parametrize("kind", reader.INTEGER_FIELDS)
def test_reuses_one_session_and_checks_integer_recovery_types(kind):
    library = FakeLibrary()
    library.field["valueType"] = kind
    source = session(library)
    for _ in range(3):
        row = source.sample()
        assert row["health"]["memory.free"] == 15835
        assert row["memory_bytes"]["free"] == 15835 * MIB + 128
        assert row["health"]["power.draw"] == "25.02"
    source.close()
    source.close()
    assert library.calls.count("nvmlInit_v2") == library.calls.count("nvmlShutdown") == 1
    assert library.calls.count("nvmlDeviceGetFieldValues") == 3
    with pytest.raises(guard.GuardError, match="closed"):
        source.sample()


@pytest.mark.parametrize("recovery", [-1, 1, 2, 3, 4, 5, 100])
def test_recovery_stops_queries_and_cannot_be_retried(recovery):
    library = FakeLibrary()
    library.field["valueType"] = 5
    library.recovery = recovery
    source = session(library)
    begin = len(library.calls)
    with pytest.raises(guard.GuardError, match="recovery action"):
        source.sample()
    with pytest.raises(guard.GuardError, match="failed"):
        source.sample()
    source.close()
    assert library.calls[begin:] == ["nvmlDeviceGetFieldValues", "nvmlShutdown"]


@pytest.mark.parametrize("name,value", [("fieldId", 99), ("scopeId", 1), ("valueType", 0),
    ("valueType", 99), ("nvmlReturn", 3), ("nvmlReturn", 15), ("timestamp", 0), ("latencyUsec", -1)])
def test_bad_recovery_fields_stop_before_other_queries(name, value):
    library = FakeLibrary()
    library.field[name] = value
    source = session(library)
    begin = len(library.calls)
    with pytest.raises(guard.GuardError):
        source.sample()
    source.close()
    assert library.calls[begin:] == ["nvmlDeviceGetFieldValues", "nvmlShutdown"]


@pytest.mark.parametrize("name", [name for name in dir(FakeLibrary) if name.startswith("nvml") and name != "nvmlShutdown"])
def test_each_api_error_is_fatal(name):
    library = FakeLibrary()
    library.errors[name] = 15
    source = None
    with pytest.raises(guard.GuardError, match=name):
        source = session(library)
        source.sample()
    if source:
        source.close()
    assert library.calls.count(name) == 1
    assert library.calls[-1] == ("nvmlInit_v2" if name == "nvmlInit_v2" else "nvmlShutdown")


@pytest.mark.parametrize("attribute,value", [("uuid", "wrong"), ("driver", "wrong"),
    ("pci", "00000000:02:00.0"), ("utilization", 101)])
def test_identity_and_utilization_fail_closed(attribute, value):
    library = FakeLibrary()
    source = session(library)
    setattr(library, attribute, value)
    with pytest.raises(guard.GuardError):
        source.sample()
    source.close()


@pytest.mark.parametrize("attribute,value", [("version", 0), ("total", 0),
    ("free", 2048 * MIB - 1), ("used", 2**64 - 1), ("reserved", 2**64 - 1)])
def test_invalid_or_insufficient_memory_is_fatal(attribute, value):
    library = FakeLibrary()
    source = session(library)
    library.memory[attribute] = value
    with pytest.raises(guard.GuardError):
        source.sample()
    source.close()


def test_optional_fields_only_allow_not_supported():
    library = FakeLibrary()
    library.errors.update(nvmlDeviceGetTemperature=3, nvmlDeviceGetPowerUsage=3)
    source = session(library)
    row = source.sample()["health"]
    source.close()
    assert row["temperature.gpu"] == row["power.draw"] == "[N/A]"


def test_null_handle_and_cleanup_failure_retain_both_errors():
    library = FakeLibrary()
    library.handle = None
    library.errors["nvmlShutdown"] = 15
    with pytest.raises(guard.GuardError, match="null NVML device handle; cleanup failed: nvmlShutdown"):
        session(library)
    assert library.calls[-1] == "nvmlShutdown"


@pytest.mark.parametrize("payload", ["{}\n", "null\n", "not-json\n", '{"sequence":true,"operation":"sample"}\n',
    '{"sequence":2,"operation":"sample"}\n', '{"sequence":1,"operation":"retry"}\n',
    '{"sequence":1,"operation":"sample","operation":"sample"}\n', '{"sequence":NaN,"operation":"sample"}\n',
    "x" * 4097 + "\n", '{"sequence":1,"operation":"sample"}'])
def test_invalid_request_never_initializes(payload):
    output = io.StringIO()
    def forbidden():
        pytest.fail("invalid request initialized NVML")
    assert reader.serve(io.StringIO(payload), output, forbidden) == 1
    assert "error" in json.loads(output.getvalue())


def test_ordered_protocol_stops_on_first_fault_even_with_buffered_requests():
    library = FakeLibrary()
    library.recovery = 1
    output = io.StringIO()
    requests = '\n'.join(json.dumps(dict(sequence=i, operation="sample")) for i in (1, 2, 3)) + '\n'
    assert reader.serve(io.StringIO(requests), output, lambda: session(library)) == 1
    assert len(output.getvalue().splitlines()) == 1
    assert library.calls.count("nvmlDeviceGetFieldValues") == library.calls.count("nvmlShutdown") == 1


def test_empty_input_never_creates_session():
    assert reader.serve(io.StringIO(), io.StringIO(), lambda: pytest.fail("idle init")) == 0


@pytest.mark.parametrize("section,key,value", [
    ("health", "memory.free", True), ("health", "memory.free", 15835.0),
    ("health", "utilization.gpu", True), ("health", "gpu_recovery_action", "Reset"),
    ("health", "temperature.gpu", "nan"), ("health", "power.draw", "-1"),
    ("health", "uuid", UUID + "\n"), ("memory_bytes", "free", True),
    ("memory_bytes", "free", -1), ("memory_bytes", "total", 2**64),
    ("recovery_field", "value", False), ("recovery_field", "timestamp_us", 0),
    ("recovery_field", "latency_us", -1),
])
def test_sample_validation_rejects_coercions_and_corrupt_metadata(section, key, value):
    row = sample()
    row[section][key] = value
    with pytest.raises((guard.GuardError, ValueError)):
        reader.validate_sample(row, UUID, DRIVER, PCI)


def worker_command(mode, path):
    return [sys.executable, "-B", str(Path(__file__).resolve()), "--cpu-worker", mode, str(path)]


def client(tmp_path, mode="normal", check_host=lambda: None, **budgets):
    return HealthClient(worker_command(mode, tmp_path / "calls"), UUID, DRIVER, PCI,
                        check_host, subprocess.DEVNULL, **budgets)


def calls(tmp_path):
    path = tmp_path / "calls"
    return path.read_text().splitlines() if path.exists() else []


def test_client_idle_and_close_do_not_query(tmp_path):
    source = client(tmp_path)
    assert source.close() == 0
    assert calls(tmp_path) == []


def test_client_requests_are_ordered_and_one_init_is_reused(tmp_path):
    checks = []
    source = client(tmp_path, check_host=lambda: checks.append(True))
    try:
        for sequence in (1, 2, 3):
            row = source.sample()
            assert row["response"]["sequence"] == sequence
            assert row["sent_ns"] <= row["received_ns"] <= row["accepted_ns"]
            assert row["health"]["memory.free"] == 15835
    finally:
        assert source.close() == 0
    assert len(checks) == 6
    assert calls(tmp_path).count("nvmlInit_v2") == calls(tmp_path).count("nvmlShutdown") == 1
    assert calls(tmp_path).count("nvmlDeviceGetFieldValues") == 3


@pytest.mark.parametrize("when", [1, 2, 3])
def test_host_fault_prevents_any_further_request(tmp_path, when):
    count = 0
    def host():
        nonlocal count
        count += 1
        if count == when:
            raise guard.GuardError("fabricated host fault")
    source = client(tmp_path, check_host=host)
    with pytest.raises(guard.GuardError, match="host fault"):
        source.sample()
        source.sample()
    assert source.process.poll() is not None
    with pytest.raises(guard.GuardError, match="closed"):
        source.sample()
    assert calls(tmp_path).count("nvmlDeviceGetFieldValues") == (0 if when == 1 else 1)


def test_late_next_request_is_refused_without_query(tmp_path):
    source = client(tmp_path)
    source.sample()
    source.last_ns -= 2_000_000_000
    with pytest.raises(guard.GuardError, match="before request"):
        source.sample()
    assert calls(tmp_path).count("nvmlDeviceGetFieldValues") == 1
    assert source.process.poll() is not None


def test_host_check_delay_is_included_in_the_gap(tmp_path):
    count = 0
    def host():
        nonlocal count
        count += 1
        if count == 4:
            time.sleep(.15)
    source = client(tmp_path, check_host=host, max_gap=.1)
    source.sample()
    with pytest.raises(guard.GuardError, match="after host check"):
        source.sample()
    assert source.process.poll() is not None
    assert calls(tmp_path).count("nvmlDeviceGetFieldValues") == 2


def test_first_request_has_a_bounded_bootstrap(tmp_path):
    source = client(tmp_path, "stall-first", bootstrap_timeout=.15)
    with pytest.raises(guard.GuardError, match="deadline"):
        source.sample()
    assert source.process.poll() is not None


def test_closed_worker_is_not_reported_as_reaped_when_stop_fails(monkeypatch):
    class Process:
        stdin, stdout = io.BytesIO(), io.BytesIO()
        def poll(self):
            return None
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("CPU fake", timeout)
    source = HealthClient.__new__(HealthClient)
    source.closed, source.process = False, Process()
    monkeypatch.setattr(guard, "stop_child", lambda process: None)
    assert source.close() is None
    assert source.close() is None
    with pytest.raises(guard.GuardError, match="closed"):
        source.sample()


def test_unsolicited_response_is_rejected_before_request(tmp_path):
    source = client(tmp_path, "unsolicited")
    assert __import__("select").select([source.process.stdout], [], [], 3)[0]
    with pytest.raises(guard.GuardError, match="unsolicited"):
        source.sample()
    assert calls(tmp_path) == [] and source.process.poll() is not None


def test_declared_memory_reserve_cannot_be_lowered_by_worker(tmp_path):
    source = client(tmp_path, minimum_free=15836)
    with pytest.raises(guard.GuardError, match="insufficient"):
        source.sample()
    assert source.process.poll() is not None


def test_blocked_worker_is_reaped_without_touching_unrelated_child(tmp_path):
    source = client(tmp_path, "stall-second", max_gap=.15)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        source.sample()
        started = time.monotonic()
        with pytest.raises(guard.GuardError, match="deadline"):
            source.sample()
        assert time.monotonic() - started < 5
        assert source.process.poll() is not None and unrelated.poll() is None
        assert calls(tmp_path) == ["request", "request"]
    finally:
        source.close()
        unrelated.terminate()
        unrelated.wait(timeout=3)


@pytest.mark.parametrize("mode", ["bad-sequence", "boolean-sequence", "bad-clock", "missing-field",
    "bad-bytes", "bad-recovery", "extra-key", "duplicate-key", "nan", "multi-line", "oversized", "eof", "error"])
def test_client_rejects_bad_responses_and_reaps(tmp_path, mode):
    source = client(tmp_path, mode)
    with pytest.raises((guard.GuardError, ValueError)):
        source.sample()
    assert source.process.poll() is not None
    assert calls(tmp_path) == ["request"]


@pytest.mark.parametrize("budget", [dict(minimum_free=2047), dict(minimum_free=float("nan")),
    dict(minimum_free=True), dict(max_gap=1.500001), dict(max_gap=0), dict(max_gap=True),
    dict(max_gap=float("nan")), dict(bootstrap_timeout=3.001), dict(bootstrap_timeout=float("inf"))])
def test_invalid_budgets_never_spawn(tmp_path, budget):
    with pytest.raises(ValueError):
        client(tmp_path, **budget)
    assert not (tmp_path / "calls").exists()


def cpu_worker(mode, path):
    if mode == "normal":
        return reader.serve(sys.stdin, sys.stdout, lambda: session(FakeLibrary(path)))
    if mode == "unsolicited":
        print("{}", flush=True)
    for line in sys.stdin:
        started = time.monotonic_ns()
        request = json.loads(line)
        with path.open("a") as target:
            target.write("request\n")
        if mode == "stall-first" or (mode == "stall-second" and request["sequence"] == 2):
            time.sleep(60)
        if mode == "eof":
            return 0
        row = dict(sequence=request["sequence"], started_ns=started,
                   finished_ns=time.monotonic_ns(), sample=sample())
        if mode == "bad-sequence": row["sequence"] += 1
        if mode == "boolean-sequence": row["sequence"] = True
        if mode == "bad-clock": row["started_ns"] = 0
        if mode == "missing-field": del row["sample"]["health"]["memory.free"]
        if mode == "bad-bytes": row["sample"]["memory_bytes"]["free"] += MIB
        if mode == "bad-recovery": row["sample"]["recovery_field"]["value"] = 1
        if mode == "extra-key": row["extra"] = 1
        if mode == "error": row = dict(sequence=request["sequence"], error="CPU fixture failure")
        text = json.dumps(row)
        if mode == "duplicate-key": text = text.replace('"sequence": 1', '"sequence": 1, "sequence": 1')
        if mode == "nan": text = text.replace('"sequence": 1', '"sequence": NaN')
        if mode == "multi-line": text += "\n" + text
        if mode == "oversized": text = "x" * 65537
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    assert sys.argv[1] == "--cpu-worker"
    raise SystemExit(cpu_worker(sys.argv[2], Path(sys.argv[3])))
