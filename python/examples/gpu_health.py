"""Experimental request-driven NVML reader. Importing this module never loads NVML.

Not wired into gpu_guard.py or any declared job. Hardware use needs its own
approval, library/source pins, field-parity checks and cadence qualification.
"""

import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import sys
import time

import gpu_guard as guard


class PciInfo(ct.Structure):
    _fields_ = [("busIdLegacy", ct.c_char * 16), ("domain", ct.c_uint),
                ("bus", ct.c_uint), ("device", ct.c_uint), ("pciDeviceId", ct.c_uint),
                ("pciSubSystemId", ct.c_uint), ("busId", ct.c_char * 32)]


class Memory(ct.Structure):
    _fields_ = [("version", ct.c_uint), ("total", ct.c_ulonglong), ("reserved", ct.c_ulonglong),
                ("free", ct.c_ulonglong), ("used", ct.c_ulonglong)]


class Utilization(ct.Structure):
    _fields_ = [("gpu", ct.c_uint), ("memory", ct.c_uint)]


class Value(ct.Union):
    _fields_ = [("dVal", ct.c_double), ("siVal", ct.c_int), ("uiVal", ct.c_uint),
                ("ulVal", ct.c_ulong), ("ullVal", ct.c_ulonglong), ("sllVal", ct.c_longlong)]


class FieldValue(ct.Structure):
    _fields_ = [("fieldId", ct.c_uint), ("scopeId", ct.c_uint), ("timestamp", ct.c_longlong),
                ("latencyUsec", ct.c_longlong), ("valueType", ct.c_uint),
                ("nvmlReturn", ct.c_uint), ("value", Value)]


RECOVERY_FIELD = 230  # NVML_FI_DEV_GET_GPU_RECOVERY_ACTION; absent in the older local SDK.
MEMORY_VERSION = ct.sizeof(Memory) | (2 << 24)
INTEGER_FIELDS = {1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal", 5: "siVal"}


def require(ok, reason):
    if not ok:
        raise guard.GuardError(reason)


def decode(text):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result
    def constant(_value):
        raise guard.GuardError("non-finite JSON value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_sample(sample, uuid, driver, pci, minimum_free=2048):
    require(isinstance(sample, dict) and set(sample) == {"health", "memory_bytes", "recovery_field"},
            "invalid sample schema")
    health, raw, recovery = (sample[name] for name in ("health", "memory_bytes", "recovery_field"))
    require(isinstance(health, dict) and set(health) == set(guard.FIELDS), "health fields differ")
    require(isinstance(raw, dict) and set(raw) == {"total", "used", "free", "reserved"}, "memory fields differ")
    for name, value in raw.items():
        require(type(value) is int and 0 <= value < 2**64, "invalid memory bytes")
        counter = health["memory." + name]
        require(type(counter) is int and counter == value // (1024 * 1024), "memory conversion differs")
    require(raw["total"] > 0 and all(value <= raw["total"] for value in raw.values()), "invalid memory counters")
    require(type(health["utilization.gpu"]) is int, "invalid utilization type")
    for name in ("uuid", "driver_version", "pci.bus_id", "gpu_recovery_action", "temperature.gpu", "power.draw"):
        value = health[name]
        require(isinstance(value, str) and not any(char in value for char in ',\r\n"'), "invalid health text")
    for name in ("temperature.gpu", "power.draw"):
        value = health[name]
        require(value == "[N/A]" or (math.isfinite(float(value)) and float(value) >= 0), "invalid optional telemetry")
    require(isinstance(recovery, dict) and set(recovery) == {
        "field_id", "value_type", "timestamp_us", "latency_us", "value"}, "recovery fields differ")
    require(all(type(value) is int for value in recovery.values()) and recovery["field_id"] == RECOVERY_FIELD
            and recovery["value_type"] in INTEGER_FIELDS and recovery["value"] == 0
            and recovery["timestamp_us"] > 0 and recovery["latency_us"] >= 0, "invalid recovery field")
    parsed = guard.parse_health(",".join(str(health[name]) for name in guard.FIELDS), uuid, driver, minimum_free)
    require(parsed["pci.bus_id"].lower() == pci.lower(), "health PCI adapter changed")
    return parsed


class NvmlSession:
    def __init__(self, library, uuid, driver, pci):
        self.library, self.uuid, self.driver, self.pci = library, uuid, driver, pci.lower()
        self.device = ct.c_void_p()
        self.initialized = False
        self.failed = False
        pointer = ct.POINTER
        signatures = {
            "nvmlInit_v2": [], "nvmlShutdown": [],
            "nvmlDeviceGetHandleByUUID": [ct.c_char_p, pointer(ct.c_void_p)],
            "nvmlSystemGetDriverVersion": [pointer(ct.c_char), ct.c_uint],
            "nvmlDeviceGetUUID": [ct.c_void_p, pointer(ct.c_char), ct.c_uint],
            "nvmlDeviceGetPciInfo_v3": [ct.c_void_p, pointer(PciInfo)],
            "nvmlDeviceGetFieldValues": [ct.c_void_p, ct.c_int, pointer(FieldValue)],
            "nvmlDeviceGetMemoryInfo_v2": [ct.c_void_p, pointer(Memory)],
            "nvmlDeviceGetUtilizationRates": [ct.c_void_p, pointer(Utilization)],
            "nvmlDeviceGetTemperature": [ct.c_void_p, ct.c_uint, pointer(ct.c_uint)],
            "nvmlDeviceGetPowerUsage": [ct.c_void_p, pointer(ct.c_uint)],
        }
        for name, arguments in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ct.c_uint
        try:
            self.call("nvmlInit_v2")
            self.initialized = True
            require(self.driver_string() == driver, "NVML driver changed")
            self.call("nvmlDeviceGetHandleByUUID", uuid.encode("ascii"), ct.byref(self.device))
            require(self.device.value is not None, "null NVML device handle")
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup:
                raise guard.GuardError(f"{error}; cleanup failed: {cleanup}") from error
            raise

    def call(self, name, *arguments):
        result = getattr(self.library, name)(*arguments)
        require(result == 0, f"{name} failed: NVML error {result}")

    def driver_string(self):
        output = ct.create_string_buffer(80)
        self.call("nvmlSystemGetDriverVersion", output, len(output))
        return output.value.decode("ascii")

    def optional_uint(self, name, *arguments):
        output = ct.c_uint()
        result = getattr(self.library, name)(self.device, *arguments, ct.byref(output))
        if result == 3:  # NVML_ERROR_NOT_SUPPORTED is allowed only for optional telemetry.
            return None
        require(result == 0, f"{name} failed: NVML error {result}")
        return output.value

    def sample(self):
        require(self.initialized and not self.failed, "NVML session is closed or failed")
        try:
            return self._sample()
        except Exception:
            self.failed = True
            raise

    def _sample(self):
        # Check recovery first. A nonzero/unknown/error value prevents further queries.
        field = FieldValue(fieldId=RECOVERY_FIELD)
        self.call("nvmlDeviceGetFieldValues", self.device, 1, ct.byref(field))
        require(field.nvmlReturn == 0, f"recovery field failed: NVML error {field.nvmlReturn}")
        require(field.fieldId == RECOVERY_FIELD and field.scopeId == 0
                and field.valueType in INTEGER_FIELDS, "invalid recovery field identity/type")
        recovery = getattr(field.value, INTEGER_FIELDS[field.valueType])
        require(recovery == 0, f"GPU recovery action is {recovery}")
        require(field.timestamp > 0 and field.latencyUsec >= 0, "invalid recovery field clock")
        driver = self.driver_string()
        require(driver == self.driver, "NVML driver changed")
        uuid = ct.create_string_buffer(96)
        self.call("nvmlDeviceGetUUID", self.device, uuid, len(uuid))
        require(uuid.value.decode("ascii") == self.uuid, "NVML UUID changed")
        pci = PciInfo()
        self.call("nvmlDeviceGetPciInfo_v3", self.device, ct.byref(pci))
        bus = bytes(pci.busId).decode("ascii").lower()
        require(bus == self.pci, "NVML PCI adapter changed")
        memory = Memory(version=MEMORY_VERSION)
        self.call("nvmlDeviceGetMemoryInfo_v2", self.device, ct.byref(memory))
        require(memory.version == MEMORY_VERSION, "NVML memory ABI changed")
        utilization = Utilization()
        self.call("nvmlDeviceGetUtilizationRates", self.device, ct.byref(utilization))
        temperature = self.optional_uint("nvmlDeviceGetTemperature", 0)
        power = self.optional_uint("nvmlDeviceGetPowerUsage")
        raw = {name: getattr(memory, name) for name in ("total", "used", "free", "reserved")}
        health = dict(uuid=self.uuid, driver_version=driver, gpu_recovery_action="None",
                      **{"pci.bus_id": bus, "utilization.gpu": utilization.gpu,
                         "temperature.gpu": "[N/A]" if temperature is None else str(temperature),
                         "power.draw": "[N/A]" if power is None else str(power / 1000)})
        health.update({"memory." + name: value // (1024 * 1024) for name, value in raw.items()})
        # Floor byte counts: never overstate directly free memory or fabricate N/A.
        sample = dict(health=health, memory_bytes=raw,
                      recovery_field=dict(field_id=field.fieldId, value_type=field.valueType,
                                          timestamp_us=field.timestamp, latency_us=field.latencyUsec, value=recovery))
        validate_sample(sample, self.uuid, driver, self.pci)
        return sample

    def close(self):
        if self.initialized:
            self.initialized = False
            self.call("nvmlShutdown")


def serve(source, target, factory):
    """One synchronous sample per ordered request. No timer, prefetch or retry."""
    session, previous = None, 0
    try:
        while True:
            line = source.readline(4097)
            if not line:
                return 0
            started = time.monotonic_ns()
            sequence = None
            try:
                require(len(line) <= 4096 and line.endswith("\n"), "invalid request framing")
                request = decode(line)
                require(isinstance(request, dict) and set(request) == {"sequence", "operation"}, "invalid request")
                sequence = request["sequence"]
                require(type(sequence) is int and sequence == previous + 1
                        and request["operation"] == "sample", "invalid request sequence/operation")
                if session is None:
                    session = factory()
                sample = session.sample()
                response = dict(sequence=sequence, started_ns=started, finished_ns=time.monotonic_ns(), sample=sample)
            except Exception as error:
                response = dict(sequence=sequence, started_ns=started, finished_ns=time.monotonic_ns(), error=str(error))
                target.write(json.dumps(response, allow_nan=False) + "\n")
                target.flush()
                return 1
            target.write(json.dumps(response, allow_nan=False) + "\n")
            target.flush()
            previous = sequence
    finally:
        if session is not None:
            session.close()  # Resource cleanup, never another health query.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--pci", required=True)
    args = parser.parse_args()
    require(sys.platform == "linux" and ct.sizeof(ct.c_void_p) == 8, "requires validated Linux 64-bit ABI")
    require(args.library.is_absolute() and args.library.is_file(), "explicit NVML library path required")
    return serve(sys.stdin, sys.stdout,
                 lambda: NvmlSession(ct.CDLL(str(args.library)), args.uuid, args.driver, args.pci))


if __name__ == "__main__":
    raise SystemExit(main())
