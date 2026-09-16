"""Host-only containment for one explicitly declared direct child; never queries NVML.

Kernel logs, boot/loaded-driver identity and process outcomes are not GPU health,
free-memory or utilization measurements. Native device assertions and the job's
own acceptance checks remain separate. No retries, descendants or host recovery.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

import gpu_guard as retained


class HostEvidence(retained.Evidence):
    def probe(self, name, command, **kwargs):
        if not command or command[0] not in {"journalctl", "lspci", "lsmod", "modinfo", "ps"}:
            raise retained.GuardError("host guard refuses a non-host probe")
        return super().probe(name, command, **kwargs)


def declaration(path):
    def unique(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate declaration key")
            result[name] = value
        return result
    config = json.loads(Path(path).read_text(), object_pairs_hook=unique)["host_guard"]
    fields = {"monitoring", "boot_id", "driver", "command", "executable_sha256", "timeout_seconds", "poll_seconds"}
    if not isinstance(config, dict) or set(config) != fields or config["monitoring"] != "host-only":
        raise ValueError("require an explicit host-only declaration")
    if str(uuid.UUID(config["boot_id"])) != config["boot_id"] or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", config["driver"]):
        raise ValueError("invalid boot or driver identity")
    for name, maximum in (("timeout_seconds", 172800), ("poll_seconds", 5)):
        value = config[name]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("invalid host guard budget")
    command = config["command"]
    if (not isinstance(command, list) or not command or any(not isinstance(value, str) or not value or '\0' in value for value in command)
            or not Path(command[0]).is_absolute()):
        raise ValueError("require an absolute direct executable and explicit arguments")
    if Path(command[0]).name in {"nvidia-smi", "nvidia-debugdump", "nvidia-bug-report.sh"}:
        raise ValueError("NVML/device diagnostic tools are not permitted")
    if not re.fullmatch(r"[0-9a-f]{64}", config["executable_sha256"]):
        raise ValueError("invalid executable identity")
    if retained.fingerprint(command[0]) != config["executable_sha256"]:
        raise ValueError("declared executable changed")
    return config


def check_host(evidence, config, cursor=None):
    started = time.monotonic()
    def identity():
        if (retained.BOOT.read_text().strip() != config["boot_id"]
                or retained.DRIVER.read_text().strip() != config["driver"]):
            raise retained.GuardError("boot or loaded driver changed")
    identity()
    cursor = retained.check_kernel(evidence, config["boot_id"], cursor)
    identity()
    evidence.event("host_check", boot_id=config["boot_id"], driver=config["driver"],
                   kernel_cursor=cursor, started_monotonic=started)
    return cursor


def run(root, declaration_path):
    config = declaration(declaration_path)  # Refuse legacy declarations before probes, outputs or a child.
    evidence = HostEvidence(root)
    retained.write_json(evidence.root / "job.json", {
        "host_guard": config, "declaration": str(Path(declaration_path).resolve()),
        "declaration_sha256": retained.fingerprint(declaration_path),
        "host_guard_sha256": retained.fingerprint(__file__),
        "helper_sha256": retained.fingerprint(retained.__file__),
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("KINDLE_", "MEGANEURA_", "VK_", "BLADE_"))},
        "RUST_BACKTRACE": "full", "child_scope": "direct-only",
    })
    process = None
    result = dict(monitoring="host-only", host_guard_passed=False, child_spawned=False, child_exit_code=None,
                  reason=None, host_recovery_performed=False, guard_nvml_queries=0,
                  gpu_telemetry="unmeasured", hardware_qualified=False)
    try:
        cursor = check_host(evidence, config)
        if retained.fingerprint(config["command"][0]) != config["executable_sha256"]:
            raise retained.GuardError("declared executable changed before launch")
        with (evidence.root / "child.stdout").open("xb") as stdout, (evidence.root / "child.stderr").open("xb") as stderr:
            process = subprocess.Popen(config["command"], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, env={**os.environ, "RUST_BACKTRACE": "full"})
            result.update(child_spawned=True, child_pid=process.pid)
            evidence.event("child_spawned", pid=process.pid)
            deadline = time.monotonic() + config["timeout_seconds"]
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise retained.GuardError("job time budget exceeded")
                cursor = check_host(evidence, config, cursor)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise retained.GuardError("job time budget exceeded")
                if process.poll() is None:
                    time.sleep(min(config["poll_seconds"], remaining))
            result["child_exit_code"] = process.returncode
            if process.returncode != 0:
                raise retained.GuardError(f"native job exited {process.returncode}")
        check_host(evidence, config, cursor)
        result["host_guard_passed"] = True
    except (Exception, KeyboardInterrupt) as error:
        result["reason"] = str(error) or type(error).__name__
        evidence.event("guard_stop", reason=result["reason"])
    finally:
        if process is not None:
            result["child_exit_code"] = retained.stop_child(process)
            if result["child_exit_code"] is None:
                evidence.unfinished.append({"pid": process.pid, "files": ["child.stdout", "child.stderr"]})
                result["host_guard_passed"] = False
        if not result["host_guard_passed"]:
            evidence.snapshot()
        evidence.seal(result)
    return result


def audit(root):
    checked = retained.audit(root)
    root = Path(root)
    result, job = (json.loads((root / name).read_text()) for name in ("result.json", "job.json"))
    if (result["monitoring"] != "host-only" or result["guard_nvml_queries"] != 0
            or result["gpu_telemetry"] != "unmeasured" or result["hardware_qualified"] is not False
            or result["host_recovery_performed"] is not False or job["host_guard"]["monitoring"] != "host-only"):
        raise retained.GuardError("host-only result identity differs")
    if result["host_guard_passed"] and (not result["child_spawned"] or result["child_exit_code"] != 0
                                       or result["unfinished_children"] or checked["unsealed_files"] or result["reason"] is not None):
        raise retained.GuardError("incomplete child cannot pass the host guard")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    checks = [event for event in events if event["event"] == "host_check"]
    children = [event for event in events if event["event"] == "child_spawned"]
    if result["host_guard_passed"] and (len(children) != 1 or len(checks) < 2
            or children[0]["pid"] != result["child_pid"]
            or not checks[0]["monotonic"] <= children[0]["monotonic"] <= checks[-1]["monotonic"]):
        raise retained.GuardError("host preflight/child/postflight evidence incomplete")
    for check in checks:
        if (check["boot_id"] != job["host_guard"]["boot_id"] or check["driver"] != job["host_guard"]["driver"]
                or not check["kernel_cursor"] or not check["started_monotonic"] <= check["monotonic"]):
            raise retained.GuardError("host check identity differs")
    for event in events:
        if event["event"] == "health":
            raise retained.GuardError("GPU telemetry in host-only evidence")
        if event["event"] in {"kernel_fault", "guard_stop"} and result["host_guard_passed"]:
            raise retained.GuardError("host guard pass contradicts retained events")
        if event["event"] == "probe":
            receipt = event["receipt"]
            receipt = receipt if isinstance(receipt, dict) else json.loads((root / receipt).read_text())
            if receipt["command"][0] not in {"journalctl", "lspci", "lsmod", "modinfo", "ps"}:
                raise retained.GuardError("non-host probe in retained evidence")
    return dict(**checked, host_guard_passed=result["host_guard_passed"], monitoring="host-only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--declaration", type=Path)
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(audit(args.root)))
        return 0
    if args.declaration is None:
        parser.error("run requires --declaration")
    def interrupted(_signal, _frame):
        raise InterruptedError("host guard interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    result = run(args.root, args.declaration)
    print(json.dumps(result))
    return 0 if result["host_guard_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
