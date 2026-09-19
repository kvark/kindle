"""Bounded NVIDIA incident capture and a one-process, fail-closed GPU guard.

This is an optional envelope for newly declared native jobs, not a scheduler or
runtime qualification. It never resets hardware, retries work, or changes drivers.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


FIELDS = (
    "uuid", "pci.bus_id", "driver_version", "gpu_recovery_action",
    "utilization.gpu", "memory.total", "memory.used", "memory.free",
    "memory.reserved", "temperature.gpu", "power.draw",
)
FAULT = re.compile(
    r"NVRM: Xid|PMU has halted|GPU is probably locked|RmInitAdapter failed"
    r"|Cannot initialize GSP|NVRM: API mismatch|GSP.*(?:timeout|Timeout)"
)
BOOT = Path("/proc/sys/kernel/random/boot_id")
DRIVER = Path("/sys/module/nvidia/version")


class GuardError(RuntimeError):
    pass


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def read_prefix(path, limit):
    with Path(path).open("rb") as source:
        return source.read(limit).decode(errors="replace")


def stop_child(process):
    """Signal only our still-unreaped direct child; never look up arbitrary PIDs."""
    if process.poll() is not None:
        return process.returncode
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            process.send_signal(sig)
            return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    # A task blocked inside the kernel may not respond even to SIGKILL.
    return None


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700)
        self.counter = 0
        self.unfinished = []
        write_json(self.root / "start.json", {
            "unix": time.time(), "boot_id": BOOT.read_text().strip(),
            "kernel": os.uname().release, "guard_sha256": fingerprint(__file__),
            "host_recovery_performed": False,
        })

    def event(self, kind, **data):
        row = {"event": kind, "unix": time.time(), "monotonic": time.monotonic(), **data}
        with (self.root / "events.jsonl").open("a") as target:
            target.write(json.dumps(row, allow_nan=False) + "\n")

    def probe(self, name, command, timeout=3, limit=2 * 1024 * 1024):
        self.counter += 1
        stem = f"{self.counter:05d}-{name}"
        paths = [self.root / f"{stem}.{suffix}" for suffix in ("stdout", "stderr")]
        result = {"command": command, "start_unix": time.time(), "pid": None,
                  "exit_code": None, "error": None, "files": [p.name for p in paths]}
        with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
            process = None
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr,
                                           env={**os.environ, "LC_ALL": "C"})
                result["pid"] = process.pid
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise GuardError("probe timed out")
                    if sum(os.fstat(f.fileno()).st_size for f in (stdout, stderr)) > limit:
                        raise GuardError("probe output limit exceeded")
                    time.sleep(0.02)
                if sum(os.fstat(f.fileno()).st_size for f in (stdout, stderr)) > limit:
                    raise GuardError("probe output limit exceeded")
                result["exit_code"] = process.returncode
            except (OSError, GuardError) as error:
                result["error"] = str(error)
            finally:
                if process is not None:
                    result["exit_code"] = stop_child(process)
                    if result["exit_code"] is None:
                        self.unfinished.append({"pid": process.pid, "files": result["files"]})
                        result["error"] = "probe child did not exit after TERM/KILL"
        result["end_unix"] = time.time()
        failed = result["error"] is not None or result["exit_code"] != 0
        text = read_prefix(paths[0], limit)
        if not failed and name in ("kernel", "nvml"):
            # Keep repeated samples in one append-only stream, not four files per poll.
            result.pop("files")
            self.event("probe", name=name, receipt=result, stdout=text,
                       stderr=read_prefix(paths[1], limit))
            for path in paths:
                path.unlink()
        else:
            write_json(self.root / f"{stem}.json", result)
            self.event("probe", name=name, receipt=f"{stem}.json", exit_code=result["exit_code"], error=result["error"])
        if result["error"] or result["exit_code"] != 0:
            raise GuardError(f"{name}: {result['error'] or 'exit ' + str(result['exit_code'])}")
        return text

    def snapshot(self):
        """Avoid more GPU ioctls after a fault; save bounded host evidence only."""
        commands = {
            "kernel-snapshot": ["journalctl", "--no-pager", "--quiet", "-o", "json",
                                "-n", "100000", "_TRANSPORT=kernel"],
            "boots": ["journalctl", "--list-boots", "--no-pager"],
            "pci": ["lspci", "-nnk"],
            "modules": ["lsmod"],
            "driver": ["modinfo", "nvidia"],
            "processes": ["ps", "-eo", "user,pid,ppid,stat,comm,wchan:32"],
        }
        for name, command in commands.items():
            try:
                text = self.probe(name, command, timeout=5, limit=64 * 1024 * 1024)
                if name == "kernel-snapshot" and len(text.splitlines()) >= 100000:
                    self.event("capture_incomplete", name=name, error="journal record limit reached")
            except GuardError as error:
                self.event("capture_incomplete", name=name, error=str(error))

    def seal(self, result):
        result["unfinished_children"] = self.unfinished
        write_json(self.root / "result.json", result)
        growing = {name for child in self.unfinished for name in child["files"]}
        write_json(self.root / "manifest.json", {
            "schema": 1,
            "pins": {p.name: fingerprint(p) for p in self.root.iterdir()
                     if p.is_file() and p.name not in growing},
            "unsealed_files": sorted(growing),
        })


def parse_health(text, uuid, driver, minimum_free):
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        raise GuardError("missing, extra or malformed GPU status rows")
    row = dict(zip(FIELDS, (value.strip() for value in rows[0])))
    if row["uuid"] != uuid or row["driver_version"] != driver:
        raise GuardError("GPU UUID or driver changed")
    if row["gpu_recovery_action"] != "None":
        raise GuardError(f"GPU recovery action is {row['gpu_recovery_action']}")
    for field in FIELDS[4:9]:
        try:
            value = float(row[field])
        except ValueError as error:
            raise GuardError(f"unavailable {field}") from error
        if not math.isfinite(value) or value < 0:
            raise GuardError(f"invalid {field}")
        row[field] = value
    if row["utilization.gpu"] > 100 or row["memory.free"] < minimum_free:
        raise GuardError("invalid utilization or insufficient directly free GPU memory")
    if row["memory.total"] <= 0 or any(row[f] > row["memory.total"] for f in FIELDS[6:9]):
        raise GuardError("invalid memory counters")
    # Optional thermal/power fields remain raw; N/A never becomes a fabricated zero.
    return row


def parse_journal(text, boot, limit):
    records = []
    try:
        for line in text.splitlines():
            row = json.loads(line)
            if row["_BOOT_ID"] != boot.replace("-", "") or not row["__CURSOR"]:
                raise GuardError("kernel journal boot/cursor mismatch")
            int(row["__MONOTONIC_TIMESTAMP"])
            if not isinstance(row["MESSAGE"], str):
                raise GuardError("unreadable kernel message")
            records.append(row)
    except (ValueError, KeyError, TypeError) as error:
        raise GuardError("malformed kernel journal") from error
    if len(records) >= limit:
        raise GuardError("kernel journal limit reached; coverage is incomplete")
    return records


def check_kernel(evidence, boot, cursor=None):
    limit = 4096 if cursor else 50001
    command = ["journalctl", "--no-pager", "--quiet", "-o", "json", f"--boot={boot.replace('-', '')}",
               "-n", str(limit), "_TRANSPORT=kernel"]
    if cursor:
        command.append(f"--after-cursor={cursor}")
    name = "kernel" if cursor else "kernel-baseline"
    records = parse_journal(evidence.probe(name, command, timeout=3, limit=64 * 1024 * 1024), boot, limit)
    if not cursor and not records:
        raise GuardError("no readable kernel journal baseline")
    for row in records:
        if FAULT.search(row["MESSAGE"]):
            evidence.event("kernel_fault", record=row)
            raise GuardError("NVIDIA kernel fault; this boot requires explicit recovery review")
    return records[-1]["__CURSOR"] if records else cursor


def check_health(evidence, boot, uuid, driver, minimum_free):
    if BOOT.read_text().strip() != boot or DRIVER.read_text().strip() != driver:
        raise GuardError("boot or loaded driver changed")
    text = evidence.probe("nvml", ["nvidia-smi", "-i", uuid,
                                  f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"])
    health = parse_health(text, uuid, driver, minimum_free)
    evidence.event("health", **health)
    return health


def run(evidence, command, uuid, driver, declaration, timeout, minimum_free=2048, interval=1):
    if not command or not 0 < timeout <= 172800 or not 0 < interval <= 5 or minimum_free < 2048:
        raise ValueError("require a command, bounded timeout/polling and at least 2048 MiB reserve")
    executable = shutil.which(command[0])
    if executable is None:
        raise ValueError("executable not found")
    boot = BOOT.read_text().strip()
    write_json(evidence.root / "job.json", {
        "argv": command, "executable": executable, "executable_sha256": fingerprint(executable),
        "declaration": str(Path(declaration).resolve()), "declaration_sha256": fingerprint(declaration),
        "uuid": uuid, "driver": driver, "timeout_seconds": timeout, "poll_seconds": interval,
        "minimum_free_mib": minimum_free, "environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("KINDLE_", "MEGANEURA_", "VK_", "BLADE_"))
        }, "RUST_BACKTRACE": "full", "guards_are_not_runtime_qualification": True,
    })
    process = None
    result = {"child_spawned": False, "child_exit_code": None, "guard_passed": False,
              "reason": None, "host_recovery_performed": False}
    try:
        cursor = check_kernel(evidence, boot)
        check_health(evidence, boot, uuid, driver, minimum_free)
        # Close the journal gap while the status command was running.
        cursor = check_kernel(evidence, boot, cursor)
        with (evidence.root / "child.stdout").open("xb") as stdout, (evidence.root / "child.stderr").open("xb") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, env={**os.environ, "RUST_BACKTRACE": "full"})
            result["child_spawned"] = True
            result["child_pid"] = process.pid
            evidence.event("child_spawned", pid=process.pid)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                cursor = check_kernel(evidence, boot, cursor)
                check_health(evidence, boot, uuid, driver, minimum_free)
                if time.monotonic() >= deadline:
                    raise GuardError("job time budget exceeded")
                time.sleep(interval)
            result["child_exit_code"] = process.returncode
            if process.returncode != 0:
                raise GuardError(f"native job exited {process.returncode}")
        check_health(evidence, boot, uuid, driver, minimum_free)
        check_kernel(evidence, boot, cursor)
        result["guard_passed"] = True
    except (GuardError, OSError, KeyboardInterrupt, InterruptedError) as error:
        result["reason"] = str(error) or type(error).__name__
        evidence.event("guard_stop", reason=result["reason"])
    finally:
        if process is not None:
            result["child_exit_code"] = stop_child(process)
            if result["child_exit_code"] is None:
                evidence.unfinished.append({"pid": process.pid, "files": ["child.stdout", "child.stderr"]})
                result["guard_passed"] = False
        if not result["guard_passed"]:
            evidence.snapshot()
        evidence.seal(result)
    return result


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    pins = manifest["pins"]
    unsealed = manifest["unsealed_files"]
    actual = {p.name for p in root.iterdir() if p.is_file()} - {"manifest.json"}
    if manifest["schema"] != 1 or not {"start.json", "result.json"}.issubset(pins):
        raise GuardError("incomplete evidence manifest")
    if set(pins) | set(unsealed) != actual or set(pins) & set(unsealed):
        raise GuardError("evidence file set differs")
    for name, expected in pins.items():
        if Path(name).name != name or fingerprint(root / name) != expected:
            raise GuardError(f"evidence changed: {name}")
    result = json.loads((root / "result.json").read_text())
    if unsealed and result.get("guard_passed"):
        raise GuardError("a live child cannot pass the guard")
    return {"verified_pins": len(pins), "unsealed_files": unsealed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("snapshot", "audit", "run"):
        child = sub.add_parser(mode)
        child.add_argument("root", type=Path, help="new output directory (existing only for audit)")
        if mode == "run":
            child.add_argument("--uuid", required=True)
            child.add_argument("--driver", required=True)
            child.add_argument("--declaration", type=Path, required=True)
            child.add_argument("--timeout", type=float, required=True)
            child.add_argument("--minimum-free-mib", type=int, default=2048)
            child.add_argument("--poll-seconds", type=float, default=1)
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        boundary = argv.index("--")
        argv, command = argv[:boundary], argv[boundary + 1:]
    args = parser.parse_args(argv)
    if args.mode == "audit":
        print(json.dumps(audit(args.root)))
        return 0
    evidence = Evidence(args.root)
    if args.mode == "snapshot":
        evidence.snapshot()
        evidence.seal({"snapshot_only": True, "gpu_work_started": False, "host_recovery_performed": False})
        return 0
    def interrupted(_signal, _frame):
        raise InterruptedError("guard interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    result = run(evidence, command, args.uuid, args.driver, args.declaration,
                 args.timeout, args.minimum_free_mib, args.poll_seconds)
    print(json.dumps(result))
    return 0 if result["guard_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
