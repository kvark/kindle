"""Bounded transport for the experimental NVML worker; not a GPU-job launcher."""

import json
import math
import os
import select
import subprocess
import time

import gpu_guard as guard
import gpu_health as reader


class HealthClient:
    def __init__(self, command, uuid, driver, pci, check_host, stderr, *, minimum_free=2048,
                 max_gap=1.5, bootstrap_timeout=3):
        if (type(minimum_free) is not int or minimum_free < 2048
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in (max_gap, bootstrap_timeout))
                or not 0 < max_gap <= 1.5 or not 0 < bootstrap_timeout <= 3):
            raise ValueError("invalid health budget")
        self.uuid, self.driver, self.pci = uuid, driver, pci.lower()
        self.check_host, self.minimum_free = check_host, minimum_free
        self.max_gap, self.bootstrap_timeout = max_gap, bootstrap_timeout
        self.sequence, self.last_ns, self.closed = 0, None, False
        # The worker must not load NVML until its first explicit sample request.
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, bufsize=0, start_new_session=True)

    def sample(self):
        if self.closed:
            raise guard.GuardError("health client is closed")
        try:
            self.check_host()  # Kernel/boot/driver checks before every request.
            sent = time.monotonic_ns()
            deadline = (self.last_ns + int(self.max_gap * 1e9) if self.last_ns is not None
                        else sent + int(self.bootstrap_timeout * 1e9))
            if sent >= deadline:
                raise guard.GuardError("health sampling gap exceeded before request")
            if self.process.poll() is not None or select.select([self.process.stdout], [], [], 0)[0]:
                raise guard.GuardError("worker exited or emitted unsolicited output")
            self.sequence += 1
            request = json.dumps(dict(sequence=self.sequence, operation="sample")).encode() + b"\n"
            self.process.stdin.write(request)
            buffer = bytearray()
            while b"\n" not in buffer:
                remaining = (deadline - time.monotonic_ns()) / 1e9
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise guard.GuardError("health request deadline exceeded")
                chunk = os.read(self.process.stdout.fileno(), 65537 - len(buffer))
                if not chunk:
                    raise guard.GuardError("worker exited without a response")
                buffer.extend(chunk)
                if len(buffer) > 65536:
                    raise guard.GuardError("health response exceeds limit")
            received = time.monotonic_ns()
            if received > deadline or not buffer.endswith(b"\n") or buffer.count(b"\n") != 1:
                raise guard.GuardError("late or malformed health response")
            row = reader.decode(buffer)
            if (not isinstance(row, dict) or type(row.get("sequence")) is not int
                    or row["sequence"] != self.sequence):
                raise guard.GuardError("health response sequence differs")
            if "error" in row:
                raise guard.GuardError("health worker: " + str(row["error"]))
            if (set(row) != {"sequence", "started_ns", "finished_ns", "sample"}
                    or type(row["started_ns"]) is not int or type(row["finished_ns"]) is not int
                    or not sent <= row["started_ns"] <= row["finished_ns"] <= received):
                raise guard.GuardError("invalid health response clock/schema")
            parsed = reader.validate_sample(row["sample"], self.uuid, self.driver, self.pci, self.minimum_free)
            self.check_host()  # A response may straddle a kernel fault.
            accepted = time.monotonic_ns()
            if accepted > deadline:
                raise guard.GuardError("health sampling gap exceeded after host check")
            self.last_ns = received
            return dict(sent_ns=sent, received_ns=received, accepted_ns=accepted, worker_pid=self.process.pid,
                        response=row, health=parsed)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.closed:
            return self.process.poll()
        self.closed = True
        self.process.stdin.close()
        try:
            result = self.process.wait(timeout=.25)
        except subprocess.TimeoutExpired:
            result = guard.stop_child(self.process)
        self.process.stdout.close()
        return result  # None means the owning controller must retain an unfinished helper.
