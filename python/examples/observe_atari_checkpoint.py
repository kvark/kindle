"""Watch a caller-owned training child and retain its first declared midpoint.

The caller must propagate failures to its guard. This module never starts or
signals a process, retries a copy, or treats a retained prefix as a completed run.
"""

import json
import math
import os
from pathlib import Path
import time

import retain_atari_checkpoint as retention

MAX_ROW_BYTES = 1024 * 1024
MAX_WAIT_SECONDS = 64800
require = retention.require


def process(pid, proc=Path('/proc')):
    require(type(pid) is int and pid > 0, 'invalid process PID')
    directory = proc / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        command = (directory / 'cmdline').read_bytes().split(b'\0')
        return dict(pid=pid, state=fields[0], parent_pid=int(fields[1]), start_ticks=fields[19],
                    command=[os.fsdecode(arg) for arg in command[:-1]])
    except (FileNotFoundError, ProcessLookupError):
        return None


def require_child(bound, current):
    require(current is not None and current['state'] not in ('Z', 'X', 'x'), 'bound training child is not live')
    require(all(current[key] == bound[key] for key in ('pid', 'start_ticks', 'parent_pid', 'command')),
            'training child identity changed')


def bind_child(pid, command, parent_pid, *, inspect=process):
    current = inspect(pid)
    require(current is not None and current['state'] not in ('Z', 'X', 'x'), 'training child is not live at binding')
    require(current['command'] == command and current['parent_pid'] == parent_pid,
            'training command or owning controller differs')
    bound = {key: current[key] for key in ('pid', 'start_ticks', 'parent_pid', 'command')}
    require_child(bound, inspect(pid))
    return bound


class Midpoint:
    """Incremental log reader: incomplete writes wait; missed or moving saves fail."""

    def __init__(self, log, expected_header, action):
        require(type(action) is int and action > 0 and action % 6 == 0, 'invalid checkpoint action')
        retention.check_training_header(expected_header, action)
        require(action < expected_header['steps'], 'midpoint must precede the final action budget')
        self.log = Path(log)
        self.expected_header = {key: value for key, value in expected_header.items()
                                if key not in retention.RUNTIME_HEADER}
        self.action = action
        self.position = self.size = self.last_step = 0
        self.file_identity = self.header = self.event = None
        self.failed = False

    def poll(self):
        require(not self.failed and self.event is None, 'midpoint reader is already terminal')
        try:
            return self._poll()
        except BaseException:
            self.failed = True
            raise

    def _poll(self):
        require(not self.log.is_symlink(), 'training log is a symlink')
        try:
            source = self.log.open('rb')
        except FileNotFoundError:
            require(self.file_identity is None, 'training log disappeared')
            return None
        with source:
            state = os.fstat(source.fileno())
            identity = (state.st_dev, state.st_ino)
            path_state = self.log.stat()
            require(not self.log.is_symlink() and identity == (path_state.st_dev, path_state.st_ino),
                    'training log path changed')
            require(self.file_identity in (None, identity) and state.st_size >= self.size,
                    'training log was replaced or truncated')
            self.file_identity, self.size = identity, state.st_size
            source.seek(self.position)
            while True:
                line = source.readline(MAX_ROW_BYTES + 1)
                require(len(line) <= MAX_ROW_BYTES, 'oversized training row')
                if not line.endswith(b'\n'):
                    return None
                self.position = source.tell()
                row = json.loads(line)
                if self.header is None:
                    require(row.get('event') == 'run_start'
                        and {key: value for key, value in row.items() if key not in retention.RUNTIME_HEADER}
                        == self.expected_header, 'training header differs from the declared recipe')
                    self.header = row
                    continue
                require(row.get('event') != 'run_start', 'duplicate training header')
                require(row.get('event') != 'run_end', 'training ended before the midpoint was retained')
                step = row.get('run_step')
                require(type(step) is int and self.last_step <= step <= self.action, 'midpoint save was missed or reordered')
                self.last_step = step
                if row.get('event') == 'checkpoint':
                    require(step == self.action, 'undeclared checkpoint cadence')
                    self.event = row
                    return row


def observe_midpoint(log, checkpoint, destination, *, action, expected_header, schema, encoder,
                     bound, timeout_seconds=MAX_WAIT_SECONDS, inspect=process,
                     emit=lambda *_args, **_fields: None):
    """Return only after one successful copy while the same training child is live."""
    require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
            and 0 < timeout_seconds <= MAX_WAIT_SECONDS, 'invalid bounded observer timeout')
    reader = Midpoint(log, expected_header, action)
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), 'midpoint archive already exists')
    deadline = time.monotonic() + timeout_seconds
    while True:
        require_child(bound, inspect(bound['pid']))
        require(time.monotonic() < deadline, 'bounded midpoint wait expired')
        event = reader.poll()
        if event is not None:
            require_child(bound, inspect(bound['pid']))
            emit('midpoint_retention_start', child=bound, checkpoint_event=event)
            result = retention.retain(log, checkpoint, destination, action=action,
                expected_header=expected_header, schema=schema, encoder=encoder)
            require_child(bound, inspect(bound['pid']))
            require(time.monotonic() < deadline, 'bounded midpoint wait expired during copy')
            require(result['source_checkpoint_event'] == event, 'retained midpoint event changed')
            emit('midpoint_retention_complete', child=bound, retained=str(destination),
                 marker_sha256=retention.digest(destination / 'retained.json'))
            return result
        time.sleep(min(1, max(0, deadline - time.monotonic())))
