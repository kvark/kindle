"""`run_gpu_queue.py` with `gpu_host_guard_reviewed.py` as the per-job guard.

Usage: run_gpu_queue_reviewed.py QUEUE.json NEW_GUARDS_DIR REVIEWED_FAULTS.json
The queue, its checks and its stop-on-failure behaviour are `run_gpu_queue.run`.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import gpu_host_guard_reviewed
import run_gpu_queue

GUARD = Path(__file__).with_name("gpu_host_guard_reviewed.py")


def execute(command, **kwargs):
    if Path(command[1]).name != "gpu_host_guard.py":
        raise ValueError("unexpected guard command")
    return subprocess.run([command[0], str(GUARD), *command[2:]], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reviewed_faults", type=Path)
    args = parser.parse_args()
    gpu_host_guard_reviewed.load(args.reviewed_faults)
    os.environ["KINDLE_REVIEWED_KERNEL_FAULTS"] = str(args.reviewed_faults.resolve())
    run_gpu_queue.run(json.loads(args.specification.read_text()), args.output, execute=execute)


if __name__ == "__main__":
    main()
