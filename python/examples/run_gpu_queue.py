"""Run a declared list of guarded native jobs serially; stop on any failure.

The systemd service owns this controller and its process group. Each host guard
owns the direct native-bearing worker, not this controller. No retries, polling
NVML, host recovery or implicit resume. Completed entries are reusable only by
starting a NEW queue containing different, unstarted output directories.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(specification, output, *, execute=subprocess.run):
    output.mkdir(parents=True, exist_ok=False)
    host = specification["host"]
    names = [job["name"] for job in specification["jobs"]]
    if not names or len(set(names)) != len(names) or any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("unique plain job names required")
    (output / "specification.json").write_text(json.dumps(specification, indent=2) + "\n")
    for prerequisite in specification.get("prerequisites", []):
        path = Path(prerequisite["guard_result"])
        deadline = time.monotonic() + prerequisite["wait_seconds"]
        print(f"Waiting for {path}", flush=True)
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"prerequisite did not finish: {path}")
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        state = json.loads(path.read_text())
        if not state["host_guard_passed"] or state["child_exit_code"] != 0 or state["unfinished_children"]:
            raise RuntimeError(f"failed prerequisite: {path}")
        result = json.loads(Path(prerequisite["result"]).read_text())
        if any(result.get(key) != value for key, value in prerequisite["expected"].items()):
            raise RuntimeError(f"incomplete prerequisite result: {path}")
    completed = []
    for job in specification["jobs"]:
        result_file = Path(job["result"]) if "result" in job else None
        if result_file is not None and result_file.exists():
            raise FileExistsError(f"refusing to reuse result: {result_file}")
        executable = Path(job["command"][0]).resolve()
        with executable.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != job["executable_sha256"]:
            raise ValueError(f"executable changed: {executable}")
        declaration = {"host_guard": dict(**host, command=job["command"], executable_sha256=digest,
                                          timeout_seconds=job["timeout_seconds"])}
        path = output / (job["name"] + ".json")
        path.write_text(json.dumps(declaration, indent=2) + "\n")
        guard = output / job["name"]
        print(f"Starting {job['name']}", flush=True)
        execute([sys.executable, str(Path(__file__).with_name("gpu_host_guard.py")),
                 "run", str(guard), "--declaration", str(path)], check=True,
                env={**os.environ, **job.get("environment", {})})
        state = json.loads((guard / "result.json").read_text())
        if not state["host_guard_passed"] or state["child_exit_code"] != 0 or state["unfinished_children"]:
            raise RuntimeError(f"guard did not complete: {job['name']}")
        if result_file is not None:
            if job.get("result_format") == "jsonl_last":
                with result_file.open() as stream:
                    last = None
                    for line in stream:
                        if line.strip():
                            last = line
                result = json.loads(last) if last is not None else {}
            else:
                result = json.loads(result_file.read_text())
            for key, value in job.get("expected", {"status": "complete"}).items():
                if result.get(key) != value:
                    raise RuntimeError(f"unexpected {job['name']} result field {key}: {result.get(key)}")
            for key, value in job.get("minimum", {}).items():
                if key not in result or result[key] < value:
                    raise RuntimeError(f"incomplete {job['name']} result field {key}")
        if "stdout_contains" in job:
            if job["stdout_contains"] not in (guard / "child.stdout").read_text():
                raise RuntimeError(f"missing positive test result: {job['name']}")
        completed.append(job["name"])
        (output / "progress.json").write_text(json.dumps(dict(completed=completed), indent=2) + "\n")
        print(f"Completed {job['name']}", flush=True)
    (output / "result.json").write_text(json.dumps(dict(status="complete", completed=completed), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(json.loads(args.specification.read_text()), args.output)


if __name__ == "__main__":
    main()
