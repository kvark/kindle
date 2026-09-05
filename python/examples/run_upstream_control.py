"""Run a pinned, locally installed DreamerV3 Atari control with provenance.

Invoke with the upstream Python environment, not Kindle's extension environment.
The source checkout must be clean except for the declared wrapper config below.
No packages, weights or games are downloaded by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
PUBLISHED_CONFIG = """
kindle_published:
  env.atari100k.actions: all
  env.atari100k.noops: 0
  env.atari100k.length: 100000
  env.atari100k.use_seed: True
"""


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True)


def validate_source(source: Path) -> str:
    if git(source, "rev-parse", "HEAD").strip() != REVISION:
        raise ValueError(f"upstream must be at {REVISION}")
    changed = git(source, "diff", "--name-only", "HEAD").splitlines()
    if changed != ["dreamerv3/configs.yaml"]:
        raise ValueError("only the declared published-wrapper config may differ from upstream")
    original = git(source, "show", f"{REVISION}:dreamerv3/configs.yaml")
    actual = (source / "dreamerv3/configs.yaml").read_text()
    if actual != original + PUBLISHED_CONFIG:
        raise ValueError("append PUBLISHED_CONFIG from this runner to the pinned configs.yaml")
    return git(source, "diff", "HEAD", "--", "dreamerv3/configs.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--size", choices=("1m", "12m"), default="12m")
    parser.add_argument("--game", default="pong")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--cuda-root", type=Path,
                        help="optional CUDA nvcc package directory containing bin/ptxas")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 10:
        parser.error("--steps must be a positive multiple of the upstream 10-step driver block")
    source = args.source.resolve()
    patch = validate_source(source)
    logdir = args.logdir.resolve()
    # Refuse to silently resume or overwrite any earlier control.
    logdir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(source / "dreamerv3/main.py"),
        "--configs", "atari100k", f"size{args.size}", "kindle_published",
        "--task", f"atari100k_{args.game}", "--seed", str(args.seed),
        "--run.steps", str(args.steps), "--logdir", str(logdir),
        "--logger.outputs", "jsonl", "--run.log_every", "60",
    ]
    environment = os.environ.copy()
    if args.cuda_root:
        cuda_root = args.cuda_root.resolve()
        if not (cuda_root / "bin/ptxas").is_file():
            parser.error("--cuda-root must contain bin/ptxas")
        environment["PATH"] = str(cuda_root / "bin") + os.pathsep + environment.get("PATH", "")
        environment["XLA_FLAGS"] = (
            environment.get("XLA_FLAGS", "") + f" --xla_gpu_cuda_data_dir={cuda_root}"
        ).strip()
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    manifest = {
        "source_revision": REVISION,
        "source_config_diff": patch,
        "source_config_diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "command": command,
        "python": platform.python_version(),
        "packages": dict(sorted(packages.items())),
        "environment": {name: environment.get(name) for name in
                        ("XLA_FLAGS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
        "protocol": "published",
        "step_accounting": "upstream driver records include action-free reset observations",
        "model_input": "learned 64x64 RGB encoder; no DINO or Kindle model code",
        "status": "running",
    }
    manifest_path = logdir / "reference-manifest.json"

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
    started = time.perf_counter()
    with (logdir / "console.log").open("w") as output:
        result = subprocess.run(command, cwd=source, env=environment,
                                stdout=output, stderr=subprocess.STDOUT)
    manifest.update(status="complete" if result.returncode == 0 else "failed",
                    exit_code=result.returncode, elapsed_seconds=time.perf_counter() - started)
    save_manifest()
    if result.returncode == 0:
        (logdir / "RUN_COMPLETE").write_text("upstream exited with status zero\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
