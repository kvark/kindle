"""Run a pinned, locally installed DreamerV3 Atari control with provenance.

Invoke with the upstream Python environment, not Kindle's extension environment.
The source checkout must match the declared wrapper config and ALE corrections.
No packages, weights or games are downloaded by this runner.
The upstream learner runs in this process so the host-only guard owns it.
Application GPU telemetry is disabled. Stock JAX0.6.2 uses NVML internally;
normal backend initialization is allowed by the September 26 user direction.
Hardware and memory gates still require a separate bounded declaration.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
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


def configured_source(original: str) -> str:
    # Elements only permits overrides of declared keys. Expose the existing
    # wrapper defaults before overriding them in the published profile.
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("    atari100k:"):
            lines[index] = line.replace("clip_reward: False}",
                "clip_reward: False, length: 108000, use_seed: False}")
            if lines[index] == line:
                raise ValueError("unrecognized pinned Atari default declaration")
            return "".join(lines) + PUBLISHED_CONFIG
    raise ValueError("missing pinned Atari defaults")


def compatible_atari_source(original: str) -> str:
    seed = "self.ale.setInt(b'random_seed', self.rng.integers(0, 2 ** 31))"
    sticky = "    self.ale.setFloat('repeat_action_probability', 0.25 if sticky else 0.0)\n"
    initialized = "      self.ale.setLoggerMode(ale_py.LoggerMode.Error)\n"
    if any(original.count(line) != 1 for line in (seed, sticky, initialized)):
        raise ValueError("unrecognized pinned ALE initialization")
    source = original.replace(seed, "self.ale.setInt('random_seed', int(self.rng.integers(0, 2 ** 31)))")
    # ALE caches the sticky probability when loadROM constructs the environment.
    return source.replace(sticky, "").replace(initialized, initialized + "  " + sticky)


def validate_source(source: Path) -> str:
    if git(source, "rev-parse", "HEAD").strip() != REVISION:
        raise ValueError(f"upstream must be at {REVISION}")
    if git(source, "ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("upstream must not contain untracked source files")
    changed = git(source, "diff", "--name-only", "HEAD").splitlines()
    if changed != ["dreamerv3/configs.yaml", "embodied/envs/atari.py"]:
        raise ValueError("require exactly the declared wrapper config and ALE corrections")
    original = git(source, "show", f"{REVISION}:dreamerv3/configs.yaml")
    actual = (source / "dreamerv3/configs.yaml").read_text()
    if actual != configured_source(original):
        raise ValueError("configs.yaml must match configured_source and PUBLISHED_CONFIG from this runner")
    original = git(source, "show", f"{REVISION}:embodied/envs/atari.py")
    if (source / "embodied/envs/atari.py").read_text() != compatible_atari_source(original):
        raise ValueError("Atari source must match the exact ALE corrections")
    return git(source, "diff", "HEAD", "--", *changed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--size", choices=("1m", "12m"), default="12m")
    parser.add_argument("--game", default="pong")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--matched-actions", action="store_true",
                        help="Phase 2 shared Atari wrapper and exact actual-action/update accounting")
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--cuda-root", type=Path,
                        help="optional CUDA nvcc package directory containing bin/ptxas")
    args = parser.parse_args()
    if not args.matched_actions and (args.steps <= 0 or args.steps % 10):
        parser.error("--steps must be a positive multiple of the upstream 10-step driver block")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.matched_actions and (args.steps <= 0 or args.steps % args.num_envs):
        parser.error("matched steps must be a positive multiple of num-envs")
    if args.matched_actions and (args.size != "12m" or args.compute_dtype != "float32"):
        parser.error("Phase 2 matched control requires size12m and float32")
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
        "--run.envs", str(args.num_envs), "--run.debug", "True",
        "--jax.compute_dtype", args.compute_dtype,
        "--run.usage.nvsmi", "False", "--run.usage.gputil", "False",
    ]
    if args.matched_actions:
        command += ["--env.atari100k.sticky", "True", "--replay.size", "100000",
                    "--jax.profiler", "False"]
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
        "protocol": "phase2-matched-actions-v1" if args.matched_actions else "published",
        "wrapper_corrections": ["ALE seed API types", "sticky probability set before ROM load"],
        "step_accounting": ("exact actual actions; resets earn no updates" if args.matched_actions else
                            "upstream driver records include action-free reset observations"),
        "model_input": "learned 64x64 RGB encoder; no DINO or Kindle model code",
        "process_scope": "direct native-bearing process; synchronous environments",
        "gpu_telemetry": "application collectors disabled",
        "backend_nvml": "normal initialization permitted; application telemetry disabled",
        "status": "running",
    }
    manifest_path = logdir / "reference-manifest.json"

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
    started = time.perf_counter()
    exit_code = 1
    restore_train = None
    try:
        os.environ.update(environment)
        if args.matched_actions:
            # Keep the pinned agent/math untouched; replace only the collection
            # harness whose stock step counter includes action-free resets.
            from functools import partial
            sys.path.insert(0, str(source))
            import embodied
            from upstream_matched import train
            restore_train = embodied.run.train
            embodied.run.train = partial(train, game=args.game, seed=args.seed)
        os.chdir(source)
        sys.argv = command[1:]
        with (logdir / "console.log").open("x") as output, redirect_stdout(output), redirect_stderr(output):
            try:
                runpy.run_path(command[1], run_name="__main__")
                exit_code = 0
            except SystemExit as error:
                exit_code = error.code or 0
                if not isinstance(exit_code, int):
                    print(exit_code, file=sys.stderr)
                    exit_code = 1
    finally:
        if args.matched_actions and restore_train is not None:
            embodied.run.train = restore_train
        manifest.update(status="complete" if exit_code == 0 else "failed",
                        exit_code=exit_code, elapsed_seconds=time.perf_counter() - started)
        save_manifest()
    if exit_code == 0:
        (logdir / "RUN_COMPLETE").write_text("upstream exited with status zero\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
