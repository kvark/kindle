"""Strict aggregation of Kindle Atari JSONL score windows."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


SCORE_WINDOW_FRAMES = (350_000, 400_000)

KINDLE_TARGET_RUNTIME = "kindle-ale_py-0.12.1"
UPSTREAM_D3_TARGET_RUNTIME = "upstream-d3-ale_py-0.9.0"

ATARI_PROFILES = {
    "current": {
        "action_repeat": 4,
        "full_action_space": False,
        "noop_max": 30,
        "max_episode_frames": 108_000,
    },
    "published": {
        "action_repeat": 4,
        "full_action_space": True,
        "noop_max": 0,
        "max_episode_frames": 100_000,
    },
    "published-minimal": {
        "action_repeat": 4,
        "full_action_space": False,
        "noop_max": 0,
        "max_episode_frames": 100_000,
    },
}

# Reference scores are keyed by exact wrapper protocol. The 12M values are the
# rounded DreamerV3XS results reported by Wang et al. (ICLR 2025); its released
# comparison omits raw XS curves, so these remain secondary targets. The 200M
# values come from the D3 score artifact bundled at upstream commit 2411f7d1.
# That artifact predates the pinned e3f02248 implementation reference, so it is
# historical result evidence rather than an executable golden for that code.
# Both use all 18 actions and therefore stay under `published`;
# `published-minimal` has only its own three-seed random control.
ATARI_TARGETS = {
    "published": {
        "ALE/Pong-v5": {
            "matched_random_3_seed": -20.622710622710624,
            "dreamerv3_12m_reported": -10.0,
            "dreamerv3_200m_published": -4.537,
        },
        "ALE/PrivateEye-v5": {
            "matched_random_3_seed": 46.666666666666664,
            "dreamerv3_12m_reported": 207.0,
            "dreamerv3_200m_published": 2_895.24,
        },
        "ALE/Freeway-v5": {
            "matched_random_3_seed": 0.0,
            "dreamerv3_12m_reported": 0.0,
            "dreamerv3_200m_published": 0.0,
        },
    },
    "published-minimal": {
        "ALE/Pong-v5": {
            "matched_random_3_seed": -20.446031746031746,
        },
    },
}

# The pinned upstream executable and Kindle use different ALE releases. A
# paired control feeds the exact same three seeded random-action streams to
# both published-protocol wrappers for 100,000 decisions. Their trajectories
# first differ at decision 15. In the 350k--400k emulator-frame score window,
# upstream ALE 0.9 averages -1854/91 = -20.373626... across seeds 0--2 (13,
# 13, and 14 episodes), while Kindle ALE 0.12 reproduces the -20.622710...
# target above. Keep the historical D3 targets shared, but never attribute the
# ALE 0.12 random floor to the locally reproduced ALE 0.9 reference curve.
ATARI_TARGET_OVERRIDES = {
    UPSTREAM_D3_TARGET_RUNTIME: {
        "published": {
            "ALE/Pong-v5": {
                "matched_random_3_seed": -20.373626373626372,
            },
        },
    },
}

UPSTREAM_D3_REFERENCE = {
    (
        "source",
        "repository",
    ): "https://github.com/danijar/dreamerv3",
    ("source", "revision"): "e3f02248693a79dc8b0ebd62c93683888ddaccfe",
    (
        "source",
        "config_only_diff_sha256",
    ): "334ca81ec0662714e58d39cc48a7b32473e201e5d7fee006f840a16490d90717",
    (
        "runtime",
        "package_freeze_sha256",
    ): "cc5d751228c82831548251e44640f49317bd95ba708786cf2ec3172d796f0999",
    ("runtime", "python"): "3.11.15",
    ("runtime", "jax"): "0.6.2",
    ("runtime", "jaxlib"): "0.6.2",
    ("runtime", "cuda_nvcc"): "12.9.86",
    ("runtime", "ale_py"): "0.9.0",
    ("runtime", "compute_dtype"): "bfloat16",
    ("runtime", "device_class"): "NVIDIA CUDA",
    ("experiment", "task"): "atari100k_pong",
    ("experiment", "environment"): "ALE/Pong-v5",
    ("experiment", "model_input"): "learned 64x64 RGB encoder",
    ("experiment", "world_backpropagation"): "full 64-step recurrence",
    (
        "experiment",
        "configs",
    ): ["atari100k", "size1m", "kindle_published"],
    ("experiment", "steps"): 100_000,
    ("experiment", "train_ratio"): 256,
    ("experiment", "batch_size"): 16,
    ("experiment", "batch_length"): 64,
    ("experiment", "replay_context"): 1,
    ("atari_protocol", "name"): "published",
    ("atari_protocol", "action_repeat"): 4,
    ("atari_protocol", "actions"): "all",
    ("atari_protocol", "sticky_actions"): False,
    ("atari_protocol", "reset_noop_max"): 0,
    ("atari_protocol", "episode_frame_cap"): 100_000,
    ("atari_protocol", "frame_pool"): "max of final 2",
    ("atari_protocol", "resize"): "Pillow bilinear",
    ("atari_protocol", "environment_seeded"): True,
    ("score_window_frames",): [350_000, 400_000],
    (
        "completion_evidence",
    ): (
        "RUN_COMPLETE is installed only after the upstream process exits with "
        "status zero"
    ),
    (
        "kindle_commit_at_configuration",
    ): "f395188cc08b0968c03ebdd91fd3e706570962ea",
}


class AtariScoreError(ValueError):
    """Raised when logs cannot prove a protocol-complete score."""


def _read_events(path: Path) -> Iterator[dict[str, object]]:
    """Yield validated JSONL events without retaining the full log."""

    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise AtariScoreError(f"missing required file: {path}") from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise AtariScoreError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(event, dict):
                raise AtariScoreError(
                    f"{path}:{line_number}: JSONL event must be an object"
                )
            yield event


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AtariScoreError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise AtariScoreError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise AtariScoreError(f"{path}: JSON value must be an object")
    return value


def _manifest_value(
    manifest: dict[str, object], path: tuple[str, ...], source: Path
) -> object:
    value: object = manifest
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise AtariScoreError(
                f"{source}: manifest is missing {'.'.join(path)!r}"
            )
        value = value[key]
    return value


def load_upstream_d3_segments(
    logdirs: Iterable[Path],
) -> list[dict[str, object]]:
    """Load completed, provenance-pinned upstream D3 reference curves."""
    segments = []
    for logdir in logdirs:
        logdir = Path(logdir)
        completion = logdir / "RUN_COMPLETE"
        if not completion.is_file():
            raise AtariScoreError(
                f"{logdir}: upstream run has no RUN_COMPLETE evidence"
            )

        manifest_path = logdir / "reference-manifest.json"
        manifest = _read_json_object(manifest_path)
        for path, expected in UPSTREAM_D3_REFERENCE.items():
            actual = _manifest_value(manifest, path, manifest_path)
            if type(actual) is not type(expected) or actual != expected:
                raise AtariScoreError(
                    f"{manifest_path}: expected {'.'.join(path)}={expected!r}, "
                    f"found {actual!r}"
                )

        experiment = manifest["experiment"]
        protocol = manifest["atari_protocol"]
        assert isinstance(experiment, dict)
        assert isinstance(protocol, dict)
        seed = experiment.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AtariScoreError(
                f"{manifest_path}: experiment.seed must be a non-negative integer"
            )
        action_repeat = int(protocol["action_repeat"])
        end_frame = int(experiment["steps"]) * action_repeat

        scores_path = logdir / "scores.jsonl"
        records = _read_events(scores_path)
        episodes = []
        seen_steps = set()
        for line_number, record in enumerate(records, 1):
            if "step" not in record or "episode/score" not in record:
                raise AtariScoreError(
                    f"{scores_path}:{line_number}: expected step and "
                    "episode/score"
                )
            raw_step = record["step"]
            if isinstance(raw_step, bool) or not isinstance(raw_step, int):
                raise AtariScoreError(
                    f"{scores_path}:{line_number}: step must be an integer"
                )
            if not 0 <= raw_step <= end_frame:
                raise AtariScoreError(
                    f"{scores_path}:{line_number}: step {raw_step} is outside "
                    f"the completed 0--{end_frame} frame run"
                )
            if raw_step in seen_steps:
                raise AtariScoreError(
                    f"{scores_path}:{line_number}: duplicate episode step "
                    f"{raw_step}"
                )
            seen_steps.add(raw_step)
            raw_score = record["episode/score"]
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)
            ):
                raise AtariScoreError(
                    f"{scores_path}:{line_number}: episode/score must be finite"
                )
            episodes.append(
                {"environment_frames": raw_step, "return": float(raw_score)}
            )

        segments.append(
            {
                "source": str(scores_path),
                "environment": str(experiment["environment"]),
                "seed": seed,
                "mode": "train",
                "target_runtime": UPSTREAM_D3_TARGET_RUNTIME,
                "atari_protocol": str(protocol["name"]),
                "action_repeat": action_repeat,
                "full_action_space": protocol["actions"] == "all",
                "noop_max": int(protocol["reset_noop_max"]),
                "max_episode_frames": int(protocol["episode_frame_cap"]),
                "score_window_frames": tuple(manifest["score_window_frames"]),
                "start_frame": 0,
                "end_frame": end_frame,
                "episodes": episodes,
            }
        )
    if not segments:
        raise AtariScoreError("no upstream D3 log directories provided")
    return segments


def load_segments(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Load complete runs, including explicit checkpoint-recovery segments."""

    recovery_identity_fields = (
        "environment",
        "seed",
        "mode",
        "atari_protocol",
        "action_repeat",
        "full_action_space",
        "noop_max",
        "max_episode_frames",
        "score_window_frames",
        "actions",
        "action_meanings",
        "dino_model_id",
        "dino_checkpoint_revision",
        "dino_checkpoint_sha256",
        "trainable_parameters",
        "config",
    )

    def start_segment(path: Path, event: dict[str, object]) -> dict[str, object]:
        action_repeat = int(event["action_repeat"])
        return {
            "source": str(path),
            "environment": str(event["environment"]),
            "seed": int(event["seed"]),
            "mode": str(event["mode"]),
            "target_runtime": (
                KINDLE_TARGET_RUNTIME
                if event.get("ale_py_version") in (None, "0.12.1")
                else f"kindle-ale_py-{event['ale_py_version']}"
            ),
            "atari_protocol": str(event["atari_protocol"]),
            "action_repeat": action_repeat,
            "full_action_space": bool(event["full_action_space"]),
            "noop_max": int(event["noop_max"]),
            "max_episode_frames": int(event["max_episode_frames"]),
            "score_window_frames": tuple(event["score_window_frames"]),
            "start_frame": (
                int(event["starting_environment_step"]) * action_repeat
            ),
            "episodes": [],
            "_run_start": event,
            "_checkpoints": {},
        }

    def finish_segment(
        current: dict[str, object], end_frame: int
    ) -> dict[str, object]:
        current["end_frame"] = end_frame
        if current["end_frame"] < current["start_frame"]:
            raise AtariScoreError(
                f"{current['source']}: run ends before it starts"
            )
        current.pop("_run_start")
        current.pop("_checkpoints")
        return current

    segments = []
    for path in paths:
        current = None
        for event in _read_events(path):
            event_type = event.get("event")
            if event_type == "run_start":
                if current is not None:
                    if event.get("output_appended") is not True:
                        raise AtariScoreError(f"{path}: nested run_start event")
                    previous_start = current["_run_start"]
                    for field in recovery_identity_fields:
                        if previous_start.get(field) != event.get(field):
                            raise AtariScoreError(
                                f"{path}: checkpoint recovery changed {field}"
                            )
                    action_repeat = int(current["action_repeat"])
                    resume_step = int(event["starting_environment_step"])
                    resume_frame = resume_step * action_repeat
                    checkpoints = current["_checkpoints"]
                    if resume_step not in checkpoints:
                        raise AtariScoreError(
                            f"{path}: checkpoint recovery at step {resume_step} "
                            "has no matching checkpoint event"
                        )
                    restored_learner = int(event["starting_learner_step"])
                    if restored_learner != checkpoints[resume_step]:
                        raise AtariScoreError(
                            f"{path}: checkpoint recovery learner step "
                            f"{restored_learner} does not match "
                            f"{checkpoints[resume_step]}"
                        )
                    if resume_frame <= int(current["start_frame"]):
                        raise AtariScoreError(
                            f"{path}: checkpoint recovery did not advance the run"
                        )
                    current["episodes"] = [
                        episode
                        for episode in current["episodes"]
                        if episode["environment_frames"] <= resume_frame
                    ]
                    current["checkpoint_recovered"] = True
                    segments.append(finish_segment(current, resume_frame))
                current = start_segment(path, event)
                if event.get("output_appended") is True:
                    current["checkpoint_recovery_start"] = True
            elif event_type == "checkpoint" and current is not None:
                checkpoint_step = int(event["environment_step"])
                learner_step = int(event["learner_step"])
                previous = current["_checkpoints"].setdefault(
                    checkpoint_step, learner_step
                )
                if previous != learner_step:
                    raise AtariScoreError(
                        f"{path}: conflicting checkpoint at step {checkpoint_step}"
                    )
            elif event_type == "episode" and current is not None:
                current["episodes"].append(
                    {
                        "environment_frames": int(event["environment_frames"]),
                        "return": float(event["return"]),
                    }
                )
            elif event_type == "run_end":
                if current is None:
                    raise AtariScoreError(f"{path}: run_end without run_start")
                if str(event["environment"]) != current["environment"]:
                    raise AtariScoreError(f"{path}: environment changed within run")
                if int(event["seed"]) != current["seed"]:
                    raise AtariScoreError(f"{path}: seed changed within run")
                end_frame = int(event["environment_frames"])
                segments.append(finish_segment(current, end_frame))
                current = None
        if current is not None:
            raise AtariScoreError(f"{path}: incomplete run has no run_end event")
    if not segments:
        raise AtariScoreError("no complete Atari runs found")
    return segments


def _validate_segment(
    segment: dict[str, object], expected_protocol: str, expected_mode: str
) -> None:
    source = segment["source"]
    if segment["atari_protocol"] != expected_protocol:
        raise AtariScoreError(
            f"{source}: expected {expected_protocol!r} protocol, "
            f"found {segment['atari_protocol']!r}"
        )
    if segment["mode"] != expected_mode:
        raise AtariScoreError(
            f"{source}: expected {expected_mode!r} mode, found {segment['mode']!r}"
        )
    if tuple(segment["score_window_frames"]) != SCORE_WINDOW_FRAMES:
        raise AtariScoreError(f"{source}: unexpected score window")
    if expected_protocol not in ATARI_PROFILES:
        raise AtariScoreError(f"unknown Atari protocol {expected_protocol!r}")
    for field, expected in ATARI_PROFILES[expected_protocol].items():
        if segment[field] != expected:
            raise AtariScoreError(
                f"{source}: {expected_protocol} profile requires "
                f"{field}={expected!r}, found {segment[field]!r}"
            )


def _covers_window(intervals: list[tuple[int, int]]) -> bool:
    cursor = SCORE_WINDOW_FRAMES[0]
    for start, end in sorted(intervals):
        if end < cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= SCORE_WINDOW_FRAMES[1]:
            return True
    return False


def summarize_scores(
    segments: list[dict[str, object]],
    expected_protocol: str = "published",
    expected_mode: str = "train",
) -> dict[str, object]:
    """Compute episode means per seed, then the equally weighted seed mean."""
    grouped = defaultdict(list)
    target_runtimes = {}
    for segment in segments:
        _validate_segment(segment, expected_protocol, expected_mode)
        environment = segment["environment"]
        target_runtime = segment["target_runtime"]
        previous_runtime = target_runtimes.setdefault(environment, target_runtime)
        if previous_runtime != target_runtime:
            raise AtariScoreError(
                f"{environment}: cannot mix target runtimes "
                f"{previous_runtime!r} and {target_runtime!r}"
            )
        grouped[(segment["environment"], segment["seed"])].append(segment)

    environments = defaultdict(list)
    window_start, window_end = SCORE_WINDOW_FRAMES
    for (environment, seed), seed_segments in sorted(grouped.items()):
        intervals = [
            (int(segment["start_frame"]), int(segment["end_frame"]))
            for segment in seed_segments
        ]
        if not _covers_window(intervals):
            raise AtariScoreError(
                f"{environment} seed {seed}: logs do not cover the full "
                f"{window_start}--{window_end} frame score window"
            )

        episodes_by_frame = {}
        for segment in seed_segments:
            for episode in segment["episodes"]:
                frame = int(episode["environment_frames"])
                if not window_start <= frame <= window_end:
                    continue
                if frame in episodes_by_frame:
                    raise AtariScoreError(
                        f"{environment} seed {seed}: duplicate episode at frame {frame}"
                    )
                episode_return = float(episode["return"])
                if not math.isfinite(episode_return):
                    raise AtariScoreError(
                        f"{segment['source']}: episode at frame {frame} has a "
                        "non-finite return"
                    )
                episodes_by_frame[frame] = episode_return
        if not episodes_by_frame:
            raise AtariScoreError(
                f"{environment} seed {seed}: complete window contains no episodes"
            )
        returns = [episodes_by_frame[frame] for frame in sorted(episodes_by_frame)]
        seed_result = {
            "seed": seed,
            "score": sum(returns) / len(returns),
            "completed_episodes": len(returns),
            "segment_count": len(seed_segments),
        }
        checkpoint_recoveries = sum(
            bool(segment.get("checkpoint_recovery_start"))
            for segment in seed_segments
        )
        if checkpoint_recoveries:
            seed_result["checkpoint_recoveries"] = checkpoint_recoveries
        environments[environment].append(seed_result)

    environment_summaries = {}
    protocol_targets = ATARI_TARGETS.get(expected_protocol, {})
    for environment, seeds in sorted(environments.items()):
        seed_mean = sum(seed["score"] for seed in seeds) / len(seeds)
        target_runtime = target_runtimes[environment]
        environment_targets = dict(protocol_targets.get(environment, {}))
        if target_runtime != KINDLE_TARGET_RUNTIME:
            environment_targets.pop("matched_random_3_seed", None)
        environment_targets.update(
            ATARI_TARGET_OVERRIDES.get(target_runtime, {})
            .get(expected_protocol, {})
            .get(environment, {})
        )
        targets = {
            name: {
                "score": target,
                "delta": seed_mean - target,
                "met": seed_mean >= target,
            }
            for name, target in environment_targets.items()
        }
        environment_summaries[environment] = {
            "target_runtime": target_runtime,
            "seed_results": seeds,
            "seed_mean": seed_mean,
            "seed_count": len(seeds),
            "targets": targets,
        }

    return {
        "atari_protocol": expected_protocol,
        "mode": expected_mode,
        "score_window_frames": list(SCORE_WINDOW_FRAMES),
        "environments": environment_summaries,
    }


def compare_runtime_scores(
    kindle: dict[str, object], upstream_d3: dict[str, object]
) -> dict[str, object]:
    """Compare Kindle and upstream D3 after removing their ALE random floors."""
    identity_fields = ("atari_protocol", "mode", "score_window_frames")
    for field in identity_fields:
        if kindle.get(field) != upstream_d3.get(field):
            raise AtariScoreError(
                f"cannot compare score summaries with different {field}"
            )

    kindle_environments = kindle["environments"]
    upstream_environments = upstream_d3["environments"]
    if not isinstance(kindle_environments, dict) or not isinstance(
        upstream_environments, dict
    ):
        raise AtariScoreError("score summaries have invalid environment mappings")
    if kindle_environments.keys() != upstream_environments.keys():
        raise AtariScoreError("score summaries cover different environments")

    comparisons = {}
    for environment in sorted(kindle_environments):
        kindle_result = kindle_environments[environment]
        upstream_result = upstream_environments[environment]
        if not isinstance(kindle_result, dict) or not isinstance(upstream_result, dict):
            raise AtariScoreError(f"{environment}: invalid score summary")
        if kindle_result.get("target_runtime") != KINDLE_TARGET_RUNTIME:
            raise AtariScoreError(
                f"{environment}: first summary is not the Kindle runtime"
            )
        if upstream_result.get("target_runtime") != UPSTREAM_D3_TARGET_RUNTIME:
            raise AtariScoreError(
                f"{environment}: second summary is not the pinned upstream runtime"
            )

        try:
            kindle_score = float(kindle_result["seed_mean"])
            upstream_score = float(upstream_result["seed_mean"])
            kindle_random = float(
                kindle_result["targets"]["matched_random_3_seed"]["score"]
            )
            upstream_random = float(
                upstream_result["targets"]["matched_random_3_seed"]["score"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AtariScoreError(
                f"{environment}: summaries lack matched random targets"
            ) from error

        kindle_gain = kindle_score - kindle_random
        upstream_gain = upstream_score - upstream_random
        comparisons[environment] = {
            "kindle_score": kindle_score,
            "upstream_d3_score": upstream_score,
            "raw_score_delta": kindle_score - upstream_score,
            "kindle_random_score": kindle_random,
            "upstream_d3_random_score": upstream_random,
            "kindle_gain_over_random": kindle_gain,
            "upstream_d3_gain_over_random": upstream_gain,
            "random_adjusted_delta": kindle_gain - upstream_gain,
        }

    return {
        "atari_protocol": kindle["atari_protocol"],
        "mode": kindle["mode"],
        "score_window_frames": kindle["score_window_frames"],
        "kindle": kindle_environments,
        "upstream_d3": upstream_environments,
        "comparisons": comparisons,
    }
