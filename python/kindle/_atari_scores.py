"""Strict aggregation of Kindle Atari JSONL score windows."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SCORE_WINDOW_FRAMES = (350_000, 400_000)

PUBLISHED_PROFILE = {
    "action_repeat": 4,
    "full_action_space": True,
    "noop_max": 0,
    "max_episode_frames": 100_000,
}

# The 12M values are the rounded DreamerV3XS results reported by Wang et al.
# (ICLR 2025). The released comparison does not include its raw XS curves, so
# these remain secondary targets. The 200M values come from the pinned D3 score
# artifact and use the exact score window above.
ATARI_TARGETS = {
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
}


class AtariScoreError(ValueError):
    """Raised when logs cannot prove a protocol-complete score."""


def _read_events(path: Path) -> list[dict[str, object]]:
    events = []
    with path.open(encoding="utf-8") as stream:
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
            events.append(event)
    return events


def load_segments(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Load complete run segments, including multiple appended runs per file."""
    segments = []
    for path in paths:
        current = None
        for event in _read_events(path):
            event_type = event.get("event")
            if event_type == "run_start":
                if current is not None:
                    raise AtariScoreError(f"{path}: nested run_start event")
                action_repeat = int(event["action_repeat"])
                current = {
                    "source": str(path),
                    "environment": str(event["environment"]),
                    "seed": int(event["seed"]),
                    "mode": str(event["mode"]),
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
                }
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
                current["end_frame"] = int(event["environment_frames"])
                if current["end_frame"] < current["start_frame"]:
                    raise AtariScoreError(f"{path}: run ends before it starts")
                segments.append(current)
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
    if expected_protocol == "published":
        for field, expected in PUBLISHED_PROFILE.items():
            if segment[field] != expected:
                raise AtariScoreError(
                    f"{source}: published profile requires {field}={expected!r}, "
                    f"found {segment[field]!r}"
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
    for segment in segments:
        _validate_segment(segment, expected_protocol, expected_mode)
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
                episodes_by_frame[frame] = float(episode["return"])
        if not episodes_by_frame:
            raise AtariScoreError(
                f"{environment} seed {seed}: complete window contains no episodes"
            )
        returns = [episodes_by_frame[frame] for frame in sorted(episodes_by_frame)]
        environments[environment].append(
            {
                "seed": seed,
                "score": sum(returns) / len(returns),
                "completed_episodes": len(returns),
                "segment_count": len(seed_segments),
            }
        )

    environment_summaries = {}
    for environment, seeds in sorted(environments.items()):
        seed_mean = sum(seed["score"] for seed in seeds) / len(seeds)
        targets = {
            name: {
                "score": target,
                "delta": seed_mean - target,
                "met": seed_mean >= target,
            }
            for name, target in ATARI_TARGETS.get(environment, {}).items()
        }
        environment_summaries[environment] = {
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
