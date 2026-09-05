"""Constant-memory diagnostics for Kindle Atari training windows."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator


class AtariTrainingError(ValueError):
    """Raised when a training log cannot prove the requested window."""


def _events(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise AtariTrainingError(f"missing training log: {path}") from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise AtariTrainingError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(event, dict):
                raise AtariTrainingError(
                    f"{path}:{line_number}: JSONL event must be an object"
                )
            yield line_number, event


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise AtariTrainingError(f"learner report contains non-finite value {value}")
    return number


def _numeric_leaves(
    value: object, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, float]]:
    number = _finite_number(value)
    if number is not None:
        yield "/".join(prefix), number
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _numeric_leaves(child, (*prefix, str(key)))


def _pong_action_name(action: str) -> str:
    if action.startswith("DOWN"):
        action = action[4:]
    elif action.startswith("UP"):
        action = action[2:]
    return action or "NOOP"


def _effective_actions(
    environment: str,
    action_meanings: list[str],
    action_counts: list[int],
) -> dict[str, object]:
    if environment != "ALE/Pong-v5":
        return {
            "names": action_meanings,
            "counts": action_counts,
            "fractions": _fractions(action_counts),
            "entropy": _entropy(action_counts),
        }

    names: list[str] = []
    counts: list[int] = []
    indices: dict[str, int] = {}
    for meaning, count in zip(action_meanings, action_counts):
        name = _pong_action_name(meaning)
        if name not in indices:
            indices[name] = len(names)
            names.append(name)
            counts.append(0)
        counts[indices[name]] += count
    return {
        "names": names,
        "counts": counts,
        "fractions": _fractions(counts),
        "entropy": _entropy(counts),
    }


def _fractions(counts: list[int]) -> list[float]:
    total = sum(counts)
    return [count / total for count in counts] if total else [0.0] * len(counts)


def _entropy(counts: list[int]) -> float:
    return -sum(
        fraction * math.log(fraction)
        for fraction in _fractions(counts)
        if fraction
    )


def summarize_training_window(
    path: Path,
    *,
    start_step: int | None = None,
    end_step: int | None = None,
    default_window_steps: int = 5_000,
) -> dict[str, object]:
    """Summarize one exact `(start_step, end_step]` training window.

    The first pass chooses the latest complete interval when ``end_step`` is
    omitted. The second pass aggregates only records inside the fixed window,
    so a concurrently appended log cannot change the requested result.
    """

    path = Path(path)
    run_start = None
    run_start_count = 0
    run_end_step = None
    latest_interval_step = None
    for _, event in _events(path):
        event_type = event.get("event")
        if event_type == "run_start":
            run_start_count += 1
            if run_start is None:
                run_start = event
        elif event_type == "interval":
            latest_interval_step = int(event["environment_step"])
        elif event_type == "run_end":
            run_end_step = int(event["environment_step"])

    if run_start is None:
        raise AtariTrainingError(f"{path}: no run_start event")
    if run_start_count != 1:
        raise AtariTrainingError(
            f"{path}: expected one uninterrupted run_start, found {run_start_count}"
        )
    if latest_interval_step is None:
        raise AtariTrainingError(f"{path}: no complete interval events")
    if end_step is None:
        end_step = latest_interval_step
    if start_step is None:
        start_step = max(0, end_step - default_window_steps)
    if (
        isinstance(start_step, bool)
        or not isinstance(start_step, int)
        or isinstance(end_step, bool)
        or not isinstance(end_step, int)
    ):
        raise AtariTrainingError("window steps must be integers")
    if not 0 <= start_step < end_step:
        raise AtariTrainingError(
            f"invalid training window ({start_step}, {end_step}]"
        )
    declared_steps = int(run_start["steps"])
    if end_step > declared_steps:
        raise AtariTrainingError(
            f"window ends at {end_step}, beyond declared run length {declared_steps}"
        )

    environment = str(run_start["environment"])
    action_meanings = [str(value) for value in run_start["action_meanings"]]
    action_count = int(run_start["actions"])
    if len(action_meanings) != action_count:
        raise AtariTrainingError(
            f"run_start has {len(action_meanings)} meanings for {action_count} actions"
        )

    intervals: list[tuple[int, int]] = []
    interval_reward = 0.0
    interval_episodes = 0
    interval_updates = 0
    optional_interval_fields = (
        "intrinsic_reward", "positive_reward_events", "negative_reward_events",
    )
    optional_interval_sums = dict.fromkeys(optional_interval_fields, 0)
    optional_interval_counts = dict.fromkeys(optional_interval_fields, 0)
    action_counts = [0] * action_count
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    learner_events = 0
    first_learner_step = None
    last_learner_step = None
    metric_sums: defaultdict[str, float] = defaultdict(float)
    metric_counts: defaultdict[str, int] = defaultdict(int)
    reward_prediction_fields = {
        "positive": (
            "positive_reward_prediction_mean",
            "positive_reward_count",
        ),
        "zero": ("zero_reward_prediction_mean", "zero_reward_count"),
        "negative": (
            "negative_reward_prediction_mean",
            "negative_reward_count",
        ),
        "rewarded": ("rewarded_prediction_mean", "rewarded_count"),
    }
    reward_prediction_reports: defaultdict[str, int] = defaultdict(int)
    reward_prediction_groups: defaultdict[str, int] = defaultdict(int)
    reward_prediction_positive_groups: defaultdict[str, int] = defaultdict(int)
    reward_prediction_samples: defaultdict[str, int] = defaultdict(int)
    reward_prediction_weighted_sums: defaultdict[str, float] = defaultdict(float)

    for line_number, event in _events(path):
        event_type = event.get("event")
        if event_type == "interval":
            interval_end = int(event["environment_step"])
            interval_size = int(event["interval_steps"])
            interval_start = interval_end - interval_size
            overlaps = interval_end > start_step and interval_start < end_step
            contained = interval_start >= start_step and interval_end <= end_step
            if overlaps and not contained:
                raise AtariTrainingError(
                    f"{path}:{line_number}: interval ({interval_start}, "
                    f"{interval_end}] crosses requested window boundary"
                )
            if contained:
                intervals.append((interval_start, interval_end))
                interval_reward += float(event["reward"])
                interval_episodes += int(event["completed_episodes"])
                interval_updates += int(event["learner_updates"])
                for name in optional_interval_fields:
                    if name not in event:
                        continue
                    value = _finite_number(event[name])
                    if value is None:
                        raise AtariTrainingError(
                            f"{path}:{line_number}: {name} must be finite"
                        )
                    if name.endswith("_events") and (
                        not value.is_integer() or not 0 <= value <= interval_size
                    ):
                        raise AtariTrainingError(
                            f"{path}:{line_number}: invalid {name} count"
                        )
                    optional_interval_sums[name] += (
                        int(value) if name.endswith("_events") else value
                    )
                    optional_interval_counts[name] += 1
                counts = event["action_counts"]
                if not isinstance(counts, list) or len(counts) != action_count:
                    raise AtariTrainingError(
                        f"{path}:{line_number}: invalid interval action counts"
                    )
                for index, count in enumerate(counts):
                    action_counts[index] += int(count)
        elif event_type == "episode":
            step = int(event["environment_step"])
            if start_step < step <= end_step:
                episode_returns.append(float(event["return"]))
                episode_lengths.append(int(event["length"]))
        elif event_type == "learner":
            step = int(event["environment_step"])
            if start_step < step <= end_step:
                report = event.get("report")
                if not isinstance(report, dict):
                    raise AtariTrainingError(
                        f"{path}:{line_number}: learner report must be an object"
                    )
                learner_events += 1
                learner_step = int(report["learner_step"])
                if first_learner_step is None:
                    first_learner_step = learner_step
                last_learner_step = learner_step
                for name, value in _numeric_leaves(report):
                    metric_sums[name] += value
                    metric_counts[name] += 1
                world = report.get("world")
                if isinstance(world, dict):
                    for label, (mean_field, count_field) in (
                        reward_prediction_fields.items()
                    ):
                        prediction = _finite_number(world.get(mean_field))
                        raw_count = _finite_number(world.get(count_field))
                        if prediction is None or raw_count is None:
                            continue
                        if raw_count < 0 or not raw_count.is_integer():
                            raise AtariTrainingError(
                                f"{path}:{line_number}: {count_field} must be a "
                                "non-negative integer"
                            )
                        count = int(raw_count)
                        reward_prediction_reports[label] += 1
                        reward_prediction_samples[label] += count
                        reward_prediction_weighted_sums[label] += prediction * count
                        if count:
                            reward_prediction_groups[label] += 1
                            if prediction > 0:
                                reward_prediction_positive_groups[label] += 1

    intervals.sort()
    cursor = start_step
    for interval_start, interval_end in intervals:
        if interval_start != cursor:
            raise AtariTrainingError(
                f"{path}: interval coverage jumps from {cursor} to {interval_start}"
            )
        cursor = interval_end
    if cursor != end_step:
        raise AtariTrainingError(
            f"{path}: interval coverage ends at {cursor}, expected {end_step}"
        )
    expected_actions = end_step - start_step
    if sum(action_counts) != expected_actions:
        raise AtariTrainingError(
            f"{path}: action counts sum to {sum(action_counts)}, "
            f"expected {expected_actions}"
        )
    if learner_events != interval_updates:
        raise AtariTrainingError(
            f"{path}: found {learner_events} learner events but intervals report "
            f"{interval_updates} updates"
        )
    for name, count in optional_interval_counts.items():
        if count not in (0, len(intervals)):
            raise AtariTrainingError(f"{path}: incomplete interval coverage for {name}")

    metric_means = {
        name: metric_sums[name] / metric_counts[name]
        for name in sorted(metric_sums)
    }
    reward_predictions = {
        label: {
            "reports": reward_prediction_reports[label],
            "sampled_update_groups": reward_prediction_groups[label],
            "prediction_above_zero_groups": reward_prediction_positive_groups[label],
            "sample_count": reward_prediction_samples[label],
            "count_weighted_mean": (
                reward_prediction_weighted_sums[label]
                / reward_prediction_samples[label]
                if reward_prediction_samples[label]
                else None
            ),
        }
        for label in reward_prediction_fields
        if reward_prediction_reports[label]
    }
    action_entropy = _entropy(action_counts)
    effective_actions = _effective_actions(
        environment, action_meanings, action_counts
    )
    effective_entropy = float(effective_actions["entropy"])
    return {
        "source": str(path),
        "environment": environment,
        "seed": int(run_start["seed"]),
        "atari_protocol": str(run_start["atari_protocol"]),
        "config": run_start.get("config"),
        "model_provenance": run_start.get("model_provenance"),
        "run_complete": run_end_step is not None,
        "run_end_step": run_end_step,
        "window": {
            "start_step_exclusive": start_step,
            "end_step_inclusive": end_step,
            "steps": expected_actions,
            "start_frame_exclusive": start_step * int(run_start["action_repeat"]),
            "end_frame_inclusive": end_step * int(run_start["action_repeat"]),
        },
        "episodes": {
            "count": len(episode_returns),
            "returns": episode_returns,
            "mean_return": (
                sum(episode_returns) / len(episode_returns)
                if episode_returns
                else None
            ),
            "mean_length": (
                sum(episode_lengths) / len(episode_lengths)
                if episode_lengths
                else None
            ),
        },
        "intervals": {
            "count": len(intervals),
            "reward": interval_reward,
            "completed_episodes": interval_episodes,
            "learner_updates": interval_updates,
            **{
                name: (optional_interval_sums[name] if optional_interval_counts[name] else None)
                for name in optional_interval_fields
            },
        },
        "actions": {
            "names": action_meanings,
            "counts": action_counts,
            "fractions": _fractions(action_counts),
            "entropy": action_entropy,
            "effective": effective_actions,
            # H(raw action) - H(environment-distinct action) is the expected
            # entropy among aliases of the selected effective control.
            "within_effective_group_entropy": (
                action_entropy - effective_entropy
            ),
        },
        "learner": {
            "events": learner_events,
            "first_step": first_learner_step,
            "last_step": last_learner_step,
            "metric_means": metric_means,
            "reward_predictions": reward_predictions,
        },
    }
