import json

import pytest

from kindle._atari_training import AtariTrainingError, summarize_training_window


ACTION_MEANINGS = [
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
]


def write_events(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def run_start():
    return {
        "event": "run_start",
        "environment": "ALE/Pong-v5",
        "steps": 10_000,
        "seed": 7,
        "atari_protocol": "published",
        "action_repeat": 4,
        "actions": 18,
        "action_meanings": ACTION_MEANINGS,
    }


def interval(step, counts, *, reward=0, updates=1):
    return {
        "event": "interval",
        "environment_step": step,
        "interval_steps": 1_000,
        "reward": reward,
        "completed_episodes": 1,
        "learner_updates": updates,
        "action_counts": counts,
    }


def learner(step, learner_step, loss):
    return {
        "event": "learner",
        "environment_step": step,
        "report": {
            "learner_step": learner_step,
            "world": {"reconstruction_loss": loss},
            "behavior": {"policy_entropy": loss / 10},
        },
    }


def test_summarizes_latest_complete_window_and_pong_aliases(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    first = [0] * 18
    first[2] = 1_000  # UP is an environment-equivalent NOOP in Pong.
    second = [0] * 18
    second[11] = 1_000
    write_events(
        path,
        [
            run_start(),
            learner(500, 1, 20),
            interval(1_000, first, reward=-2),
            {
                "event": "episode",
                "environment_step": 1_500,
                "return": -18,
                "length": 1_500,
            },
            learner(1_500, 2, 10),
            interval(2_000, second, reward=-1),
        ],
    )

    summary = summarize_training_window(path, default_window_steps=2_000)
    assert summary["window"]["end_step_inclusive"] == 2_000
    assert summary["episodes"] == {
        "count": 1,
        "returns": [-18.0],
        "mean_return": -18.0,
        "mean_length": 1_500.0,
    }
    assert summary["intervals"]["reward"] == -3.0
    assert summary["learner"]["events"] == 2
    assert summary["learner"]["metric_means"] == {
        "behavior/policy_entropy": 1.5,
        "learner_step": 1.5,
        "world/reconstruction_loss": 15.0,
    }
    effective = summary["actions"]["effective"]
    assert effective["names"] == [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE",
    ]
    assert effective["counts"] == [1_000, 0, 0, 0, 1_000, 0]
    assert effective["entropy"] == pytest.approx(0.6931471805599453)


def test_rejects_window_without_exact_interval_coverage(tmp_path) -> None:
    path = tmp_path / "gap.jsonl"
    counts = [0] * 18
    counts[0] = 1_000
    write_events(path, [run_start(), interval(1_000, counts, updates=0)])

    with pytest.raises(AtariTrainingError, match="coverage ends at 1000"):
        summarize_training_window(path, start_step=0, end_step=2_000)


def test_rejects_recovered_or_concatenated_log(tmp_path) -> None:
    path = tmp_path / "recovered.jsonl"
    counts = [0] * 18
    counts[0] = 1_000
    write_events(path, [run_start(), interval(1_000, counts, updates=0), run_start()])

    with pytest.raises(AtariTrainingError, match="one uninterrupted run_start"):
        summarize_training_window(path)


def test_rejects_learner_count_disagreement(tmp_path) -> None:
    path = tmp_path / "updates.jsonl"
    counts = [0] * 18
    counts[0] = 1_000
    write_events(path, [run_start(), interval(1_000, counts, updates=1)])

    with pytest.raises(AtariTrainingError, match="0 learner events"):
        summarize_training_window(path)


def test_rejects_non_integer_window_bounds(tmp_path) -> None:
    path = tmp_path / "bounds.jsonl"
    counts = [0] * 18
    counts[0] = 1_000
    write_events(path, [run_start(), interval(1_000, counts, updates=0)])

    with pytest.raises(AtariTrainingError, match="window steps must be integers"):
        summarize_training_window(path, start_step=0.5, end_step=1_000)
