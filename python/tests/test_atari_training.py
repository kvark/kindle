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
    assert summary["intervals"]["intrinsic_reward"] is None
    assert summary["intervals"]["positive_reward_events"] is None
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
    assert summary["actions"]["within_effective_group_entropy"] == 0


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


def test_reward_prediction_means_are_weighted_by_sample_count(tmp_path) -> None:
    path = tmp_path / "reward-calibration.jsonl"
    counts = [0] * 18
    counts[0] = 1_000
    first = learner(250, 1, 10)
    first["report"]["world"].update(
        positive_reward_prediction_mean=-0.5,
        positive_reward_count=1,
        zero_reward_prediction_mean=-0.01,
        zero_reward_count=999,
        negative_reward_prediction_mean=-0.8,
        negative_reward_count=24,
        rewarded_prediction_mean=-0.75,
        rewarded_count=25,
    )
    second = learner(750, 2, 10)
    second["report"]["world"].update(
        positive_reward_prediction_mean=0.25,
        positive_reward_count=3,
        zero_reward_prediction_mean=-0.02,
        zero_reward_count=997,
        negative_reward_prediction_mean=-0.9,
        negative_reward_count=21,
        rewarded_prediction_mean=-0.75,
        rewarded_count=24,
    )
    write_events(
        path,
        [run_start(), first, second, interval(1_000, counts, updates=2)],
    )

    predictions = summarize_training_window(path)["learner"]["reward_predictions"]
    assert predictions["positive"] == {
        "reports": 2,
        "sampled_update_groups": 2,
        "prediction_above_zero_groups": 1,
        "sample_count": 4,
        "count_weighted_mean": pytest.approx(0.0625),
    }
    assert predictions["negative"]["sample_count"] == 45
    assert predictions["negative"]["count_weighted_mean"] == pytest.approx(
        -0.8466666667
    )
    assert predictions["zero"]["sample_count"] == 1_996
    assert predictions["rewarded"]["sample_count"] == 49


def test_separates_game_reward_from_intrinsic_reward_and_sparse_events(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    counts = [1_000] + [0] * 17
    start = run_start()
    start["config"] = {"extrinsic_reward_scale": 1.0, "intrinsic_reward_scale": 0.1}
    start["model_provenance"] = {"visitation_hash_version": 2}
    first = interval(1_000, counts, reward=100, updates=0)
    second = interval(2_000, counts, reward=-1, updates=0)
    first.update(intrinsic_reward=250.5, positive_reward_events=1, negative_reward_events=0)
    second.update(intrinsic_reward=100.0, positive_reward_events=0, negative_reward_events=1)
    write_events(path, [start, first, second])
    summary = summarize_training_window(path, default_window_steps=2_000)
    assert summary["intervals"]["reward"] == 99
    assert summary["intervals"]["intrinsic_reward"] == 350.5
    assert summary["intervals"]["positive_reward_events"] == 1
    assert summary["intervals"]["negative_reward_events"] == 1
    assert summary["config"] == start["config"]
    assert summary["model_provenance"] == start["model_provenance"]


def test_rejects_partial_intrinsic_interval_coverage(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    counts = [1_000] + [0] * 17
    first = interval(1_000, counts, updates=0)
    first["intrinsic_reward"] = 0.0
    write_events(path, [run_start(), first, interval(2_000, counts, updates=0)])
    with pytest.raises(AtariTrainingError, match="incomplete interval coverage for intrinsic_reward"):
        summarize_training_window(path, default_window_steps=2_000)


@pytest.mark.parametrize("value", [True, None, -1, 0.5, 1_001])
def test_rejects_invalid_sparse_event_counts(tmp_path, value) -> None:
    path = tmp_path / "events.jsonl"
    record = interval(1_000, [1_000] + [0] * 17, updates=0)
    record["positive_reward_events"] = value
    write_events(path, [run_start(), record])
    with pytest.raises(AtariTrainingError, match="positive_reward_events"):
        summarize_training_window(path)
