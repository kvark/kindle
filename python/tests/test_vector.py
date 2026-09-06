import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import kindle
from kindle._vector_audit import audit, require_numbers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari_vector
import profile_atari_vector


def test_vector_api_rejects_invalid_config_before_loading_weights():
    with pytest.raises(ValueError, match="positive"):
        kindle.VectorAgent("unused", 0, {})
    config = kindle.default_config(18)
    config["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        kindle.VectorAgent("unused", 4, config)


@pytest.mark.parametrize("args, message", [
    (["--steps", "7", "--num-envs", "2"], "multiple"),
    (["--num-envs", "0"], "multiple"),
    (["--greedy"], "frozen evaluation"),
    (["--world-microbatch-size", "0"], "must be positive"),
    (["--restore", "unused", "--batch-size", "32"], "overrides require a fresh run"),
])
def test_runner_rejects_ambiguous_budgets_before_gpu(monkeypatch, capsys, tmp_path, args, message):
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "unused", "--output", str(tmp_path / "log.jsonl"), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("args, message", [
    (["--batch-size", "0"], "must be positive"),
    (["--num-envs", "2", "2"], "distinct"),
    (["--batch-size", "64"], "warmup"),
])
def test_profiler_rejects_unusable_windows_before_starting_jobs(monkeypatch, capsys, tmp_path, args, message):
    directory = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["profile_atari_vector.py", "unused", str(directory), *args])
    with pytest.raises(SystemExit) as error:
        profile_atari_vector.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not directory.exists()


def test_profiler_retains_failed_jobs_without_claiming_a_completed_matrix(monkeypatch, tmp_path):
    directory = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["profile_atari_vector.py", "unused", str(directory), "--num-envs", "2"])
    monitor = SimpleNamespace(terminate=lambda: None, wait=lambda **_: None)
    monkeypatch.setattr(profile_atari_vector.subprocess, "Popen", lambda *_, **__: monitor)
    monkeypatch.setattr(profile_atari_vector.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=1))
    with pytest.raises(SystemExit) as error:
        profile_atari_vector.main()
    assert error.value.code == 1
    results = json.loads((directory / "summary.json").read_text())
    assert results[0]["status"] == "failed"
    assert results[0]["num_envs"] == 2 and results[0]["exit_code"] == 1
    assert "actions_per_second" not in results[0]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.0", True])
def test_learner_metrics_cannot_hide_nonfinite_values_as_null(bad):
    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        require_numbers({"world": {"total_loss": bad}})


def fixture_events():
    config = dict(batch_size=1, batch_length=2, replay_context=1, replay_capacity=32, train_ratio=2.0, action_count=3)
    events = [dict(event="run_start", protocol="kindle-vector-v1", num_envs=2, steps=6,
                   environment_seeds=[0, 1000003], mode="train", config=config,
                   starting_learner_step=0, starting_environment_step=0, action_repeat=4)]
    updates = 0
    for tick in range(1, 4):
        events.append(dict(event="transition", run_step=tick * 2, vector_tick=tick,
                           actions=[1, 2], rewards=[1.0, 0.0], stored_rewards=[[1.0, 0.0], [0.0, 0.0]],
                           terminated=[tick == 2, False], truncated=[False, False],
                           executed_action_frames=[tick * 4, tick * 4]))
        for _ in range(0 if tick == 1 else 1 if tick == 2 else 2):
            updates += 1
            events.append(dict(event="learner", run_step=tick * 2,
                               report=dict(learner_step=updates, replay_len=2 + tick * 2 + int(tick == 3),
                                           world={"total_loss": 1.0}, behavior={"total_loss": 2.0}, timing={"total_seconds": 0.5})))
        if tick == 2:
            events.append(dict(event="episode", stream=0, run_step=4, stream_step=2, episode=0,
                               episode_return=2.0, episode_length=2, terminated=True, truncated=False))
            events.append(dict(event="reset", run_step=4, streams=[0]))
    events.append(dict(event="run_end", run_step=6, vector_ticks=3, environment_step=6, learner_step=3,
                       replay_len=9, training_debt=0.0, executed_action_frames=[12, 12],
                       total_rewards=[3.0, 0.0], episode_counts=[1, 0], partial_returns=[1.0, 0.0],
                       partial_lengths=[1, 3], stage_seconds={"act": 0.1}, elapsed_seconds=2.0,
                       actions_per_second=3.0, aggregate_simulated_wall_ratio=0.2,
                       per_stream_simulated_wall_ratio=[0.1, 0.1],
                       reason="budget_complete", learner_updates=3, completed_games=1,
                       natural_wins=1, mean_completed_return=2.0))
    return events


def write_log(tmp_path, events):
    path = tmp_path / "vector.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def test_vector_accounting_keeps_reset_records_out_of_action_credit(tmp_path):
    result = audit(write_log(tmp_path, fixture_events()))
    assert result["actions"] == 6
    assert result["updates"] == 3
    assert result["natural_wins"] == 1


@pytest.mark.parametrize("mutation", ["lost_update", "wrong_reset", "null_loss", "wrong_reward", "ticks_as_actions", "aggregate_as_per_stream", "missing_end"])
def test_vector_audit_rejects_broken_ledgers(tmp_path, mutation):
    events = copy.deepcopy(fixture_events())
    if mutation == "lost_update":
        events.pop(next(i for i, e in enumerate(events) if e["event"] == "learner"))
    elif mutation == "wrong_reset":
        next(e for e in events if e["event"] == "reset")["streams"] = [1]
    elif mutation == "null_loss":
        next(e for e in events if e["event"] == "learner")["report"]["world"]["total_loss"] = None
    elif mutation == "wrong_reward":
        events[-1]["total_rewards"][0] = 4.0
    elif mutation == "ticks_as_actions":
        events[1]["run_step"] = 1
    elif mutation == "aggregate_as_per_stream":
        events[-1]["per_stream_simulated_wall_ratio"] = [0.2, 0.2]
    else:
        events.pop()
    with pytest.raises(ValueError):
        audit(write_log(tmp_path, events))
