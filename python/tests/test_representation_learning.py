import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from summarize_representation_learning import markdown, read_run, summarize


def fixture(seed=1009, *, upstream=False, ticks=4, reward=1.):
    config = dict(batch_size=16, batch_length=64, train_ratio=256)
    start = dict(event="run_start", seed=seed, num_envs=6, steps=6*ticks,
                 environment_seeds=[seed+i*1000003 for i in range(6)], full_action_space=True,
                 sticky_actions=.25, action_repeat=4, noop_max=0, max_episode_frames=100000)
    if upstream:
        start.update(game="pong", protocol="phase2-matched-actions-v1", observation_size=64,
                     reward_action_aids="none", compute_dtype="float32", **config)
    else:
        start.update(environment="ALE/Pong-v5", config=dict(seed=seed, **config), mode="train",
                     starting_environment_step=0, starting_learner_step=0, restored_checkpoint=None,
                     observation_size="native")
    rows, returns, lengths, counts = [start], [0.] * 6, [0] * 6, [0] * 6
    for tick in range(1, ticks+1):
        done = tick % 3 == 0
        rows.append(dict(event="transition", run_step=6*tick, actions=[0]*6, rewards=[reward]*6,
                         terminated=[done]*6, truncated=[False]*6))
        updates = 1 + (6*tick-6)//4
        rows.append(dict(event="learner", run_step=6*tick))
        for i in range(6):
            returns[i] += reward
            lengths[i] += 1
            if done:
                rows.append(dict(event="episode", stream=i, episode=counts[i], run_step=6*tick,
                                 episode_return=returns[i], episode_length=lengths[i],
                                 terminated=True, truncated=False, elapsed_seconds=float(tick)))
                counts[i] += 1
                returns[i], lengths[i] = 0., 0
        rows.append(dict(event="progress", run_step=6*tick, learner_step=updates, elapsed_seconds=float(tick),
                         total_rewards=[reward*tick]*6, episode_counts=counts.copy(),
                         partial_returns=returns.copy(), partial_lengths=lengths.copy()))
    rows.append(rows[-1] | dict(event="run_end", reason="budget_complete", learner_updates=updates,
                                elapsed_seconds=float(ticks)+.5))
    return rows


def write(tmp_path, rows, name="log.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.mark.parametrize("upstream", [False, True])
def test_learning_reader_reconciles_complete_episodes_and_partial_tails(tmp_path, upstream):
    path = write(tmp_path, fixture(upstream=upstream))
    run = read_run(path)
    assert len(run["episodes"]) == 6
    assert run["final"]["partial_returns"] == [1.]*6
    assert run["final"]["partial_lengths"] == [1]*6
    assert run["curve"][-1]["score"] == 3
    assert [p["actions"] for p in run["curve"]] == [6, 12, 18, 24]
    assert run["curve"][-1]["seconds"] == 4.5
    assert run["curve"][0]["score"] is None
    assert len(run["source_sha256"]) == 64 and run["first_training_action"] == 6


@pytest.mark.parametrize("corruption", ["missing_end", "missing_transition", "missing_episode", "reward", "count", "time", "updates", "nan", "extra_end"])
def test_learning_reader_rejects_incomplete_or_inconsistent_logs(tmp_path, corruption):
    rows = fixture()
    if corruption == "missing_end":
        rows.pop()
    elif corruption == "missing_transition":
        del rows[1]
    elif corruption == "missing_episode":
        rows.pop(next(i for i, row in enumerate(rows) if row["event"] == "episode"))
    elif corruption == "reward":
        next(row for row in rows if row["event"] == "episode")["episode_return"] = 10
    elif corruption == "count":
        rows[-1]["episode_counts"] = [2]*6
    elif corruption == "time":
        rows[-1]["elapsed_seconds"] = -1
    elif corruption == "updates":
        rows[-1].update(learner_step=6, learner_updates=6)
    elif corruption == "nan":
        rows[-1]["elapsed_seconds"] = float("nan")
    else:
        rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError):
        read_run(write(tmp_path, rows))


def test_learning_summary_uses_seed_not_episode_bootstrap_and_keeps_failures(tmp_path):
    inputs = []
    for seed, reward in [(1009, -1.), (2017, 0.), (3019, 2.)]:
        inputs.append(("pretrained_tiny", write(tmp_path, fixture(seed, reward=reward), f"{seed}.jsonl")))
    result = summarize(inputs, budget=24)
    row = result["results"][0]
    assert result["status"] == "partial_learning_comparison" and not result["phase2_complete"]
    assert row["aggregate"]["final_score"] == dict(mean=1., ci95=[-3., 6.], seeds=[-3., 0., 6.])
    assert row["aggregate"]["final_hns"]["mean"] == pytest.approx((1+20.7)/35.3)
    assert row["aggregate"]["by_time"][0]["seconds"] == 3
    assert len(row["runs"]) == 3 and all(len(run["episodes"]) == 6 for run in row["runs"])
    assert "Human-normalized" in markdown(result, "data.json")
    partial = summarize(inputs[:2], budget=24)
    assert partial["results"][0]["aggregate"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize(inputs + inputs[:1], budget=24)
    with pytest.raises(ValueError, match="budget"):
        summarize(inputs)


def test_learning_curve_keeps_all_episodes_but_scores_only_last_fifty(tmp_path):
    rows = fixture(ticks=30)
    # Make the first six episodes different, retaining a consistent raw ledger.
    for row in rows:
        if row["event"] == "transition" and row["run_step"] <= 18:
            row["rewards"] = [-1.]*6
        elif row["event"] == "episode" and row["episode"] == 0:
            row["episode_return"] = -3.
        elif row["event"] in ("progress", "run_end"):
            tick = row["run_step"] // 6
            row["total_rewards"] = [tick - 2*min(tick, 3)]*6
            if tick < 3:
                row["partial_returns"] = [-tick]*6
    run = read_run(write(tmp_path, rows))
    assert len(run["episodes"]) == 60 and run["curve"][-1]["completed_episodes"] == 60
    assert run["episodes"][0]["episode_return"] == -3
    assert run["curve"][-1]["score"] == 3


def test_upstream_delayed_metrics_do_not_shift_first_update_accounting(tmp_path):
    rows = fixture(upstream=True)
    del rows[2]  # No metric returned by the first asynchronous train call.
    rows[-1]["first_training_action"] = 6
    run = read_run(write(tmp_path, rows))
    assert run["first_training_action"] == 6
