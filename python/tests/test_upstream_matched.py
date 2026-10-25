import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import upstream_matched as matched


def test_update_schedule_discards_prefill_and_reset_credit():
    schedule = matched.UpdateSchedule()
    for _ in range(100):
        assert schedule.observe(6, False) == 0
    assert schedule.observe(6, True) == 1
    assert schedule.observe(0, True) == 0  # A reset is not an actual action.
    assert schedule.observe(6, True) == 1
    assert schedule.observe(6, True) == 2
    assert schedule.credit == 0


def test_cutoff_bootstraps_and_true_terminal_does_not():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    timeout = matched.observation((frame, 2, False, True, {}))
    assert timeout["is_last"] and not timeout["is_terminal"]
    terminal = matched.observation((frame, 2, True, False, {}))
    assert terminal["is_last"] and terminal["is_terminal"]
    reset = matched.observation((frame, {}), first=True)
    assert reset["is_first"] and not reset["is_last"] and reset["reward"] == 0


@pytest.mark.parametrize("batch,length,ratio,streams,first,updates", [
    (16, 64, 256, 6, 1392, 103),
    (8, 16, 32, 8, 248, 539),
])
def test_matched_loop_counts_only_actions_and_keeps_terminal_then_reset(
        tmp_path, monkeypatch, batch, length, ratio, streams, first, updates):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    class Env:
        executed_action_frames = 0
        reset_noop_frames = 0
        emulator_resets = 1
        episode_steps = 0
        closed = False

        def step(self, action):
            assert action == 1
            self.executed_action_frames += 4
            self.episode_steps += 1
            return frame, 1, False, self.episode_steps == 100, {}

        def reset(self):
            self.episode_steps = 0
            self.emulator_resets += 1
            return frame, {}

        def close(self):
            self.closed = True

    class Replay:
        def __init__(self, **kwargs):
            assert kwargs == dict(length=length+1, capacity=100000-streams*length, seed=1009)
            self.records = [[] for _ in range(streams)]

        def add(self, record, stream):
            rows = self.records[stream]
            if rows:
                assert bool(record["is_first"]) == bool(rows[-1]["is_last"])
            if record["is_last"]:
                assert record["action"] == 0 and not record["is_terminal"]
            rows.append(record)

        def __len__(self):
            return sum(max(0, len(rows)-length) for rows in self.records)

    class Agent:
        updates = 0
        jaxcfg = SimpleNamespace(profiler=True)

        def stream(self, stream): return stream
        def init_train(self, batch): return 0
        def init_policy(self, batch):
            # Exact container layout of the pinned JAX wrapper: model tuple,
            # dictionaries and per-stream lists, not a list of model tuples.
            return ((), dict(deter=[np.array([i, 0]) for i in range(batch)]),
                    dict(action=[np.array(0) for _ in range(batch)]))

        def policy(self, carry, obs, mode):
            assert mode == "train"
            assert set(obs) == {"image", "reward", "is_first", "is_last", "is_terminal"}
            empty, state, previous = carry
            assert empty == ()
            size = len(obs["reward"])
            assert len(state["deter"]) == len(previous["action"]) == size
            for i in range(size):
                stream, steps = state["deter"][i]
                state["deter"][i] = np.array([stream, 0 if obs["is_first"][i] else steps+1])
            return carry, dict(action=np.ones(size, dtype=np.int32)), {}

        def train(self, carry, data):
            self.updates += 1
            return carry, {}, dict(loss=1.0)

        def save(self):
            return dict(counters=dict(updates=self.updates), params={
                name: np.array([self.updates], dtype=float) for name in ("dyn/weight", "enc/weight", "pol/weight")})

    class Budget:
        minimum_headroom = 3 << 30
        def __init__(self, *args): pass
        def check(self, *args): pass
        def close(self): pass

    environments = [Env() for _ in range(streams)]
    agent = Agent()
    created = []

    def replay(**kwargs):
        created.append(Replay(**kwargs))
        return created[-1]

    def stream(replay, mode):
        assert mode == "train"
        while True:
            assert len(replay) >= batch * length
            yield {}

    monkeypatch.setitem(sys.modules, "embodied", SimpleNamespace(replay=SimpleNamespace(Replay=replay)))
    monkeypatch.setattr(matched, "GpuBudget", Budget)
    monkeypatch.setattr(matched, "make_environments", lambda *a: (environments,
                       [matched.observation((frame, {}), first=True) for _ in environments]))
    args = SimpleNamespace(logdir=tmp_path, batch_size=batch, batch_length=length, replay_context=1,
                           consec_train=1, train_ratio=ratio, steps=float(streams*300), envs=streams)
    matched.train(lambda: agent, None, None, stream, None, args, game="pong", seed=1009)
    result = json.loads((tmp_path / "comparison-result.json").read_text())
    assert result["run_step"] == streams*300
    assert result["first_training_action"] == first
    assert result["learner_updates"] == updates
    assert result["emulator_resets"] == [4]*streams
    assert result["executed_action_frames"] == [1200]*streams
    assert result["completed_episodes"] == 3*streams
    start = json.loads((tmp_path / "comparison.jsonl").read_text().splitlines()[0])
    assert (start["batch_size"], start["batch_length"], start["train_ratio"]) == (batch, length, ratio)
    assert all(e.closed for e in environments)
    for records in created[0].records:
        assert len(records) == 304 and sum(not r["is_first"] for r in records) == 300


@pytest.mark.parametrize("overrides", [dict(replay_context=0), dict(consec_train=2),
                                     dict(batch_length=0), dict(envs=0), dict(batch_size=100000)])
def test_invalid_recipe_is_rejected_before_gpu_initialization(monkeypatch, overrides):
    monkeypatch.setitem(sys.modules, "embodied", SimpleNamespace())
    monkeypatch.setattr(matched, "GpuBudget", lambda *a: pytest.fail("created GPU budget"))
    values = dict(batch_size=8, batch_length=16, replay_context=1, consec_train=1,
                  train_ratio=32, steps=2400, envs=8)
    values.update(overrides)
    with pytest.raises(ValueError):
        matched.train(None, None, None, None, None, SimpleNamespace(**values), game="pong", seed=1009)


def test_policy_carry_subsets_preserve_opaque_arrays_and_unselected_streams():
    values = [object() for _ in range(6)]
    carry = ((), dict(deter=values.copy(), stoch=values.copy()), dict(action=values.copy()))
    subset = matched.map_carry(lambda rows: [rows[i] for i in [5, 2]], carry)
    assert subset[0] == () and subset[1]["deter"] == [values[5], values[2]]
    assert subset[1]["deter"] is not carry[1]["deter"]
    replacements = [object(), object()]
    updated = matched.map_carry(lambda _: replacements.copy(), subset)

    def commit(rows, new):
        for stream, value in zip([5, 2], new):
            rows[stream] = value
        return rows

    matched.map_carry(commit, carry, updated)
    expected = [values[0], values[1], replacements[1], values[3], values[4], replacements[0]]
    assert carry[1]["deter"] == carry[1]["stoch"] == carry[2]["action"] == expected
    with pytest.raises(ValueError):
        matched.map_carry(lambda x: x, np.zeros((6, 2)))
    with pytest.raises(ValueError):
        matched.map_carry(lambda *x: x, carry, ((), {}, {}))
