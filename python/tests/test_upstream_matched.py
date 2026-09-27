import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


def test_matched_loop_counts_only_actions_and_keeps_terminal_then_reset(tmp_path, monkeypatch):
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
        def __init__(self, length, capacity, seed):
            assert (length, capacity, seed) == (65, 99616, 1009)
            self.records = [[] for _ in range(6)]

        def add(self, record, stream):
            rows = self.records[stream]
            if rows:
                assert bool(record["is_first"]) == bool(rows[-1]["is_last"])
            if record["is_last"]:
                assert record["action"] == 0 and not record["is_terminal"]
            rows.append(record)

        def __len__(self):
            return sum(max(0, len(rows)-64) for rows in self.records)

    class Agent:
        updates = 0

        def stream(self, stream): return stream
        def init_train(self, batch): return 0
        def init_policy(self, batch): return list(range(batch))

        def policy(self, carry, obs, mode):
            assert mode == "train"
            assert set(obs) == {"image", "reward", "is_first", "is_last", "is_terminal"}
            return carry, dict(action=np.ones(len(carry), dtype=np.int32)), {}

        def train(self, carry, data):
            self.updates += 1
            return carry, {}, dict(loss=1.0)

        def save(self):
            return dict(counters=dict(updates=self.updates), params=dict(weight=np.array([self.updates], dtype=float)))

    class Budget:
        minimum_headroom = 3 << 30
        def __init__(self, *args): pass
        def check(self, *args): pass
        def close(self): pass

    environments = [Env() for _ in range(6)]
    agent = Agent()
    created = []

    def replay(**kwargs):
        created.append(Replay(**kwargs))
        return created[-1]

    def stream(replay, mode):
        assert mode == "train"
        while True:
            assert len(replay) >= 1024
            yield {}

    monkeypatch.setitem(sys.modules, "embodied", SimpleNamespace(replay=SimpleNamespace(Replay=replay)))
    monkeypatch.setattr(matched, "GpuBudget", Budget)
    monkeypatch.setattr(matched, "make_environments", lambda *a: (environments,
                       [matched.observation((frame, {}), first=True) for _ in environments]))
    args = SimpleNamespace(logdir=tmp_path, batch_size=16, batch_length=64, replay_context=1,
                           consec_train=1, train_ratio=256, steps=1800, envs=6)
    matched.train(lambda: agent, None, None, stream, None, args, game="pong", seed=1009)
    result = json.loads((tmp_path / "comparison-result.json").read_text())
    assert result["run_step"] == 1800
    assert result["first_training_action"] == 1392
    assert result["learner_updates"] == 103  # One at 1392, then 408 / 4.
    assert result["emulator_resets"] == [4]*6
    assert result["executed_action_frames"] == [1200]*6
    assert result["completed_episodes"] == 18
    assert all(e.closed for e in environments)
    for records in created[0].records:
        assert len(records) == 304 and sum(not r["is_first"] for r in records) == 300
