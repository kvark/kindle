import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import probe_fixed_latents as probe


class Environment:
    action_space = SimpleNamespace(n=3)

    def __init__(self):
        self.tick = 0
        self.unwrapped = SimpleNamespace(ale=SimpleNamespace(getRAM=self.ram))

    def ram(self):
        value = np.zeros(128, dtype=np.uint8)
        value[70] = self.tick
        return value

    def reset(self, *, seed=None):
        if seed is not None:
            self.tick = 0
        return self.tick, {}

    def step(self, action):
        assert 0 <= action < 3
        self.tick += 1
        return self.tick, float(self.tick == 2), self.tick in (3, 7), self.tick == 5, {}


class Agent:
    learner_step = 1987
    environment_step = 8192

    def begin_episode(self, frame):
        self.visual_observation = [frame, -frame]

    def act(self, *, action_mask):
        return action_mask.index(True)

    def observe(self, frame, **kwargs):
        self.environment_step += 1
        self.begin_episode(frame)


def test_collection_keeps_terminal_targets_and_separate_reset_arrivals():
    actor, checks = Agent(), []
    data = probe.collect_trace(actor, Environment(), 123, 8, lambda: checks.append(True))
    assert actor.learner_step == 1987 and actor.environment_step == 8200
    np.testing.assert_array_equal(data["episodes"], [0, 0, 0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(data["features"][data["following"], 0], np.arange(1, 9))
    np.testing.assert_array_equal(data["positions"][data["following"], 0], np.arange(1, 9))
    np.testing.assert_array_equal(data["phase"][data["current"]], [0, 1, 2, 0, 1, 0, 1, 0])
    assert len(data["features"]) == 12 and len(checks) == 2
    rng = random.Random(123 ^ probe.ACTION_SEED_XOR)
    assert data["actions"].tolist() == [rng.randrange(3) for _ in range(8)]
    for i in range(7):
        assert (data["following"][i] == data["current"][i + 1]) == (data["episodes"][i] == data["episodes"][i + 1])


def test_chunk_wrap_does_not_reset_episode_or_drop_actions():
    actor = Agent()
    data = probe.collect_trace(actor, Environment(), 123, 26, lambda: None)
    assert data["episodes"][-1] == 3
    np.testing.assert_array_equal(data["phase"][data["following"]][7:], np.arange(1, 20) % 16)
    assert len(data["actions"]) == 26


@pytest.mark.parametrize("failure", ["action", "updates", "nonfinite"])
def test_collection_rejects_invalid_actor(failure):
    actor = Agent()
    if failure == "action":
        actor.act = lambda **_: -1
    else:
        original = actor.observe

        def observe(*args, **kwargs):
            original(*args, **kwargs)
            if failure == "updates":
                actor.learner_step += 1
            else:
                actor.visual_observation = [float("nan"), 0]

        actor.observe = observe
    with pytest.raises(RuntimeError):
        probe.collect_trace(actor, Environment(), 123, 8, lambda: None)


def test_trajectory_split_is_disjoint_and_test_has_three_seeds():
    seeds = [seed for group in probe.SPLITS.values() for seed in group]
    assert len(seeds) == len(set(seeds)) and len(probe.SPLITS["test"]) == 3
