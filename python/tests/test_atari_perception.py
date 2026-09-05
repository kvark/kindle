import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import probe_atari_perception as probe  # noqa: E402


@pytest.mark.parametrize("boundary", ["terminal", "missing_label"])
def test_motion_pairs_do_not_cross_boundaries(monkeypatch, boundary) -> None:
    class Environment:
        action_space = SimpleNamespace(n=4)
        index = 0
        label = None
        closed = False

        def reset(self, **kwargs):
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

        def step(self, action):
            value = [1, 2, 50, 100, 101][self.index]
            self.label = np.full(4, value, dtype=np.float32)
            interrupted = self.index == 2
            if interrupted and boundary == "missing_label":
                self.label = None
            self.index += 1
            return (np.full((1, 1, 3), value, dtype=np.uint8), 0.0,
                    interrupted and boundary == "terminal", False, {})

        def close(self):
            self.closed = True

    class Encoder:
        def encode(self, frame):
            value = float(frame[0, 0, 0])
            return [value, value], [value]

    class Agent:
        config = {"action_count": 4}
        observed = 0
        resets = 0

        def begin_episode(self, frame):
            self.resets += 1

        def act(self, *, action_mask):
            return action_mask.index(True)

        def observe(self, frame, **kwargs):
            self.observed += 1

        @property
        def latent_feature(self):
            return [self.observed]

    environment = Environment()
    monkeypatch.setattr(probe.gym, "make", lambda *args, **kwargs: environment)
    monkeypatch.setattr(probe, "DreamerAtariPreprocessing", lambda env, **kwargs: env)
    monkeypatch.setattr(probe, "pong_labels", lambda env: env.label)
    agent = Agent()
    features, labels = probe.collect_split(Encoder(), "ALE/Pong-v5", 0, 2,
                                          motion=True, agent=agent)
    np.testing.assert_array_equal(features["pooled7_history"], [[2, 1], [101, 1]])
    np.testing.assert_array_equal(labels, np.ones((2, 4)))
    np.testing.assert_array_equal(features["rssm"], [[2], [5]])
    assert agent.observed == 5
    assert agent.resets == (2 if boundary == "terminal" else 1)
    assert environment.closed
