"""End-to-end: train a kindle agent for a short budget on gymnasium CartPole.

Skipped if gymnasium isn't installed.
"""

import pytest

gym = pytest.importorskip("gymnasium")

import kindle


def test_cartpole_short_training():
    env = gym.make("CartPole-v1")
    obs, _info = env.reset(seed=0)

    # gymnasium returns ndarray; wrap so the Rust side can iterate floats.
    class ListEnv:
        def __init__(self, inner):
            self.inner = inner

        def reset(self):
            obs, _ = self.inner.reset()
            return list(float(x) for x in obs)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.inner.step(int(action) % 2)
            if terminated or truncated:
                obs, _ = self.inner.reset()
            return (list(float(x) for x in obs), float(reward), bool(terminated), bool(truncated), info)

    agent = kindle.Agent(obs_dim=len(list(obs)), num_actions=2, env_id=42, seed=0)
    wrapped = ListEnv(env)
    agent.train(wrapped, steps=50)

    diag = agent.diagnostics()
    assert agent.step_count() == 50
    # All four reward components should be finite.
    for key in ("reward_surprise", "reward_novelty", "reward_homeo", "reward_order"):
        assert key in diag
        val = diag[key]
        assert val == val  # not NaN
        assert val not in (float("inf"), float("-inf"))
