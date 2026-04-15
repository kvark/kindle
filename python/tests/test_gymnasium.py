"""End-to-end: train a kindle agent for a short budget on gymnasium CartPole.

Skipped if gymnasium isn't installed. Exercises both the user-driven
act/observe loop and the built-in `run` convenience wrapper. CartPole is
used (not LunarLander) because it has no Box2D dependency, keeping CI
light.
"""

import math

import pytest

gym = pytest.importorskip("gymnasium")

import kindle


def _cartpole_homeo(obs):
    # obs = [cart_x, cart_vx, pole_angle, pole_angular_vel]
    return [
        {"value": float(obs[2]), "target": 0.0, "tolerance": 0.05},  # pole upright
        {"value": float(obs[0]), "target": 0.0, "tolerance": 0.5},   # cart centered
    ]


def test_cartpole_act_observe_loop():
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    agent = kindle.Agent(obs_dim=len(obs), num_actions=2, env_id=1, seed=0)

    returns: list[float] = []
    episode_return = 0.0
    obs_list = [float(x) for x in obs]

    for _ in range(60):
        action = agent.act(obs_list)
        assert isinstance(action, int)
        assert 0 <= action < 2

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        next_list = [float(x) for x in next_obs]

        agent.observe(next_list, action, homeostatic=_cartpole_homeo(next_list))

        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            obs_list = [float(x) for x in obs]
            agent.mark_boundary()
        else:
            obs_list = next_list

    d = agent.diagnostics()
    assert agent.step_count() == 60
    for key in ("reward_surprise", "reward_novelty", "reward_homeo", "reward_order"):
        assert key in d
        assert math.isfinite(d[key])
    # At least one of the four reward components should be non-zero
    # within 60 steps — homeostatic fires from step 1 when the pole is
    # off-vertical.
    assert any(
        abs(d[key]) > 1e-6
        for key in ("reward_surprise", "reward_novelty", "reward_homeo", "reward_order")
    )


def test_cartpole_run_convenience():
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=2, seed=0)

    returns = agent.run(env, steps=60, homeo_fn=_cartpole_homeo)
    assert isinstance(returns, list)
    # CartPole episodes are short at random-policy; we should complete at
    # least one within 60 steps.
    assert len(returns) >= 1
    for r in returns:
        assert isinstance(r, float)
        assert math.isfinite(r)
    assert agent.step_count() == 60


def test_action_out_of_range_raises():
    agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=3, seed=0)
    with pytest.raises(ValueError):
        agent.observe([0.0, 0.0, 0.0, 0.0], action=5)


def test_homeo_accepts_tuples():
    agent = kindle.Agent(obs_dim=2, num_actions=2, env_id=4, seed=0)
    # Tuple form: (value, target, tolerance)
    agent.observe([0.1, 0.2], action=0, homeostatic=[(0.1, 0.0, 0.05)])
    d = agent.diagnostics()
    # Homeostatic penalty for value=0.1, target=0, tol=0.05 is
    # -max(0, 0.1 - 0.05) = -0.05, multiplied by the default weight 2.0.
    assert d["reward_homeo"] < 0.0
