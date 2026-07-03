"""Error-path coverage for the pyo3 bindings.

Every case here used to cross the FFI boundary as a Rust panic
(pyo3 PanicException, a BaseException) or silently mis-train; the
bindings now validate up front and raise ValueError (audit M-14,
M-15, M-19). Kept GPU-light: constructors plus a step or two.
"""

import pytest

import kindle


def test_num_actions_over_max_raises_value_error():
    # MAX_ACTION_DIM is 18 in kindle::adapter.
    with pytest.raises(ValueError, match="num_actions"):
        kindle.Agent(obs_dim=4, num_actions=100)


def test_batch_num_actions_over_max_raises_value_error():
    with pytest.raises(ValueError, match="num_actions"):
        kindle.BatchAgent(obs_dim=4, num_actions=19, batch_size=1)


def test_obs_dim_over_token_dim_raises_value_error():
    with pytest.raises(ValueError, match="obs_dim"):
        kindle.Agent(obs_dim=kindle.OBS_TOKEN_DIM + 1, num_actions=2)


def test_agent_wrong_obs_length_raises():
    agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=0, seed=0)
    # Too short (would silently zero-pad before the fix).
    with pytest.raises(ValueError, match="obs_dim"):
        agent.act([0.0, 0.0])
    # Too long (would silently truncate).
    with pytest.raises(ValueError, match="obs_dim"):
        agent.act([0.0] * 5)
    # Correct length works, including in observe.
    action = agent.act([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="obs_dim"):
        agent.observe([0.0, 0.0, 0.0], action)
    agent.observe([0.0, 0.1, 0.0, 0.1], action)
    assert agent.step_count() == 1


def test_batch_agent_wrong_obs_length_raises():
    agent = kindle.BatchAgent(obs_dim=4, num_actions=2, batch_size=2, seed=0)
    with pytest.raises(ValueError, match="obs_dim"):
        agent.act([[0.0] * 4, [0.0] * 3])
    actions = agent.act([[0.0] * 4, [0.0] * 4])
    with pytest.raises(ValueError, match="obs_dim"):
        agent.observe([[0.0] * 4, [0.0] * 5], actions)


def test_batch_agent_wrong_homeo_length_raises():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=2, seed=0)
    obs = [[0.0, 0.0], [0.0, 0.0]]
    actions = agent.act(obs)
    homeo = [(0.1, 0.0, 0.05)]
    # Too short: used to be silently padded with empty lists (M-19).
    with pytest.raises(ValueError, match="batch_size"):
        agent.observe(obs, actions, homeostatic=[[homeo[0]]])
    # Too long.
    with pytest.raises(ValueError, match="batch_size"):
        agent.observe(obs, actions, homeostatic=[[], [], []])
    # Exact length works.
    agent.observe(obs, actions, homeostatic=[[homeo[0]], []])
    assert agent.step_count() == 1


def test_batch_agent_setter_length_checks():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=2, seed=0)
    # set_extrinsic_reward expects batch_size entries — used to panic
    # (or silently no-op under the alpha gate) on mismatch (M-14).
    with pytest.raises(ValueError, match="set_extrinsic_reward"):
        agent.set_extrinsic_reward([1.0])
    with pytest.raises(ValueError, match="set_extrinsic_reward"):
        agent.set_extrinsic_reward([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="set_intrinsic_progress"):
        agent.set_intrinsic_progress([1.0])
    # set_action_masks expects batch_size * MAX_ACTION_DIM (= 18) floats.
    with pytest.raises(ValueError, match="set_action_masks"):
        agent.set_action_masks([1.0] * 5)
    # Correct lengths are accepted.
    agent.set_extrinsic_reward([0.0, 0.0])
    agent.set_intrinsic_progress([0.0, 0.0])
    agent.set_action_masks([1.0] * (2 * 18))
