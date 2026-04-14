"""Smoke test: the native extension loads and an Agent can be constructed."""

import kindle


def test_import_exposes_agent():
    assert hasattr(kindle, "Agent")
    assert kindle.OBS_TOKEN_DIM >= 4


def test_agent_constructs():
    agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=0, seed=0)
    assert agent.step_count() == 0
