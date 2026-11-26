"""Demonstration episodes: the teacher plays each lane's first episodes, then never acts again."""
import numpy as np
import pytest

from kindle._breakout_guidance import BreakoutDemonstrations, DEMONSTRATION_KIND, PixelPolicy


def config(episodes=2):
    return dict(kind=DEMONSTRATION_KIND, episodes=episodes, seed=1009)


@pytest.mark.parametrize('value', [None, 0, -1, True, 1.5, 2**32])
def test_invalid_episode_count(value):
    with pytest.raises(ValueError, match='positive u32'):
        BreakoutDemonstrations(config(value), 2, 18)


@pytest.mark.parametrize('change', [dict(probability=1.0), dict(max_actions=8), dict(kind='other')])
def test_unrecognized_fields_rejected(change):
    with pytest.raises(ValueError, match='configuration fields'):
        BreakoutDemonstrations(dict(config(), **change), 2, 18)


def test_first_episodes_are_demonstrated_in_full_and_later_ones_never():
    demo = BreakoutDemonstrations(config(2), 3, 18)
    assert demo.selected_episodes == [True] * 3 and not demo.finished
    assert all(isinstance(policy, PixelPolicy) and policy.edge_reset for policy in demo._policies)
    frames = [np.zeros((210, 160, 3), np.uint8)] * 3
    for _ in range(5000):  # no action cap inside a demonstrated episode
        assert all(action in (0, 1, 3, 4) for action in demo.actions(frames))
    demo.reset([0])
    assert demo.selected_episodes == [True, True, True]
    demo.reset([0])
    assert demo.selected_episodes == [False, True, True]
    assert demo.actions(frames)[0] is None
    for _ in range(4):
        demo.reset([0, 1, 2])
    assert demo.finished and demo.actions(frames) == [None] * 3
    assert demo.episode_counts == dict(started_episodes=[7, 5, 5], guided_episodes=[2, 2, 2])
    assert demo.overridden_actions == [5000, 5001, 5001]
