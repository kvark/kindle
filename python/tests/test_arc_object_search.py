from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_search():
    path = Path(__file__).parents[1] / "examples" / "arc_agi3_object_search.py"
    spec = importlib.util.spec_from_file_location("kindle_arc_object_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_visual_state_key_crops_volatile_border():
    search = _load_search()
    first = np.zeros((6, 6), dtype=np.uint8)
    second = first.copy()
    second[0, :] = 9
    second[-1, :] = 7

    assert search.visual_state_key(first, 1) == search.visual_state_key(second, 1)
    assert search.visual_state_key(first, 0) != search.visual_state_key(second, 0)


def test_visual_state_key_requires_grid():
    search = _load_search()
    with pytest.raises(ValueError, match="expected a 2D frame"):
        search.visual_state_key(np.zeros((2, 2, 2), dtype=np.uint8))


def test_search_finds_shortest_rewarding_toggle_combination():
    search = _load_search()
    target = 0b101

    def transition(state, action):
        child = state ^ (1 << action)
        return child, child

    result = search.breadth_first_reward_search(
        0,
        0,
        lambda _state, _obs: [0, 1, 2],
        transition,
        lambda obs: obs,
        lambda obs: int(obs == target),
        lambda obs: False,
        max_states=8,
        ordered_actions=True,
    )

    assert result.found
    assert result.state == target
    assert result.path == (0, 2)


def test_search_respects_state_budget():
    search = _load_search()

    result = search.breadth_first_reward_search(
        0,
        0,
        lambda _state, _obs: [0, 1, 2],
        lambda state, action: (state ^ (1 << action), state ^ (1 << action)),
        lambda obs: obs,
        lambda obs: 0,
        lambda obs: False,
        max_states=1,
    )

    assert not result.found
    assert result.expanded == 1


def test_search_respects_transition_budget():
    search = _load_search()

    result = search.breadth_first_reward_search(
        0,
        0,
        lambda _state, _obs: [1, 2, 3, 4],
        lambda state, action: (state + action, state + action),
        lambda obs: obs,
        lambda obs: 0,
        lambda obs: False,
        max_states=100,
        max_transitions=3,
    )

    assert not result.found
    assert result.expanded == 1
    assert result.generated == 3


def test_search_stops_at_terminal_states():
    search = _load_search()

    result = search.breadth_first_reward_search(
        0,
        0,
        lambda _state, _obs: [1],
        lambda state, action: (state + action, state + action),
        lambda obs: obs,
        lambda obs: 0,
        lambda obs: obs >= 1,
        max_states=10,
    )

    assert not result.found
    assert result.expanded == 1
    assert result.generated == 1


def test_search_regenerates_actions_for_each_state():
    search = _load_search()

    def actions_for(state, _observation):
        return ["first"] if state == 0 else ["second"]

    def transition(state, action):
        if (state, action) == (0, "first"):
            return 1, 1
        if (state, action) == (1, "second"):
            return 2, 2
        return state, state

    result = search.breadth_first_reward_search(
        0,
        0,
        actions_for,
        transition,
        lambda obs: obs,
        lambda obs: int(obs == 2),
        lambda obs: False,
        max_states=10,
    )

    assert result.found
    assert result.path == ("first", "second")


def test_search_skips_simulator_rejected_branches():
    search = _load_search()

    def transition(state, action):
        if action == "invalid":
            raise search.TransitionRejected("bad cloned branch")
        return state + 1, state + 1

    result = search.breadth_first_reward_search(
        0,
        0,
        lambda _state, _obs: ["invalid", "valid"],
        transition,
        lambda obs: obs,
        lambda obs: obs,
        lambda obs: False,
        max_states=10,
    )

    assert result.found
    assert result.path == ("valid",)
    assert result.generated == 2


def test_commuting_involution_detection():
    search = _load_search()

    def toggle(state, action):
        child = state ^ (1 << action)
        return child, child

    assert search.commuting_involutions(
        0, 0, [0, 1, 2], toggle, lambda obs: obs,
    )

    def increment(state, action):
        child = state + action + 1
        return child, child

    assert not search.commuting_involutions(
        0, 0, [0, 1], increment, lambda obs: obs,
    )
