from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_example():
    path = Path(__file__).parents[1] / "examples" / "arc_agi3_batch.py"
    spec = importlib.util.spec_from_file_location("kindle_arc_agi3_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arc_action_indices_have_fixed_game_action_meaning():
    arc = _load_arc_example()

    assert arc._valid_action_indices([1, 2, 6]) == [0, 1, 5]
    assert arc._valid_action_indices([6, 3, 6, 0, 8]) == [2, 5]


def test_arc_action_mask_gates_values_not_list_positions():
    arc = _load_arc_example()
    mask = np.ones(arc.MAX_ACTION_DIM, dtype=np.float32)

    valid = arc._fill_action_mask(mask, [2, 6])

    assert valid == [1, 5]
    np.testing.assert_array_equal(np.flatnonzero(mask), [1, 5])
