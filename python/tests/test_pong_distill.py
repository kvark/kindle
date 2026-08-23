from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_distill():
    path = Path(__file__).parents[1] / "examples" / "atari_pong_distill.py"
    spec = importlib.util.spec_from_file_location("kindle_pong_distill", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_split_is_reproducible_and_action_balanced():
    distill = _load_distill()
    rows = {
        action: {
            bytes([index]): np.asarray([action, index], dtype=np.float32)
            for index in range(8)
        }
        for action in range(4)
    }

    first = distill._balanced_split(rows, 6, 4, seed=7)
    second = distill._balanced_split(rows, 6, 4, seed=7)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    train_x, train_y, validation_x, validation_y = first
    assert train_x.shape == (16, 2)
    assert validation_x.shape == (8, 2)
    assert np.bincount(train_y).tolist() == [4, 4, 4, 4]
    assert np.bincount(validation_y).tolist() == [2, 2, 2, 2]


def test_balanced_split_rejects_missing_action_coverage():
    distill = _load_distill()
    rows = {action: {} for action in range(4)}
    rows[0][b"only"] = np.zeros(64, dtype=np.float32)

    with pytest.raises(ValueError, match="action 0 has 1 unique states"):
        distill._balanced_split(rows, 2, 1, seed=0)
