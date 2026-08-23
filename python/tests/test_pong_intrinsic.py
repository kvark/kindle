from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_intrinsic():
    path = Path(__file__).parents[1] / "examples" / "atari_pong_intrinsic.py"
    spec = importlib.util.spec_from_file_location("kindle_pong_intrinsic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_seeds_requires_unique_integers():
    intrinsic = _load_intrinsic()

    assert intrinsic._parse_seeds("42,7, 123") == [42, 7, 123]
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        intrinsic._parse_seeds("42,bad")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        intrinsic._parse_seeds("7,7")


def test_tracking_actions_follow_presence_and_intercept_error():
    intrinsic = _load_intrinsic()
    tokens = np.zeros((4, 64), dtype=np.float32)
    tokens[1:, 0] = 1.0
    tokens[2, 11] = -0.2
    tokens[3, 11] = 0.2

    desired = intrinsic._desired_tracking_actions(tokens)

    np.testing.assert_array_equal(desired, [1, 0, 2, 3])


def test_tracking_reward_balances_classes_and_signs():
    intrinsic = _load_intrinsic()
    desired = np.asarray([0, 0, 0, 0, 1, 1, 2, 3], dtype=np.int64)
    actions = np.asarray([0, 1, 0, 1, 1, 0, 2, 0], dtype=np.int64)

    rewards = intrinsic._balanced_tracking_rewards(desired, actions)

    assert rewards.dtype == np.float32
    np.testing.assert_array_equal(np.sign(rewards), [1, -1, 1, -1, 1, -1, 1, -1])
    for target in range(4):
        # Every represented target class receives the same total absolute
        # reward, preventing FIRE/no-ball states from dominating updates.
        assert float(np.abs(rewards[desired == target]).sum()) == pytest.approx(2.0)


def test_gate_requires_each_seed_to_clear_behavior_and_score_thresholds():
    intrinsic = _load_intrinsic()
    args = argparse.Namespace(
        require_best_tracking=0.99,
        require_active_tracking=0.98,
        require_positive_fraction=0.30,
        require_episodes_per_seed=3,
        require_return=-15.0,
    )
    training = {"best_tracking_accuracy": 1.0}
    result = {
        "active_tracking_accuracy": 0.99,
        "positive_fraction": 0.32,
        "episodes": 4,
        "mean_return": -13.0,
        "action_counts": [1, 2, 3, 4, 0, 0],
    }

    checks = intrinsic._gate(args, training, [result, dict(result)])

    assert all(passed for passed, _description in checks)
    result["positive_fraction"] = 0.29
    checks = intrinsic._gate(args, training, [result])
    assert not checks[2][0]
