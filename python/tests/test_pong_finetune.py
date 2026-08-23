from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_finetune():
    path = Path(__file__).parents[1] / "examples" / "atari_pong_finetune.py"
    spec = importlib.util.spec_from_file_location("kindle_pong_finetune", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_seeds_requires_unique_integers():
    finetune = _load_finetune()

    assert finetune._parse_seeds("42, 7,123") == [42, 7, 123]
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        finetune._parse_seeds("42,nope")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        finetune._parse_seeds("42,42")


def test_parameter_change_separates_encoder_and_policy_head():
    finetune = _load_finetune()
    before = {
        "policy_encoder.fc": [1.0, 2.0],
        "policy.fc1": [3.0, 4.0],
        "value.fc": [5.0],
    }
    after = {
        "policy_encoder.fc": [1.0, 2.0],
        "policy.fc1": [4.0, 2.0],
        "value.fc": [9.0],
    }

    changes = finetune._parameter_change(before, after)

    assert changes["policy_encoder."] == {
        "l1": 0.0,
        "max": 0.0,
        "parameters": 2,
    }
    assert changes["policy."] == {
        "l1": 3.0,
        "max": 2.0,
        "parameters": 2,
    }


def test_gate_requires_every_seed_and_frozen_encoder():
    finetune = _load_finetune()
    args = argparse.Namespace(
        require_deficit_improvement=0.2,
        require_event_retention=0.75,
        require_episodes_per_seed=1,
        require_return=-15.0,
    )
    source = [
        {
            "seed": 1,
            "decisions": 10,
            "episodes": 2,
            "completed_return_sum": -28.0,
            "mean_return": -14.0,
            "positive_events": 2,
            "negative_events": 12,
            "point_deficit": 10,
            "positive_point_fraction": 2 / 14,
        },
        {
            "seed": 2,
            "decisions": 10,
            "episodes": 2,
            "completed_return_sum": -28.0,
            "mean_return": -14.0,
            "positive_events": 2,
            "negative_events": 12,
            "point_deficit": 10,
            "positive_point_fraction": 2 / 14,
        },
    ]
    fine = [
        {
            **source[0],
            "point_deficit": 7,
            "positive_events": 3,
            "negative_events": 10,
            "positive_point_fraction": 3 / 13,
        },
        {
            **source[1],
            "point_deficit": 7,
            "positive_events": 3,
            "negative_events": 10,
            "positive_point_fraction": 3 / 13,
        },
    ]
    changes = {
        "policy_encoder.": {"l1": 0.0, "max": 0.0, "parameters": 2},
        "policy.": {"l1": 1.0, "max": 1.0, "parameters": 2},
    }

    checks, improvement = finetune._gate(args, source, fine, changes)

    assert improvement == pytest.approx(0.3)
    assert all(passed for passed, _description in checks)
    changes["policy_encoder."]["max"] = 0.01
    checks, _improvement = finetune._gate(args, source, fine, changes)
    assert not checks[-2][0]
