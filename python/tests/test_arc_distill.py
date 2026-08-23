from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_distill():
    path = Path(__file__).parents[1] / "examples" / "arc_agi3_distill.py"
    spec = importlib.util.spec_from_file_location("kindle_arc_distill", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(game_id: str, observation: list[float], action_index: int) -> dict:
    return {
        "game_id": game_id,
        "level": 0,
        "observation": observation,
        "action_index": action_index,
        "available_action_indices": [0, 1],
        "action_parameters": None,
        "reward": 0,
    }


def _write_payload(path: Path, samples: list[dict], encoding="hybrid_token_v2"):
    path.write_text(json.dumps({
        "format": "kindle-arc-policy-demonstrations-v1",
        "observation_encoding": encoding,
        "samples": samples,
    }), encoding="utf-8")


def test_load_demonstrations_merges_files_and_deduplicates_rows(tmp_path):
    distill = _load_distill()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    duplicate = _sample("game-a", [0.0, 1.0], 0)
    _write_payload(first, [duplicate])
    _write_payload(second, [duplicate, _sample("game-b", [1.0, 0.0], 1)])

    encoding, samples = distill._load_demonstrations([first, second])

    assert encoding == "hybrid_token_v2"
    assert [(sample["game_id"], sample["action_index"]) for sample in samples] == [
        ("game-a", 0),
        ("game-b", 1),
    ]


def test_load_demonstrations_keeps_richer_candidate_set_for_duplicate(tmp_path):
    distill = _load_distill()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sparse = _sample("game-a", [0.0, 1.0], 1)
    sparse.update({
        "action_parameters": [0.25, -0.5],
        "object_index": 0,
        "object_candidate_features": [[0.1]],
    })
    rich = dict(sparse)
    rich["object_candidate_features"] = [[0.1], [0.2], [0.3]]
    _write_payload(first, [sparse])
    _write_payload(second, [rich])

    _encoding, samples = distill._load_demonstrations([first, second])

    assert samples == [rich]


def test_load_demonstrations_rejects_incompatible_encodings(tmp_path):
    distill = _load_distill()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_payload(first, [_sample("game-a", [0.0], 0)])
    _write_payload(second, [_sample("game-b", [1.0], 1)], "object_token_k8")

    with pytest.raises(ValueError, match="does not match"):
        distill._load_demonstrations([first, second])


def test_load_demonstrations_rejects_conflicting_model_visible_state(tmp_path):
    distill = _load_distill()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_payload(first, [_sample("game-a", [0.0, 1.0], 0)])
    _write_payload(second, [_sample("game-a", [0.0, 1.0], 1)])

    with pytest.raises(ValueError, match="conflicting labels"):
        distill._load_demonstrations([first, second])


def test_parse_level_requirements_supports_mixed_depth_gate():
    distill = _load_distill()

    assert distill._parse_level_requirements("ft09:2, vc33:3") == [
        ("ft09", 2),
        ("vc33", 3),
    ]
    assert distill._parse_level_requirements(None) == []
    with pytest.raises(ValueError, match="prefix:level"):
        distill._parse_level_requirements("ft09")
    with pytest.raises(ValueError, match="non-negative"):
        distill._parse_level_requirements("ft09:-1")
