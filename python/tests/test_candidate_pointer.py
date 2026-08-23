from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_pointer():
    path = Path(__file__).parents[1] / "examples" / "candidate_pointer.py"
    spec = importlib.util.spec_from_file_location("kindle_candidate_pointer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_pointer_fits_variable_sets_and_round_trips(tmp_path):
    module = _load_pointer()
    samples = [
        {
            "game_id": "left",
            "observation": [-1.0, 0.0],
            "object_candidate_features": [[0.0], [0.5], [1.0]],
            "object_index": 0,
        },
        {
            "game_id": "left",
            "observation": [-0.5, 1.0],
            "object_candidate_features": [[0.2], [0.8]],
            "object_index": 0,
        },
        {
            "game_id": "right",
            "observation": [1.0, 0.0],
            "object_candidate_features": [[0.0], [0.5], [1.0]],
            "object_index": 2,
        },
        {
            "game_id": "right",
            "observation": [0.5, 1.0],
            "object_candidate_features": [[0.2], [0.8]],
            "object_index": 1,
        },
    ]
    pointer = module.CandidatePointer(
        ["left", "right"],
        context_dim=2,
        candidate_dim=1,
        hidden_dim=16,
        hidden_dim_2=8,
        seed=7,
    )

    result = pointer.fit(samples, epochs=2_000, learning_rate=2e-3)

    assert result["accuracy"] == 1.0
    assert pointer.accuracy(samples) == 1.0
    checkpoint = tmp_path / "pointer.npz"
    pointer.save(checkpoint)
    loaded = module.CandidatePointer.load(checkpoint)
    assert loaded.accuracy(samples) == 1.0
