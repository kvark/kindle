from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_object_features():
    path = Path(__file__).parents[1] / "examples" / "object_features.py"
    spec = importlib.util.spec_from_file_location("kindle_object_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_components_are_four_connected_and_color_separated():
    features = _load_object_features()
    grid = np.zeros((6, 7), dtype=np.int32)
    grid[0, 0] = 1
    grid[1, 1] = 1  # Diagonal contact is not 4-connected.
    grid[2:4, 3:6] = 2
    grid[2, 6] = 3  # Edge contact with a different color stays separate.

    objects = features._connected_components(grid)

    assert [(o["color"], o["area"], o["bbox"]) for o in objects] == [
        (1, 1, (0, 0, 0, 0)),
        (1, 1, (1, 1, 1, 1)),
        (2, 6, (2, 3, 3, 5)),
        (3, 1, (2, 6, 2, 6)),
    ]


def test_centroids_rank_by_area_then_position():
    features = _load_object_features()
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[5:7, 4:7] = 3
    grid[1:3, 1:3] = 2
    grid[1:3, 5:7] = 4

    assert features.object_centroids(grid, max_n=3) == [(5, 5), (1, 1), (1, 5)]


def test_modal_background_is_ignored_and_centroid_hits_concave_object():
    features = _load_object_features()
    grid = np.full((8, 8), 5, dtype=np.int32)
    grid[1:6, 1] = 3
    grid[5, 1:6] = 3

    objects = features._connected_components(grid)
    centroids = features.object_centroids(grid)

    assert len(objects) == 1
    assert objects[0]["color"] == 3
    assert centroids == [(4, 1)]
    assert grid[centroids[0]] == 3


def test_candidates_match_centroids_and_have_normalized_features():
    features = _load_object_features()
    grid = np.zeros((16, 16), dtype=np.int32)
    grid[2:5, 3:7] = 4
    grid[9:14, 10:12] = 7

    candidates = features.object_candidates(grid, max_n=8)

    assert [(y, x) for y, x, _row in candidates] == features.object_centroids(
        grid, max_n=8,
    )
    assert all(row.shape == (12,) for _y, _x, row in candidates)
    assert all(np.isfinite(row).all() for _y, _x, row in candidates)
    assert all(((0.0 <= row) & (row <= 1.0)).all() for _y, _x, row in candidates)


def test_project_to_object_centroid_snaps_to_nearest_ranked_object():
    features = _load_object_features()
    grid = np.zeros((64, 64), dtype=np.int32)
    grid[8:13, 48:53] = 2
    grid[32:37, 26:31] = 4

    assert features.project_to_object_centroid(grid, x=31, y=33) == (28, 34)
    assert features.project_to_object_centroid(grid, x=49, y=9) == (50, 10)


def test_project_to_object_centroid_clamps_when_frame_has_no_objects():
    features = _load_object_features()
    grid = np.zeros((12, 20), dtype=np.int32)

    assert features.project_to_object_centroid(grid, x=25.2, y=-4.0) == (19, 0)


def test_project_with_object_motion_extrapolates_only_a_stable_trajectory():
    features = _load_object_features()
    grid = np.zeros((64, 64), dtype=np.int32)
    for x, y in [(26, 37), (28, 35), (30, 33), (32, 31), (34, 29)]:
        grid[y, x] = 3

    stable = [(10, 53), (16, 47), (22, 41)]
    irregular = [(10, 53), (16, 47), (20, 43)]
    assert features.project_to_object_centroid(grid, x=31, y=33) == (30, 33)
    assert features.project_with_object_motion(
        grid, x=31, y=33, click_history=stable,
    ) == (28, 35)
    assert features.project_with_object_motion(
        grid, x=31, y=33, click_history=irregular,
    ) == (30, 33)

    repeated = [(46, 56), (46, 56), (46, 56)]
    grid[56, 24] = 3
    assert features.project_with_object_motion(
        grid, x=24, y=56, click_history=repeated,
    ) == (24, 56)


def test_nested_piece_ranks_ahead_of_its_larger_container():
    features = _load_object_features()
    grid = np.zeros((11, 11), dtype=np.int32)
    grid[1:8, 1:8] = 4
    grid[3:6, 3:6] = 2

    objects = features._rank_objects(features._connected_components(grid))

    assert [(o["color"], o["area"]) for o in objects] == [(2, 9), (4, 40)]
    assert features.object_centroids(grid, max_n=1) == [(4, 4)]


def test_hole_count_and_fixed_size_tokens():
    features = _load_object_features()
    grid = np.zeros((64, 64), dtype=np.int32)
    grid[10:15, 20] = 5
    grid[10:15, 24] = 5
    grid[10, 20:25] = 5
    grid[14, 20:25] = 5

    obj = features._connected_components(grid)[0]
    assert features._count_holes(obj, grid.shape) == 1

    object_token = features.object_token(grid)
    hybrid_token = features.hybrid_token(grid)
    assert object_token.shape == (64,)
    assert hybrid_token.shape == (64,)
    assert object_token.dtype == np.float32
    assert hybrid_token.dtype == np.float32
    assert np.isfinite(object_token).all()
    assert np.isfinite(hybrid_token).all()
    assert object_token[6] == 0.125
