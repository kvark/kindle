from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_atari_example():
    path = Path(__file__).parents[1] / "examples" / "atari_batch.py"
    spec = importlib.util.spec_from_file_location("kindle_atari_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_obs_tokens_preserve_frame_history_and_range():
    atari = _load_atari_example()
    frames = np.empty((2, 4, 84, 84), dtype=np.uint8)
    frames[0, 0].fill(0)
    frames[0, 1].fill(85)
    frames[0, 2].fill(170)
    frames[0, 3].fill(255)
    frames[1].fill(127)

    tokens = np.asarray(atari._visual_obs_tokens(frames), dtype=np.float32)

    assert tokens.shape == (2, 64)
    np.testing.assert_allclose(tokens[0, 0:16], 0.0)
    np.testing.assert_allclose(tokens[0, 16:32], 85.0 / 255.0)
    np.testing.assert_allclose(tokens[0, 32:48], 170.0 / 255.0)
    np.testing.assert_allclose(tokens[0, 48:64], 1.0)
    assert np.all((tokens >= 0.0) & (tokens <= 1.0))


def test_foreground_residual_removes_flat_background_from_frames_and_tokens():
    atari = _load_atari_example()
    frames = np.full((1, 4, 84, 84), 87, dtype=np.uint8)
    frames[:, :, 21:42, 21:42] = 187
    destination = np.empty_like(frames, dtype=np.float32)

    atari._write_frames_into_shared(frames, destination, foreground_residual=True)
    tokens = np.asarray(
        atari._visual_obs_tokens(frames, foreground_residual=True),
        dtype=np.float32,
    )

    np.testing.assert_allclose(destination[:, :, :21, :21], 0.0)
    np.testing.assert_allclose(destination[:, :, 21:42, 21:42], 100.0 / 255.0)
    assert tokens.shape == (1, 64)
    assert float(tokens.max()) > 0.0
    assert float(tokens.min()) == 0.0


def test_pong_heuristic_tracks_ball_and_serves_when_motion_is_absent():
    atari = _load_atari_example()
    frames = np.full((3, 4, 84, 84), 87, dtype=np.uint8)
    # Right paddle centered at y=40 in every lane.
    frames[:, -1, 37:44, 73:76] = 236
    frames[:, -2, 37:44, 73:76] = 236
    # Lane 0 ball moves above the paddle; lane 1 below it; lane 2 has no ball.
    frames[0, -2, 29:31, 64:66] = 236
    frames[0, -1, 27:29, 66:68] = 236
    frames[1, -2, 49:51, 64:66] = 236
    frames[1, -1, 51:53, 66:68] = 236

    tokens, token_paddle_y = atari._pong_motion_tokens(frames)
    actions, paddle_y = atari._pong_heuristic_actions(frames)

    assert actions == [2, 3, 1]
    np.testing.assert_allclose(paddle_y, [40.0, 40.0, 40.0])
    np.testing.assert_allclose(token_paddle_y, paddle_y)
    assert tokens.shape == (3, 64)
    assert tokens.dtype == np.float32
    assert tokens[0, 0] == 1.0 and tokens[0, 11] < 0.0
    assert tokens[1, 0] == 1.0 and tokens[1, 11] > 0.0
    assert tokens[2, 0] == 0.0 and tokens[2, 15] == 1.0
    np.testing.assert_allclose(tokens[:, 17:], 0.0)


def test_transition_observations_restore_only_completed_lanes():
    atari = _load_atari_example()
    reset_obs = np.zeros((3, 4, 84, 84), dtype=np.uint8)
    reset_obs[0].fill(10)
    reset_obs[1].fill(20)
    reset_obs[2].fill(30)
    final_obs = np.empty(3, dtype=object)
    final_obs[:] = None
    final_obs[1] = np.full((4, 84, 84), 99, dtype=np.uint8)

    transition_obs = atari._transition_observations(
        reset_obs,
        np.array([False, True, False]),
        {"final_obs": final_obs},
    )

    assert transition_obs is not reset_obs
    np.testing.assert_array_equal(transition_obs[0], reset_obs[0])
    np.testing.assert_array_equal(transition_obs[1], final_obs[1])
    np.testing.assert_array_equal(transition_obs[2], reset_obs[2])
    assert np.all(reset_obs[1] == 20), "reset batch must remain untouched"


def test_transition_observations_reuse_batch_without_boundaries():
    atari = _load_atari_example()
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)

    result = atari._transition_observations(observations, np.array([False, False]), {})

    assert result is observations
