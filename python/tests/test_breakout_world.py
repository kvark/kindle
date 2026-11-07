import numpy as np
import pytest

from kindle._breakout_world import ball_center, event_metrics, image_metrics, training_windows


def test_window_alignment_and_context_reset_do_not_modify_source():
    stream = dict(rgb=np.zeros((8, 3, 64, 64), np.uint8), action=np.arange(8),
                  reward=np.arange(8, dtype=float), terminal=np.zeros(8, bool))
    batch = training_windows([stream], [2], 4)
    assert batch['action'][:, 0].tolist() == [-1, 3, 4, 5]
    assert batch['reward'][:, 0].tolist() == [0, 3, 4, 5]
    assert stream['action'][2] == stream['reward'][2] == 2
    stream['action'][4] = -1
    with pytest.raises(ValueError, match='reset'):
        training_windows([stream], [2], 4)


def test_missing_ball_cannot_pass_on_black_background():
    target = np.full((1, 3, 64, 64), -.5)
    target[0, 0, 40, 30] = .3
    black = np.full_like(target, -.5)
    metrics = image_metrics(target, black, black)
    assert metrics['mse'] < .001
    assert metrics['ball_visible'] == 1 and metrics['ball_detected'] == 0
    assert metrics['ball_within_1px'] == 0
    assert metrics['ball_error_mean_detected'] is None
    exact = image_metrics(target, target, black)
    assert exact['ball_within_1px'] == 1 and exact['changed_mse'] == 0


def test_wrong_position_and_hallucinations_are_counted():
    target = np.full((2, 3, 64, 64), -.5)
    prediction = target.copy()
    target[0, 0, 40, 30] = .3
    prediction[:, 0, 40, 34] = .3
    metrics = image_metrics(target, prediction, np.full_like(target, -.5))
    assert metrics['ball_error_mean_detected'] == 4
    assert metrics['ball_within_1px'] == 0 and metrics['ball_false_detections'] == 1
    assert ball_center(target[1]) is None


def test_rare_endings_and_missing_support_stay_explicit():
    target = np.array([True] + [False]*99)
    result = event_metrics(target, np.zeros(100))
    assert result['brier'] == .01 and result['balanced_brier'] == .5
    assert result['positive_brier'] == 1
    assert event_metrics(np.zeros(4, bool), np.zeros(4))['balanced_brier'] is None
    with pytest.raises(ValueError):
        event_metrics(target, np.full(100, 1.1))
