import numpy as np
import pytest

from kindle._breakout_world import BRICK_COLORS, ball_center, brick_metrics, evaluation_windows, event_metrics, image_metrics, paddle_center, sample_locations, training_windows


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
    assert training_windows([stream], [2], 4, allow_resets=True)['action'][:, 0].tolist() == [-1, 3, -1, 5]
    stream['terminal'][3] = True
    kept = training_windows([stream], [3], 4, allow_resets=True, retain_initial_targets=True)
    assert kept['action'][0, 0] == -1 and kept['reward'][0, 0] == 3 and kept['terminal'][0, 0]


def test_uniform_sampling_includes_terminal_and_reset_windows():
    streams = [dict(action=np.arange(4)), dict(action=np.arange(5))]
    locations = sample_locations(streams, 3, 3000, np.random.default_rng(17))
    assert set(locations) == {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)}
    counts = [locations.count(location) for location in set(locations)]
    assert max(counts)-min(counts) < 150


def test_evaluation_keeps_terminal_target_but_never_scores_next_episode_or_padding():
    stream = dict(rgb=np.zeros((7, 3, 64, 64), np.uint8), action=np.array([-1, 2, 2, 2, -1, 3, 3]),
                  reward=np.zeros(7), terminal=np.array([False, False, False, True, False, False, False]),
                  position=np.zeros((7, 3)), lives=np.ones(7))
    batch = evaluation_windows([stream], [(0, 0)], history=2, horizon=6)
    assert batch['valid'][:, 0].tolist() == [True, True, False, False, False, False]
    assert batch['terminal'][3, 0]
    assert batch['action'][2:, 0].tolist() == [2, 2, 0, 0, 0, 0]
    tail = evaluation_windows([stream], [(0, 4)], history=2, horizon=3)
    assert tail['valid'][:, 0].tolist() == [True, False, False]


def test_empty_image_slice_stays_untested_and_paddle_motion_is_measured():
    empty = np.empty((0, 3, 64, 64))
    assert image_metrics(empty, empty, empty)['mse'] is None
    target = np.full((1, 3, 64, 64), -.5)
    target[0, 0, 58, 20:26] = .3
    predicted = np.roll(target, 2, axis=-1)
    metrics = image_metrics(target, predicted, target)
    assert metrics['paddle_detected'] == 1
    assert metrics['paddle_mae_detected'] == 2 and metrics['paddle_within_1px'] == 0


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


def test_static_full_wall_cannot_pass_a_brick_removal():
    from PIL import Image
    native = np.zeros((210, 160, 3), np.uint8)
    for row, color in enumerate(BRICK_COLORS):
        native[57+row*6:63+row*6, 8:152] = np.rint(color*255).astype(np.uint8)
    def resized(value):
        rgb = np.asarray(Image.fromarray(value).resize((64, 64), Image.Resampling.BILINEAR))
        return rgb.transpose(2, 0, 1)[None].astype(float)/255-.5
    full = resized(native)
    native[69:75, 48:56] = 0
    removed = resized(native)
    static = brick_metrics(removed, full, full)
    assert static['present'] == 107 and static['absent'] == 1
    assert static['removed'] == static['false_additions'] == 1
    assert static['removed_recall'] == 0 and static['balanced_error'] == .5
    exact = brick_metrics(removed, removed, full)
    assert exact['removed_recall'] == 1 and exact['balanced_error'] == 0


def test_ball_near_or_inside_paddle_band_does_not_shift_the_paddle_center():
    image = np.full((3, 64, 64), -.5)
    image[0, 56, 35:41] = .3
    image[0, 58, 57:60] = .3
    assert paddle_center(image) == pytest.approx(58)
    image[:] = -.5
    image[0, 58, 20:26] = .3
    image[0, 58, 45:47] = .3
    assert paddle_center(image) == pytest.approx(22.5)
    image[0, 58, 20:26] = -.5
    image[0, 58, 20:22] = .3
    assert paddle_center(image) is None
