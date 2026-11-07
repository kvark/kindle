from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
import breakout_world_branches as branches


def test_energy_score_rewards_correct_location_and_forecast_spread():
    truth = np.array([[0.], [2.]])
    assert branches.energy_score(truth, [.5, .5], truth) == .5
    assert branches.energy_score([[1.]], [1.], truth) == 1.
    assert branches.energy_score([[5.]], [1.], truth) == 4.
    with pytest.raises(ValueError): branches.energy_score(truth, [.5, -.5], truth)


def test_terminal_reward_occurs_before_absorption_and_discount_is_removed():
    # On step1, reward occurs with p=.25 and death with p=.5. Reward on
    # the dying step is retained. On step2 only the surviving half can reward.
    reward = np.array([[[2.], [4.], [99.]]])
    positive = np.array([[[.25], [1.], [1.]]])
    continuation = np.array([[[.45], [0.], [.9]]])
    survival, any_reward, total = branches.event_forecasts(reward, positive, continuation, .9)
    np.testing.assert_allclose(survival.ravel(), [.5, 0, 0])
    np.testing.assert_allclose(any_reward.ravel(), [.25, .625, .625])
    np.testing.assert_allclose(total.ravel(), [2, 4, 4])


def test_native_bricks_axes_and_ambiguous_half_cells():
    from kindle._breakout_world import BRICK_COLORS
    image = np.zeros((2, 5, 8, 210, 160, 3), np.uint8)
    color = np.rint(BRICK_COLORS[4]*255).astype(np.uint8)
    image[1, 3, 6, 81:87, 144:152] = color
    image[0, 2, 1, 81:84, 144:152] = color
    occupied, valid = branches.native_bricks(image)
    assert occupied.shape == valid.shape == (2, 5, 8, 108)
    assert occupied.sum() == 1 and occupied[1, 3, 6, 4*18+17]
    assert (~valid).sum() == 1 and not valid[0, 2, 1, 4*18+17]


def test_preprocess_uses_eight_native_images_without_cpu_geometry_change():
    calls = []
    def resize(batch):
        calls.append(np.stack(batch).shape)
        # Exact endpoint values for the native package's centered RGB contract.
        return np.full((8, 3, 64, 64), -.5, '<f4').tobytes()
    value = branches.preprocess(SimpleNamespace(preprocess=resize), np.zeros((2, 8, 210, 160, 3), np.uint8))
    assert calls == [(8, 210, 160, 3)]*2 and value.shape == (2, 8, 3, 64, 64)
    assert not value.any()
    with pytest.raises(ValueError): branches.preprocess(None, np.zeros((8, 64, 64, 3)))


def test_plans_are_fixed_before_future_outcomes_and_cover_all_serve_commands():
    serves = branches.plans('serve')
    assert len(serves) == 18
    for command, (_, actions) in enumerate(serves): np.testing.assert_array_equal(actions, [command]*32)
    assert [name for name, _ in branches.plans('contact')] == ['noop', 'left', 'right', 'reverse']
    np.testing.assert_array_equal(branches.plans('wall')[-1][1], [3]*4+[4]*28)


def test_origin_selection_rejects_short_history_life_boundary_and_terminal():
    history = [dict(position=np.array([50., 165., 40.]), terminal=False, lives=2) for _ in range(32)]
    history[-2] = dict(position=np.array([50., 160., 40.]), terminal=False, lives=2)
    assert branches.eligible('contact', history)
    assert not branches.eligible('contact', history[1:])
    history[-1]['lives'] = 1
    assert not branches.eligible('contact', history)
    history[-1]['lives'] = 2; history[-1]['terminal'] = True
    assert not branches.eligible('contact', history)


def test_snapshot_disk_roundtrip_keeps_bytes_frames_ram_and_counters(tmp_path):
    snap = branches.Snapshot(b'full', b'game', [np.zeros((2, 3, 3), np.uint8)]*2,
                             [dict(type='fixture', fields={'_elapsed_steps': 7})], np.arange(128, dtype=np.uint8))
    snap.save(tmp_path/'origin')
    restored = branches.Snapshot.load(tmp_path/'origin')
    assert restored.full == snap.full and restored.game == snap.game and restored.counters == snap.counters
    np.testing.assert_array_equal(restored.frames, snap.frames)
    np.testing.assert_array_equal(restored.ram, snap.ram)


def test_absorbing_forecast_matches_terminal_oracle_while_persistence_fails():
    draws, repeats = 2, 2
    prediction = dict(prediction=np.full((draws, 32, 8, 3, 64, 64), -.5, np.float32),
        posterior=np.full((draws, 8, 3, 64, 64), -.5, np.float32),
        reward=np.zeros((draws, 32, 8)), positive=np.zeros((draws, 32, 8)), continuation=np.zeros((draws, 32, 8)))
    terminal = np.zeros((repeats, 32, 8), bool); terminal[:, 0] = True
    oracle = dict(rgb=np.zeros((repeats, 5, 8, 3, 64, 64), np.uint8),
        native_rgb=np.zeros((repeats, 5, 8, 210, 160, 3), np.uint8),
        reward=np.zeros((repeats, 32, 8)), terminal=terminal)
    report, _ = branches.score(prediction, oracle, np.full((32, 8, 3, 64, 64), -.5), .9)
    for row in report['horizons'].values():
        assert row['energy'] == dict(ball=0., paddle=0.)
        assert row['controls']['ball']['persistence'] == 1.
        assert row['events']['end_brier'] == 0 and row['brick_cells_scored'] == 0
