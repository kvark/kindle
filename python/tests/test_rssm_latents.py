import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import fit_rssm_latents as fit


def trace():
    # Two episodes; the first terminal is still a valid forecast target.
    return dict(features=np.arange(41 * 3, dtype=np.float32).reshape(41, 3),
                current=np.r_[np.arange(20), np.arange(21, 40)],
                following=np.r_[np.arange(1, 21), np.arange(22, 41)],
                actions=np.arange(39) % 18, rewards=np.arange(39, dtype=np.float32),
                terminated=np.arange(39) == 19, truncated=np.zeros(39, bool),
                episodes=np.r_[np.zeros(20, int), np.ones(19, int)])


class Core:
    learner_step = 2048

    def __init__(self):
        self.active = False
        self.calls = []
        self.forecasts = []
        self.state = None

    def begin_episode(self, features):
        assert not self.active
        self.state = np.frombuffer(features, "<f4").copy()
        self.active = True
        self.calls.append(("reset", self.state.tolist()))

    def observe(self, action, features, reward, terminated, truncated):
        assert self.active
        self.state = np.frombuffer(features, "<f4").copy()
        self.calls.append((action, self.state.tolist(), reward, terminated, truncated))
        self.active = not (terminated or truncated)

    def forecast(self, actions):
        assert self.active
        self.forecasts.append(actions)
        return [1., float(len(actions))], [self.state + 1., self.state + len(actions)]


def test_recorded_ingest_closes_prefix_without_inventing_terminals_or_learning():
    core, data = Core(), trace()
    assert fit.replay_trace(core, data, 26) == (26, 28)
    assert core.learner_step == 2048 and not core.active
    resets = [row for row in core.calls if row[0] == "reset"]
    actions = [row for row in core.calls if row[0] != "reset"]
    assert len(resets) == 2 and [row[0] for row in actions] == data["actions"][:26].tolist()
    assert actions[19][3:] == (True, False)
    assert actions[-1][3:] == (False, True)
    np.testing.assert_array_equal([row[1] for row in actions], data["features"][data["following"][:26]])
    assert "positions" not in data  # No privileged information required.


def test_forecasts_are_prior_endpoints_and_do_not_cross_resets():
    core, data = Core(), trace()
    rows = fit.forecast_trace(core, data, 11113, 1)
    np.testing.assert_array_equal(rows[1]["origins"], np.arange(39))
    np.testing.assert_array_equal(rows[15]["origins"], np.r_[np.arange(6), np.arange(20, 25)])
    for h, row in rows.items():
        expected = data["features"][data["current"][row["origins"]]] + h
        np.testing.assert_array_equal(row["prediction"], expected)
        np.testing.assert_array_equal(row["reward_prediction"], h)
    assert core.learner_step == 2048 and len(core.forecasts) == 78
    for actions, unrelated in zip(core.forecasts[::2], core.forecasts[1::2]):
        assert len(actions) == len(unrelated) and all(0 <= a < 18 for a in unrelated)
    again = Core()
    fit.forecast_trace(again, data, 11113, 1)
    assert again.forecasts == core.forecasts


def test_training_statistics_never_consume_holdout_or_reset_arrivals(tmp_path):
    np.savez(tmp_path / "train.npz", features=np.array([[999., 999.], [1., 4.], [3., 4.]], np.float32),
             following=np.array([1, 2]), rewards=np.array([0., 2.]))
    manifest = dict(files=[dict(split="train", file="train.npz"), dict(split="test", file="does-not-exist")])
    mean, scale, reward = fit.training_statistics(tmp_path, manifest)
    np.testing.assert_array_equal(mean, [2., 4.])
    np.testing.assert_array_equal(scale, [1., 1.])
    assert mean.dtype == scale.dtype == np.float32 and reward == 1.


def test_config_changes_only_declared_settings_and_target_factor():
    base = dict(seed=2017, video_encoder="joint", train_ratio=32., replay_capacity=8192,
                loss_scales=dict(reward=1., future_prediction=.25))
    original = copy.deepcopy(base)
    mean, scale = np.zeros(3, np.float32), np.ones(3, np.float32)
    raw = fit.learner_config(base, mean, scale, False)
    standardized = fit.learner_config(base, mean, scale, True)
    assert base == original and raw["future_target_standardization"] is None
    assert standardized.pop("future_target_standardization") == dict(mean=[0.] * 3, scale=[1.] * 3)
    raw.pop("future_target_standardization")
    assert raw == standardized
    assert raw["video_encoder"] is None and raw["replay_capacity"] == 16384 and raw["train_ratio"] == 0.


def test_posterior_targets_change_only_the_optional_current_feature_head():
    base = dict(seed=2017, video_encoder='joint', loss_scales=dict(reconstruction=0., future_prediction=.25))
    mean, scale = np.ones(3, np.float32), np.full(3, .1, np.float32)
    control = fit.learner_config(base, mean, scale, True)
    candidate = fit.learner_config(base, mean, scale, True, True)
    assert candidate.pop('reconstruction_target_standardization') == candidate['future_target_standardization']
    assert control.pop('reconstruction_target_standardization') is None
    assert candidate['loss_scales'].pop('reconstruction') == .25
    assert control['loss_scales'].pop('reconstruction') == 0.
    assert candidate == control and base['loss_scales']['reconstruction'] == 0.
    with pytest.raises(ValueError, match='standardized future'):
        fit.learner_config(base, mean, scale, False, True)


def test_initial_comparison_allows_only_the_added_decoder_and_exact_shared_tensors(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir()
    right.mkdir()
    for group in ('world', 'behavior', 'slow_value'):
        values = {'weight': np.array([1.], np.float32)}
        save_file(values, left / f'{group}.safetensors')
        if group == 'world':
            values['world.decoder.trunk.weight'] = np.array([2.], np.float32)
        save_file(values, right / f'{group}.safetensors')
    assert fit.compare_initial(left, right)['world'] == dict(shared=1, added=1)
    save_file({'weight': np.array([3.], np.float32)}, right / 'behavior.safetensors')
    with pytest.raises(RuntimeError, match='initial control mismatch'):
        fit.compare_initial(left, right)
