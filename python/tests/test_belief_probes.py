import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import probe_rssm_belief as probe
from test_rssm_latents import trace


class Core:
    deterministic_size = 2
    learner_step = 2048
    environment_step = 12288

    def __init__(self):
        self.latent_feature = [0., 0., 1., 0.]
        self.encoded_observation = [0., 0.]
        self.active = False
        self.resets = 0
        self.calls = []

    @staticmethod
    def transition(state, action):
        return [state[0] + action, state[1] + 1., 0., 1.]

    def begin_episode(self, features):
        assert not self.active
        self.active = True
        self.resets += 1
        self.latent_feature = [0., 0., 1., 0.]
        self.encoded_observation = np.frombuffer(features, '<f4')[:2].tolist()

    def observe(self, action, features, reward, terminated, truncated):
        assert self.active
        self.calls.append(action)
        self.latent_feature = self.transition(self.latent_feature, action)
        self.encoded_observation = np.frombuffer(features, '<f4')[:2].tolist()
        self.environment_step += 1
        self.active = not (terminated or truncated)

    def forecast_states(self, actions):
        assert self.active
        states, features = [], []
        state = self.latent_feature
        for action in actions:
            state = self.transition(state, action)
            states.append(state)
            features.append([*state[:2], -1.])
        return [states[0], states[-1]], [features[0], features[-1]]


def data():
    value = trace()
    value.update(positions=np.zeros((41, 6), np.float32), phase=np.arange(41) % 16,
                 collection_cut=np.arange(39) == 38)
    return value


def test_collection_captures_every_arrival_and_prior_alignment():
    core, source = Core(), data()
    captured, error = probe.collect(core, source, 123)
    assert error == 0. and core.learner_step == 2048 and core.environment_step == 12327
    assert core.resets == 2 and not core.active and core.calls == source['actions'].tolist()
    np.testing.assert_array_equal(captured['adapter'], source['features'][:, :2])
    np.testing.assert_array_equal(captured['origins_h1'], np.arange(0, 39, 4))
    np.testing.assert_array_equal(captured['origins_h15'], [0, 4, 20, 24])
    following = source['following'][captured['origins_h1']]
    np.testing.assert_array_equal(captured['prior_h1'][:, :2], captured['posterior'][following, :2])
    assert all(np.isfinite(v).all() for v in captured.values())


def test_prefix_keeps_terminal_targets_and_excludes_unused_reset():
    source = probe.prefix(data(), 20)
    assert len(source['features']) == len(source['positions']) == len(source['phase']) == 21
    assert len(source['actions']) == 20 and source['terminated'][-1]
    core = Core()
    values, _ = probe.collect(core, source, 123)
    assert core.resets == 1 and not core.active and len(values['posterior']) == 21


def test_alignment_failure_is_fatal():
    core = Core()
    original = core.forecast_states

    def broken(actions):
        states, features = original(actions)
        states[0][0] += 1.
        return states, features

    core.forecast_states = broken
    with pytest.raises(RuntimeError, match='h1 prior'):
        probe.collect(core, data(), 123)


def test_loading_distinguishes_posterior_estimates_from_forecasts(tmp_path):
    corpus, output = tmp_path / 'corpus', tmp_path / 'states'
    corpus.mkdir()
    output.mkdir()
    source = data()
    source['positions'][:, 0] = np.arange(41)
    values, _ = probe.collect(Core(), source, 123)
    np.savez(corpus / 'trace.npz', **source)
    np.savez(output / 'trace.npz', **values)
    files = [dict(file='trace.npz', split='train', seed=123)]
    posterior = probe.load_data(corpus, output, files, 'train', 'posterior', 0, 2, False)
    np.testing.assert_array_equal(posterior['labels'][:, 0], source['following'])
    for stage in ('deter', 'prior', 'predicted'):
        loaded = probe.load_data(corpus, output, files, 'train', stage, 15, 2, False)
        origins = values['origins_h15']
        following = source['following'][origins + 14]
        np.testing.assert_array_equal(loaded['labels'][:, 0], following)
        np.testing.assert_array_equal(loaded['future_posterior'], values['posterior'][following])
        np.testing.assert_array_equal(loaded['current_posterior'], values['posterior'][source['current'][origins]])
        assert loaded['x'].shape[1] == {'deter': 2, 'prior': 4, 'predicted': 3}[stage]
        assert loaded['unrelated'].shape == loaded['x'].shape
