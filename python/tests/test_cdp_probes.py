import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import probe_cdp as probe


class Environment:
    action_space = SimpleNamespace(n=18)

    def __init__(self):
        self.tick = 0
        self.unwrapped = SimpleNamespace(ale=SimpleNamespace(getRAM=self.ram))

    def ram(self):
        ram = np.zeros(128, np.uint8)
        ram[70], ram[97] = self.tick % 150, self.tick % 100
        return ram

    def reset(self, *, seed=None):
        if seed is not None:
            self.tick = 0
        return self.tick, {}

    def step(self, action):
        self.tick += 1
        return self.tick, float(self.tick % 11 == 0), self.tick == 20, self.tick == 39, {}


class Agent:
    learner_step = 49939
    environment_step = 200000

    def __init__(self):
        self.active = False
        self.calls = []

    def begin_episode(self, frame):
        assert not self.active
        self.active = True
        self.latent_feature = [0., 0., 1., 0.]
        self.encoded_observation = [frame, -frame]

    @staticmethod
    def transition(state, action):
        return [state[0] + action, state[1] + 1, 0., 1.]

    def act(self, *, action_mask):
        self.action = action_mask.index(True)
        return self.action

    def observe(self, frame, **kwargs):
        assert self.active
        self.environment_step += 1
        self.latent_feature = self.transition(self.latent_feature, self.action)
        self.encoded_observation = [frame, -frame]
        self.active = not (kwargs['terminated'] or kwargs['truncated'])

    def forecast_states(self, actions):
        assert self.active
        self.calls.append((self.environment_step, actions))
        states = []
        state = self.latent_feature
        for action in actions:
            state = self.transition(state, action)
            states.append(state)
        return [states[0], states[-1]], [states[0][:2], states[-1][:2]]

    def prior_behavior_rollout(self, actions):
        return [0.] * len(actions), [1.] * len(actions), [0.] * len(actions)


def trace(*, cdp=True, steps=65):
    return probe.collect_trace(Agent(), Environment(), 9101, steps, lambda: None, cdp=cdp, deter=2)[0]


def test_collection_is_causal_keeps_boundaries_and_filters_crossings():
    agent, checks = Agent(), []
    data, info = probe.collect_trace(agent, Environment(), 9101, 65,
                                     lambda: checks.append(True), cdp=True, deter=2)
    assert info['max_h1_deter_error'] == 0 and info['actions'] == 65 and info['arrivals'] == 68
    assert not agent.active and agent.learner_step == 49939 and agent.environment_step == 200065
    assert len(checks) == 2
    np.testing.assert_array_equal(data['origins_h1'], [0, 16, 32, 48, 64])
    np.testing.assert_array_equal(data['origins_h15'], [0, 48])
    np.testing.assert_array_equal(data['prior_h1'][:, :2], data['posterior'][data['following'][data['origins_h1']], :2])
    assert data['terminated'][19] and data['truncated'][38]
    assert data['collection_cut'][-1] and not data['terminated'][-1] and not data['truncated'][-1]
    assert [step - 200000 for step, _ in agent.calls[::2]] == [0, 16, 32, 48, 64]
    for (step, actions) in agent.calls[::2]:
        np.testing.assert_array_equal(actions, data['actions'][step - 200000:step - 200000 + len(actions)])


def test_labels_do_not_enter_forecasts_and_rgb_omits_embedding_predictions(monkeypatch):
    original = trace()
    monkeypatch.setattr(probe, 'positions', lambda *args: np.array([99., 33.]))
    changed = trace()
    assert not np.array_equal(original['positions'], changed['positions'])
    for key in original:
        if key != 'positions':
            np.testing.assert_array_equal(original[key], changed[key])
    rgb = trace(cdp=False)
    assert not any(key.startswith(('predicted_', 'unrelated_predicted_')) for key in rgb)
    assert probe.trace_identity(changed) == probe.trace_identity(rgb)
    assert probe.trace_identity(original) != probe.trace_identity(changed)


@pytest.mark.parametrize('failure', ['alignment', 'updates', 'action', 'nonfinite'])
def test_actor_failures_are_fatal(failure):
    actor = Agent()
    if failure == 'action':
        actor.act = lambda **_: -1
    elif failure == 'alignment':
        original = actor.forecast_states

        def forecast(actions):
            states, predicted = original(actions)
            states[0][0] += 1
            return states, predicted

        actor.forecast_states = forecast
    else:
        original = actor.observe

        def observe(*args, **kwargs):
            original(*args, **kwargs)
            if failure == 'updates':
                actor.learner_step += 1
            else:
                actor.encoded_observation = [float('nan'), 0.]

        actor.observe = observe
    with pytest.raises(RuntimeError):
        probe.collect_trace(actor, Environment(), 1, 32, lambda: None, cdp=True, deter=2)


def test_current_state_and_future_labels_are_distinct():
    data = trace()
    state = probe.examples(data, 'cnn', 0)
    future = probe.examples(data, 'prior', 15)
    np.testing.assert_array_equal(state['x'], data['cnn'][data['following']])
    following = data['following'][future['origins'] + 14]
    np.testing.assert_array_equal(future['labels'], data['positions'][following])
    np.testing.assert_array_equal(future['future_cnn'], data['cnn'][following])
    np.testing.assert_array_equal(future['reward'], data['rewards'][future['origins'] + 14])
    assert not np.array_equal(future['x'], future['current_posterior'])


def test_cosine_and_spread_handle_zero_constant_and_antiparallel():
    p = np.array([[0., 0.], [1., 0.], [0., 2.], [-1., 0.]])
    y = np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    np.testing.assert_allclose(probe.cosine_error(p, y), [1., 0., 0., 2.])
    assert probe.spread(np.ones((8, 4)))['effective_rank'] == 0
    assert probe.spread(np.array([[-1., 0.], [1., 0.]]))['effective_rank'] == 1
    data = probe.examples(trace(), 'prior', 15)
    data['seeds'] = np.full(len(data['x']), 9101)
    result = probe.forecast_report(data, np.zeros(2))
    assert set(result['all']['cosine_distance']) == {'prior', 'unrelated_actions', 'persistence', 'training_mean'}
    assert set(result['all']['reward']) == {'prior', 'unrelated_actions', 'zero'}
    assert sum(row['count'] for row in result.get('by_trajectory', {}).values()) == result['all']['count']


def test_readout_pipeline_uses_four_heads_fixed_split_and_frozen_controls(tmp_path, monkeypatch):
    result = dict(files=[], heads=[], memory=[])
    for split, seeds in probe.SPLITS.items():
        for seed in seeds:
            path = tmp_path / f'{split}-{seed}.npz'
            np.savez_compressed(path, **trace())
            result['files'].append(dict(file=path.name, split=split, seed=seed))

    class Model:
        gpu_device = dict(device_name='NVIDIA GeForce RTX 5080', driver_info='580.178.04')
        gpu_memory_budget = dict(budget_bytes=4 << 30, usage_bytes=0)
        updates = 0

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, *args):
            pass

        def predict(self, _):
            return [0.] * 128

        def learn(self, *args, **kwargs):
            self.updates += 1
            return 1.

        def parameters(self):
            return [[0.]]

        def set_parameters(self, value):
            assert value == [[0.]]

    model = Model()
    monkeypatch.setattr(probe.kindle._native, 'RegressionProbe', lambda *args, **kwargs: model)
    probe.fit_readouts(tmp_path, result, lambda: None, steps=2)
    assert model.updates == 8
    assert [(row['stage'], row['horizon']) for row in result['heads']] == [('cnn', 0), ('posterior', 0), ('prior', 1), ('prior', 15)]
    for row in result['heads']:
        assert row['fit']['selection'] == 'validation normalized MSE; no refit or test selection'
        with np.load(tmp_path / row['evidence_file']) as data:
            assert set(data['seeds']) == set(probe.SPLITS['test'])
        if row['horizon']:
            assert set(row['readouts']) == {'fitted', 'training_mean', 'unrelated_actions', 'posterior_persistence'}
    json.dumps(result, allow_nan=False)
