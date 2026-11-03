"""Raw-frame life accounting and optional actor-boundary integration, without GPU."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import kindle
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import atari_vector
from atari import DreamerAtariPreprocessing
import kindle._return_boundaries as module
from kindle._vector_audit import audit
from test_breakout_corrections import corrective_events
from test_replay_actor import command
from test_vector import write_log


OPTIONS = ['--cdp', '--replay-actor-scale', '100', '--disagreement-scale', '1',
           '--replay-actor-observed-lower-bound', '--breakout-return-life-boundaries']


class RawEnvironment(gym.Env):
    action_space = gym.spaces.Discrete(18)
    observation_space = gym.spaces.Box(0, 255, (210, 160, 3), dtype=np.uint8)

    def __init__(self):
        self.ale = SimpleNamespace(lives=lambda: self.life_count)
        self.closed = False

    def get_action_meanings(self):
        return ['NOOP', 'FIRE'] + [str(i) for i in range(2, 18)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = 0
        self.life_count = 3
        return np.zeros((210, 160, 3), dtype=np.uint8), {'reset': True}

    def step(self, action):
        self.frame += 1
        if self.frame in (1, 6): self.life_count -= 1
        if self.frame == 2: self.life_count += 1
        return np.full((210, 160, 3), self.frame, dtype=np.uint8), 1., self.frame == 8, False, {'action': action}

    def close(self):
        self.closed = True


def test_raw_monitor_preserves_pooling_rewards_flags_and_loss_gain():
    monitored = module.LifeBoundaryMonitor(RawEnvironment())
    wrapped = DreamerAtariPreprocessing(monitored, noop_max=0, screen_size=None)
    control = DreamerAtariPreprocessing(RawEnvironment(), noop_max=0, screen_size=None)
    np.testing.assert_array_equal(wrapped.reset(seed=7)[0], control.reset(seed=7)[0])
    for expected in ((1, 1), (1, 0)):
        actual, reference = wrapped.step(3), control.step(3)
        np.testing.assert_array_equal(actual[0], reference[0])
        assert actual[1:] == reference[1:]
        assert monitored.take_changes() == expected
        assert monitored.take_changes() == (0, 0)
    assert wrapped.executed_action_frames == control.executed_action_frames == 8
    wrapped.reset()
    assert monitored.take_changes() == (0, 0)
    monitored.env.life_count = 2
    monitored.env.frame = 1
    monitored.step(0)  # A gain alone does not imply a boundary.
    assert monitored.take_changes() == (0, 1)
    monitored.env.frame = 0
    monitored.step(0)
    monitored.reset()
    assert monitored.take_changes() == (0, 0)


@pytest.mark.parametrize('extra', [[], ['--evaluate'], ['--restore', 'unused'], ['ALE/Pong-v5']])
def test_invalid_boundary_options_fail_before_environment(monkeypatch, tmp_path, capsys, extra):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    options = ['--breakout-return-life-boundaries'] if not extra else OPTIONS + extra
    args = command(tmp_path, options)
    if extra == ['ALE/Pong-v5']: args.remove('ALE/Breakout-v5')
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit): atari_vector.main()
    assert 'require fresh Breakout training' in capsys.readouterr().err


@pytest.mark.parametrize('restore', [False, True])
def test_specific_native_capability_required_before_environment(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setitem(kindle.__dict__, 'VectorAgent', SimpleNamespace(
        SUPPORTS_COMPLETED_RETURN_ACTOR=True, SUPPORTS_CONSERVATIVE_RETURN_ACTOR=True,
        act=lambda: None))
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    options = OPTIONS
    if restore:
        checkpoint = tmp_path/'checkpoint'
        checkpoint.mkdir()
        (checkpoint/'metadata.json').write_text(json.dumps(dict(config=dict(
            replay_actor_scale=100., replay_actor_observed_lower_bound=True,
            replay_actor_auxiliary_boundaries=True))))
        options = ['--restore', str(checkpoint), '--evaluate', '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', command(tmp_path, options))
    with pytest.raises(SystemExit): atari_vector.main()
    assert 'native actor-return boundary support' in capsys.readouterr().err


@pytest.mark.parametrize('mode', ['default', 'enabled', 'frozen'])
def test_runner_routes_only_training_boundaries_and_preserves_episodes(monkeypatch, tmp_path, mode):
    environments, observed = [], []

    class Agent:
        SUPPORTS_COMPLETED_RETURN_ACTOR = SUPPORTS_CONSERVATIVE_RETURN_ACTOR = SUPPORTS_ACTOR_RETURN_BOUNDARIES = True
        environment_step = learner_step = replay_len = training_debt = 0
        provenance = gpu_device = trainable_parameter_counts = {}
        cpu_worker_threads = 1

        @classmethod
        def learned_rgb(cls, streams, config):
            instance = cls()
            instance.config = config
            assert config.get('replay_actor_auxiliary_boundaries', False) == (mode == 'enabled')
            return instance

        @classmethod
        def restore_rgb(cls, path, streams):
            instance = cls()
            instance.config = kindle.default_config(18)
            instance.config.update(observation_kind='rgb64', replay_actor_scale=100.,
                replay_actor_observed_lower_bound=True, replay_actor_auxiliary_boundaries=True,
                disagreement_bonus=True, intrinsic_reward_scale=1.)
            return instance

        def begin_episodes(self, streams, frames): self.replay_len += len(streams)
        def act(self, *, greedy): return [0, 0]

        def observe(self, streams, frames, rewards, terminated, truncated, **kwargs):
            assert kwargs == ({'actor_boundaries': [True, True]} if mode == 'enabled' else {})
            observed.append((rewards, terminated, truncated))
            self.environment_step += len(streams)
            self.replay_len += len(streams)
            return [[reward, 0.] for reward in rewards]

        def learn_scheduled(self):
            assert mode != 'frozen'
            return []

    def make(*args, **kwargs):
        env = RawEnvironment()
        environments.append(env)
        return env

    monkeypatch.setitem(kindle.__dict__, 'VectorAgent', Agent)
    monkeypatch.setattr(atari_vector.gym, 'make', make)
    monkeypatch.setattr(atari_vector, 'checkpoint_identity', lambda path: {'fixture': True})
    if mode != 'enabled':
        monkeypatch.setattr(atari_vector, 'LifeBoundaryMonitor', lambda env: pytest.fail('unexpected training aid'))
    options = ['--num-envs', '2', '--steps', '6', '--train-ratio', '0']
    if mode == 'enabled': options += OPTIONS
    elif mode == 'frozen':
        options = ['--num-envs', '2', '--steps', '6', '--evaluate', '--restore', 'fixture',
                   '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', command(tmp_path, options))
    atari_vector.main()
    rows = [json.loads(line) for line in (tmp_path/'run.jsonl').read_text().splitlines()]
    assert all(env.closed for env in environments)
    assert len(observed) == 3 and observed[1] == ([4., 4.], [True, True], [False, False])
    assert rows[-1]['episode_counts'] == [1, 1] and rows[-1]['total_rewards'] == [12., 12.]
    assert ('actor_return_boundary_kind' in rows[0]) == (mode == 'enabled')
    transitions = [row for row in rows if row['event'] == 'transition']
    if mode == 'enabled':
        assert [row['actor_return_losses'] for row in transitions] == [[1, 1]] * 3
        assert [row['actor_return_gains'] for row in transitions] == [[1, 1], [0, 0], [1, 1]]
    else:
        assert all('actor_return_boundaries' not in row for row in transitions)
    assert audit(tmp_path/'run.jsonl')['updates'] == 0


@pytest.mark.parametrize('mutation', [None, 'config_type', 'lower_bound', 'scale', 'undeclared', 'kind',
    'hash', 'game', 'frozen', 'missing', 'length', 'flag_type', 'count_type', 'negative', 'mismatch', 'reward'])
def test_boundary_log_contract(tmp_path, mutation):
    rows = corrective_events()
    header, config = rows[0], rows[0]['config']
    config.update(replay_actor_scale=100., replay_actor_observed_lower_bound=True,
        replay_actor_auxiliary_boundaries=True, disagreement_bonus=True, intrinsic_reward_scale=1.)
    header.update(actor_return_boundary_kind=module.LIFE_BOUNDARY_KIND,
        actor_return_boundary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    for event in rows:
        if event['event'] == 'transition':
            event.update(actor_return_boundaries=[True, False], actor_return_losses=[1, 0], actor_return_gains=[1, 0])
        elif event['event'] == 'learner':
            event['report']['behavior'] = dict(replay_actor=dict(loss=.5, completed_rows=1, positive_rows=1, weight_sum=.5))
    target = next(event for event in rows if event['event'] == 'transition')
    if mutation == 'config_type': config['replay_actor_auxiliary_boundaries'] = 1
    elif mutation == 'lower_bound': config['replay_actor_observed_lower_bound'] = False
    elif mutation == 'scale': config['extrinsic_reward_scale'] = -1
    elif mutation == 'undeclared': del header['actor_return_boundary_kind']
    elif mutation == 'kind': header['actor_return_boundary_kind'] = 'unknown'
    elif mutation == 'hash': header['actor_return_boundary_sha256'] = '0'*64
    elif mutation == 'game': header['environment'] = 'ALE/Pong-v5'
    elif mutation == 'frozen': header['mode'] = 'evaluate_sample'
    elif mutation == 'missing': del target['actor_return_gains']
    elif mutation == 'length': target['actor_return_losses'].pop()
    elif mutation == 'flag_type': target['actor_return_boundaries'][0] = 1
    elif mutation == 'count_type': target['actor_return_losses'][0] = True
    elif mutation == 'negative': target['actor_return_gains'][0] = -1
    elif mutation == 'mismatch': target['actor_return_boundaries'][1] = True
    elif mutation == 'reward': target['stored_rewards'][0][1] = -1
    if mutation is None:
        assert audit(write_log(tmp_path, rows))['updates'] == 3
    else:
        with pytest.raises(ValueError): audit(write_log(tmp_path, rows))
