"""Completed replay-value selection and reports, with construction forbidden."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest

import atari_vector
import kindle
from kindle._vector_audit import audit
from test_breakout_corrections import corrective_events
from test_vector import write_log


def command(tmp_path, args):
    return ['atari_vector.py', '--output', str(tmp_path/'never.jsonl'), *args]


@pytest.mark.parametrize('args,message', [
    (['--restore', 'unused'], 'requires fresh training'),
    (['--evaluate'], 'requires fresh training'),
    (['--disagreement-scale', '1'], 'disagreement-scale0'),
    (['--breakout-return-life-boundaries'], 'game-level returns'),
    *[(['--replay-value-scale', scale], 'positive') for scale in ('0', '-1', 'nan', 'inf', '1e-50')],
])
def test_invalid_completed_values_fail_before_environment(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, ['--replay-value-completed', *args]))
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('failure', ['old_native', 'device_targets'])
def test_completed_values_require_capability_and_cpu_targets(monkeypatch, tmp_path, capsys, restore, failure):
    native = SimpleNamespace(SUPPORTS_COMPLETED_REPLAY_VALUE=failure != 'old_native')
    monkeypatch.setattr(kindle, 'VectorAgent', native)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    if failure == 'device_targets':
        monkeypatch.setenv('KINDLE_DEVICE_TARGETS', '1')
    args = ['--replay-value-completed']
    if restore:
        checkpoint = tmp_path/'checkpoint'
        checkpoint.mkdir()
        (checkpoint/'metadata.json').write_text(json.dumps(dict(config=dict(replay_value_completed=True))))
        args = ['--restore', str(checkpoint), '--evaluate', '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', command(tmp_path, args))
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert ('requires native' if failure == 'old_native' else 'CPU target construction') in capsys.readouterr().err


def test_completed_values_only_add_selected_config(monkeypatch, tmp_path):
    env = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **kw: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **kw: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('checked native boundary')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(
        learned_rgb=construct, SUPPORTS_COMPLETED_REPLAY_VALUE=True))
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', command(tmp_path, ['--cdp', '--cdp-centered',
            '--disagreement-scale', '0', *(['--replay-value-completed'] if enabled else [])]))
        with pytest.raises(RuntimeError, match='checked native boundary'):
            atari_vector.main()
    plain, candidate = configs
    assert 'replay_value_completed' not in plain
    assert candidate.pop('replay_value_completed') is True
    assert candidate == plain and not candidate['disagreement_bonus']


def test_completed_value_restore_preserves_checkpoint_config(monkeypatch, tmp_path):
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    config = kindle.default_config(18)
    config.update(observation_kind='rgb64', cdp_centered=True, replay_value_completed=True,
                  disagreement_bonus=False, intrinsic_reward_scale=0.)
    config['loss_scales'].update(reconstruction=0, future_prediction=500)
    (checkpoint/'metadata.json').write_text(json.dumps(dict(config=config, action_count=18)))
    env = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **kw: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **kw: env)
    monkeypatch.setattr(atari_vector, 'checkpoint_identity', lambda _: {})

    def restore(path, streams):
        assert path == str(checkpoint) and streams == 4
        assert json.loads((checkpoint/'metadata.json').read_text())['config'] == config
        raise RuntimeError('restore checked before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(restore_rgb=restore,
        SUPPORTS_COMPLETED_REPLAY_VALUE=True, SUPPORTS_REPLAY_VALUE_BOOTSTRAP=True))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, ['--restore', str(checkpoint),
        '--evaluate', '--observation-size', 'native']))
    with pytest.raises(RuntimeError, match='restore checked before GPU'):
        atari_vector.main()


@pytest.mark.parametrize('mutation', [None, 'zero', 'missing', 'unconfigured', 'negative', 'too_many',
                                    'boolean', 'bad_flag', 'bonus', 'boundary', 'zero_scale'])
def test_completed_value_reports_match_config(tmp_path, mutation):
    rows = corrective_events()
    config = rows[0]['config']
    config.update(replay_value_completed=True, disagreement_bonus=False)
    config['loss_scales']['replay_value'] = 1.
    reports = [event['report'] for event in rows if event['event'] == 'learner']
    for report in reports:
        report['behavior'] = dict(completed_value_rows=1)
    target = reports[0]['behavior']
    if mutation == 'zero': target['completed_value_rows'] = 0
    elif mutation == 'missing': target.clear()
    elif mutation == 'unconfigured': del config['replay_value_completed']
    elif mutation == 'negative': target['completed_value_rows'] = -1
    elif mutation == 'too_many': target['completed_value_rows'] = config['batch_size'] * (config['batch_length'] - 1) + 1
    elif mutation == 'boolean': target['completed_value_rows'] = True
    elif mutation == 'bad_flag': config['replay_value_completed'] = 1
    elif mutation == 'bonus': config.update(disagreement_bonus=True, intrinsic_reward_scale=1.)
    elif mutation == 'boundary': config['replay_actor_auxiliary_boundaries'] = True
    elif mutation == 'zero_scale': config['loss_scales']['replay_value'] = 0.
    if mutation in (None, 'zero'):
        assert audit(write_log(tmp_path, rows))['updates'] == 3
    else:
        with pytest.raises(ValueError):
            audit(write_log(tmp_path, rows))
