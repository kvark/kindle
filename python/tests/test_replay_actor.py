"""Completed-return actor config reaches native code without GPU construction."""
from pathlib import Path
import sys
from types import SimpleNamespace

import kindle
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import atari_vector
from kindle._vector_audit import audit
from test_breakout_corrections import corrective_events
from test_vector import write_log


def command(tmp_path, extra):
    return ['atari_vector.py', 'ALE/Breakout-v5', '--output', str(tmp_path/'run.jsonl'), *extra]


@pytest.mark.parametrize('extra,message', [
    *[(['--replay-actor-scale', value], 'finite and positive') for value in ('0', '-1', 'nan', 'inf', '1e-50')],
    (['--replay-actor-scale', '100', '--disagreement-scale', '1'], 'disagreement-scale0'),
    (['--replay-actor-scale', '100', '--restore', 'unused'], 'fresh training'),
    (['--replay-actor-scale', '100', '--evaluate'], 'fresh training'),
])
def test_invalid_actor_options_fail_before_environment(monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, extra))
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err


def test_resident_targets_fail_before_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('KINDLE_DEVICE_TARGETS', '1')
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, ['--replay-actor-scale', '100']))
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert 'CPU target construction' in capsys.readouterr().err


def test_old_native_cannot_silently_ignore_actor_loss(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(kindle.__dict__, 'VectorAgent', SimpleNamespace(
        act=lambda action_overrides=None, expert_actions=None: None))
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, ['--replay-actor-scale', '100']))
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert 'native completed-return actor support' in capsys.readouterr().err


@pytest.mark.parametrize('scale', [None, '100'])
def test_exact_native_config_and_default_omission(monkeypatch, tmp_path, scale):
    env = SimpleNamespace(action_space=SimpleNamespace(n=18), action_meanings=['NOOP']+list(map(str, range(1,18))),
                          reset=lambda **kw: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **kw: env)
    monkeypatch.setitem(kindle.__dict__, 'default_config', lambda actions, size: dict(action_count=actions, loss_scales={}))
    def construct(streams, config):
        if scale is None:
            assert 'replay_actor_scale' not in config
        else:
            assert config['replay_actor_scale'] == 100. and config['disagreement_bonus'] is False
        raise RuntimeError('checked native boundary')
    monkeypatch.setitem(kindle.__dict__, 'VectorAgent', SimpleNamespace(learned_rgb=construct,
        SUPPORTS_COMPLETED_RETURN_ACTOR=True, act=lambda action_overrides=None, expert_actions=None: None))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, [] if scale is None else ['--replay-actor-scale', scale]))
    with pytest.raises(RuntimeError, match='checked native boundary'):
        atari_vector.main()


@pytest.mark.parametrize('mutation', [None, 'missing', 'negative_loss', 'too_many', 'excess_weight',
                                    'empty_weight', 'empty_loss', 'boolean_count', 'bonus', 'bad_scale', 'unknown'])
def test_completed_return_actor_report_accounting(tmp_path, mutation):
    rows = corrective_events()
    rows[0]['config']['replay_actor_scale'] = 100.
    rows[0]['config']['disagreement_bonus'] = False
    reports = [e['report'] for e in rows if e['event'] == 'learner']
    for report in reports:
        report['behavior'] = dict(replay_actor=dict(loss=.5, completed_rows=1, positive_rows=1, weight_sum=.5))
    target = reports[0]['behavior']['replay_actor']
    if mutation == 'missing': reports[0]['behavior'].clear()
    elif mutation == 'negative_loss': target['loss'] = -1
    elif mutation == 'too_many': target['positive_rows'] = 2
    elif mutation == 'excess_weight': target['weight_sum'] = 2
    elif mutation == 'empty_weight': target['positive_rows'] = 0
    elif mutation == 'empty_loss': target.update(positive_rows=0, weight_sum=0)
    elif mutation == 'boolean_count': target['completed_rows'] = True
    elif mutation == 'bonus': rows[0]['config'].update(disagreement_bonus=True, intrinsic_reward_scale=1.)
    elif mutation == 'bad_scale': rows[0]['config']['replay_actor_scale'] = -1
    elif mutation == 'unknown': target['extra'] = 0
    if mutation is None:
        assert audit(write_log(tmp_path, rows))['updates'] == 3
    else:
        with pytest.raises(ValueError):
            audit(write_log(tmp_path, rows))
