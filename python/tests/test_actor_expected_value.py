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


@pytest.mark.parametrize('args,message', [
    (['--actor-expected-value', '--restore', 'unused'], 'requires fresh training'),
    (['--actor-expected-value', '--evaluate'], 'requires fresh training'),
    (['--actor-expected-value', '--actor-residual'], 'cannot combine'),
    (['--actor-expected-value', '--actor-dynamics-scale', '.5'], 'cannot combine'),
    (['--actor-expected-value', '--replay-actor-scale', '1', '--disagreement-scale', '0'], 'cannot combine'),
    (['--actor-expected-value', '--actor-critic-gradient'], 'cannot combine'),
])
def test_expected_value_rejects_invalid_cli_before_construction(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('restore', [False, True])
def test_expected_value_rejects_old_native(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--actor-expected-value']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(actor_expected_value=True))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'never.jsonl'), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert 'requires native expected-value actor support' in capsys.readouterr().err


def test_expected_value_only_adds_declared_config(monkeypatch, tmp_path):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct, SUPPORTS_ACTOR_EXPECTED_VALUE=True))
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
            '--cdp', '--cdp-centered', *(['--actor-expected-value'] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, expected_value = configs
    assert 'actor_expected_value' not in plain
    assert expected_value.pop('actor_expected_value') is True
    assert expected_value == plain


@pytest.mark.parametrize('mutation', [None, 'missing', 'unconfigured', 'nan', 'boolean', 'bad_flag', 'missing_timing', 'negative_timing', 'other_actor'])
def test_expected_value_report_matches_config(tmp_path, mutation):
    rows = corrective_events()
    rows[0]['config']['actor_expected_value'] = True
    reports = [event['report'] for event in rows if event['event'] == 'learner']
    for report in reports:
        report['behavior'] = dict(expected_value_actor_loss=-.4)
        report['timing'] = dict(expected_value_snapshot_seconds=.01)
    if mutation == 'missing': reports[0]['behavior'].clear()
    elif mutation == 'unconfigured': del rows[0]['config']['actor_expected_value']
    elif mutation == 'nan': reports[0]['behavior']['expected_value_actor_loss'] = float('nan')
    elif mutation == 'boolean': reports[0]['behavior']['expected_value_actor_loss'] = True
    elif mutation == 'bad_flag': rows[0]['config']['actor_expected_value'] = 1
    elif mutation == 'missing_timing': reports[0]['timing'].clear()
    elif mutation == 'negative_timing': reports[0]['timing']['expected_value_snapshot_seconds'] = -1
    elif mutation == 'other_actor': rows[0]['config']['actor_dynamics_scale'] = .5
    if mutation is None:
        assert audit(write_log(tmp_path, rows))['updates'] == 3
    else:
        with pytest.raises(ValueError):
            audit(write_log(tmp_path, rows))


@pytest.mark.parametrize('restore', [False, True])
def test_expected_value_rejects_device_targets_before_construction(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setenv('KINDLE_DEVICE_TARGETS', '1')
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(SUPPORTS_ACTOR_EXPECTED_VALUE=True))
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--actor-expected-value']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(actor_expected_value=True))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'never.jsonl'), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert 'requires CPU target construction' in capsys.readouterr().err
