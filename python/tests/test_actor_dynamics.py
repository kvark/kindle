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
    (['--actor-dynamics-scale', '0'], 'finite and positive'),
    (['--actor-dynamics-scale', '-1'], 'finite and positive'),
    (['--actor-dynamics-scale', 'nan'], 'finite and positive'),
    (['--actor-dynamics-scale', 'inf'], 'finite and positive'),
    (['--actor-dynamics-scale', '1e-60'], 'finite and positive'),
    (['--actor-dynamics-scale', '.5', '--restore', 'unused'], 'requires fresh training'),
    (['--actor-dynamics-scale', '.5', '--evaluate'], 'requires fresh training'),
])
def test_dynamics_rejects_invalid_cli_before_construction(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('restore', [False, True])
def test_dynamics_rejects_old_native(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--actor-dynamics-scale', '.5']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(actor_dynamics_scale=.5))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'never.jsonl'), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert 'requires native actor dynamics support' in capsys.readouterr().err


def test_dynamics_only_adds_declared_config(monkeypatch, tmp_path):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct, SUPPORTS_ACTOR_DYNAMICS=True))
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
            '--cdp', '--cdp-centered', *(['--actor-dynamics-scale', '.5'] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, dynamics = configs
    assert 'actor_dynamics_scale' not in plain
    assert dynamics.pop('actor_dynamics_scale') == .5
    assert dynamics == plain


@pytest.mark.parametrize('mutation', [None, 'missing', 'unconfigured', 'nan', 'boolean', 'bad_scale'])
def test_dynamics_report_matches_config(tmp_path, mutation):
    rows = corrective_events()
    rows[0]['config']['actor_dynamics_scale'] = .5
    reports = [event['report'] for event in rows if event['event'] == 'learner']
    for report in reports:
        report['behavior'] = dict(dynamics_actor_loss=-.4)
    if mutation == 'missing': reports[0]['behavior'].clear()
    elif mutation == 'unconfigured': del rows[0]['config']['actor_dynamics_scale']
    elif mutation == 'nan': reports[0]['behavior']['dynamics_actor_loss'] = float('nan')
    elif mutation == 'boolean': reports[0]['behavior']['dynamics_actor_loss'] = True
    elif mutation == 'bad_scale': rows[0]['config']['actor_dynamics_scale'] = -1
    if mutation is None:
        assert audit(write_log(tmp_path, rows))['updates'] == 3
    else:
        with pytest.raises(ValueError):
            audit(write_log(tmp_path, rows))
