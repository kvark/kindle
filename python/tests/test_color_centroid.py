"""CPU entry and accounting checks for optional native centroid supervision."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import kindle
from kindle._vector_audit import audit

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import atari_vector
from test_vector import fixture_events, write_log


SETTINGS = dict(scale=100.0, channel=0, background_scale=1.5,
                region=[8/160, 90/210, 152/160, 185/210])


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '1e300', '1e-300'])
def test_invalid_scale_fails_before_environment(monkeypatch, tmp_path, capsys, value):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment created'))
    output = tmp_path/'absent.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'ALE/Breakout-v5', '--cdp',
        '--breakout-centroid-scale', value, '--output', str(output)])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert 'positive finite F32' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('extra', [[], ['--cdp', '--evaluate'], ['--cdp', '--encoder-checkpoint', 'unused'],
    ['--cdp', '--observation-size', '64']])
def test_centroid_requires_fresh_cdp_native_frames(monkeypatch, tmp_path, capsys, extra):
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'ALE/Breakout-v5',
        '--breakout-centroid-scale', '100', '--output', str(tmp_path/'absent.jsonl'), *extra])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert 'requires fresh Breakout CDP' in capsys.readouterr().err


def test_centroid_rejects_other_game_restore_and_old_native(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment created'))
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    for game, extra, message in [('ALE/Pong-v5', [], 'requires fresh Breakout'),
            ('ALE/Breakout-v5', ['--restore', 'unused'], 'training overrides'),
            ('ALE/Breakout-v5', [], 'requires native color-centroid support')]:
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', game, '--cdp',
            '--breakout-centroid-scale=100', '--output', str(tmp_path/'absent.jsonl'), *extra])
        with pytest.raises(SystemExit):
            atari_vector.main()
        assert message in capsys.readouterr().err


@pytest.mark.parametrize('enabled', [False, True])
def test_exact_centroid_config_reaches_native_boundary(monkeypatch, tmp_path, enabled):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    def construct(streams, config):
        if enabled:
            assert config['color_centroid_supervision'] == SETTINGS
        else:
            assert config.get('color_centroid_supervision') is None
        assert config['loss_scales']['future_prediction'] == 500
        assert config['observation_kind'] == 'rgb64'
        raise RuntimeError('checked without GPU')
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct,
        SUPPORTS_COLOR_CENTROID_SUPERVISION=True))
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'ALE/Breakout-v5', '--cdp',
        '--output', str(tmp_path/'checked.jsonl'), *(['--breakout-centroid-scale', '100'] if enabled else [])])
    with pytest.raises(RuntimeError, match='checked without GPU'):
        atari_vector.main()


def centroid_events():
    events = fixture_events()
    events[0]['config'].update(observation_kind='rgb64', color_centroid_supervision=copy.deepcopy(SETTINGS))
    for e in events:
        if e['event'] == 'learner':
            e['report']['world']['color_centroid_supervision'] = dict(loss=.01, valid_fraction=.5)
    return events


def test_accounting_accepts_optional_metrics_and_exact_absent_rows(tmp_path):
    events = centroid_events()
    assert audit(write_log(tmp_path, events))['updates'] == 3
    for e in events:
        if e['event'] == 'learner':
            e['report']['world']['color_centroid_supervision'] = dict(loss=0, valid_fraction=0)
    assert audit(write_log(tmp_path, events))['updates'] == 3


@pytest.mark.parametrize('field,value', [('scale', 0), ('channel', 3), ('channel', True),
    ('background_scale', -1), ('region', [0, 0, .001, .001]), ('region', [-.1, 0, 1, 1])])
def test_accounting_rejects_invalid_centroid_config(tmp_path, field, value):
    events = centroid_events()
    events[0]['config']['color_centroid_supervision'][field] = value
    with pytest.raises(ValueError):
        audit(write_log(tmp_path, events))


@pytest.mark.parametrize('value', [None, dict(loss=-1, valid_fraction=.5),
    dict(loss=.1, valid_fraction=0), dict(loss=0, valid_fraction=1.1), dict(loss=.1)])
def test_accounting_rejects_missing_or_invalid_metrics(tmp_path, value):
    events = centroid_events()
    first = next(e for e in events if e['event'] == 'learner')
    if value is None:
        del first['report']['world']['color_centroid_supervision']
    else:
        first['report']['world']['color_centroid_supervision'] = value
    with pytest.raises(ValueError):
        audit(write_log(tmp_path, events))
