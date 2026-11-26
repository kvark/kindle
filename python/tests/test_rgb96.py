import copy
import json
import sys
from types import SimpleNamespace

import pytest
import atari_vector
import kindle
from kindle._vector_audit import audit
from test_vector import fixture_events, write_log


@pytest.mark.parametrize('restore', [False, True])
def test_rgb96_rejects_old_native_before_environment(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--image-size', '96']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(observation_kind='rgb96'))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert 'requires native RGB96 support' in capsys.readouterr().err


@pytest.mark.parametrize('args,message', [
    (['--restore', 'unused'], 'training overrides'),
    (['--encoder-checkpoint', 'unused'], 'image-size requires the learned'),
    (['--observation-size', '64'], 'native'),
])
def test_rgb96_rejects_geometry_overrides_and_double_resize(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(SUPPORTS_RGB96=True))
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), '--image-size', '96', *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('cdp', [False, True])
def test_rgb96_changes_only_geometry_and_preserves_default(monkeypatch, tmp_path, cdp):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct, SUPPORTS_RGB96=True))
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
            *(['--cdp', '--cdp-centered'] if cdp else []), *(['--image-size', '96'] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, larger = configs
    assert plain.pop('observation_kind') == 'rgb64'
    assert larger.pop('observation_kind') == 'rgb96'
    assert plain == larger
    assert atari_vector.rgb_preprocessing(dict(observation_kind='rgb96')) == (
        'single GPU Pillow-equivalent antialiased bilinear resize to 96x96, per-axis RGB8 rounding; CHW /255-0.5')


def test_rgb96_restore_uses_checkpoint_without_fresh_geometry(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(observation_kind='rgb96'))))
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    monkeypatch.setattr(atari_vector, 'checkpoint_identity', lambda path: dict(path=str(path)))

    def restore(path, streams):
        assert path == str(checkpoint) and streams == 8
        assert json.loads((checkpoint / 'metadata.json').read_text())['config']['observation_kind'] == 'rgb96'
        raise RuntimeError('restore inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(restore_rgb=restore, SUPPORTS_RGB96=True))
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'restore.jsonl'),
        '--evaluate', '--restore', str(checkpoint), '--observation-size', 'native', '--num-envs', '8'])
    with pytest.raises(RuntimeError, match='restore inspected before GPU'):
        atari_vector.main()


@pytest.mark.parametrize('kind', ['rgb96', 'rgb64', 'rgb95', 'RGB96', None, 96])
def test_rgb96_audit_accepts_declared_geometry_and_rejects_unknown(tmp_path, kind):
    rows = fixture_events()
    rows[0]['config']['observation_kind'] = kind
    if kind in ('rgb96', 'rgb64'):
        assert audit(write_log(tmp_path, rows))['updates'] == 3
    else:
        with pytest.raises(ValueError, match='unknown observation geometry'):
            audit(write_log(tmp_path, rows))
