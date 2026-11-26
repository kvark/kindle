import copy
import json
import sys
from types import SimpleNamespace

import pytest
import atari_vector
import kindle


@pytest.mark.parametrize('restore', [False, True])
def test_stride2_rejects_old_native_before_environment(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--rgb-stride2']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(rgb_stride2=True))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert 'requires native RGB stride-two support' in capsys.readouterr().err


@pytest.mark.parametrize('args,message', [
    (['--restore', 'unused'], 'training overrides'),
    (['--evaluate'], 'requires fresh training'),
    (['--encoder-checkpoint', 'unused'], 'requires fresh training'),
])
def test_stride2_rejects_incompatible_runs(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), '--rgb-stride2', *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('side', [32, 48, 64, 96])
@pytest.mark.parametrize('cdp', [False, True])
def test_stride2_changes_only_encoder_option(monkeypatch, tmp_path, side, cdp):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct, SUPPORTS_RGB_STRIDE2=True, SUPPORTS_RGB96=True))
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
            '--image-size', str(side), *(['--cdp', '--cdp-centered'] if cdp else []),
            *(['--rgb-stride2'] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, strided = configs
    assert not plain.get('rgb_stride2', False)
    assert strided.pop('rgb_stride2') is True
    assert strided == plain
    assert atari_vector.rgb_preprocessing(dict(plain, rgb_stride2=True)) == (
        atari_vector.rgb_preprocessing(plain) + '; encoder uses stride-two convolutions without max-pooling')


def test_stride2_restore_preserves_checkpoint_setting(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(observation_kind='rgb64', rgb_stride2=True))))
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    monkeypatch.setattr(atari_vector, 'checkpoint_identity', lambda path: dict(path=str(path)))

    def restore(path, streams):
        assert path == str(checkpoint) and streams == 8
        assert json.loads((checkpoint / 'metadata.json').read_text())['config']['rgb_stride2'] is True
        raise RuntimeError('restore inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(restore_rgb=restore, SUPPORTS_RGB_STRIDE2=True))
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'restore.jsonl'),
        '--evaluate', '--restore', str(checkpoint), '--observation-size', 'native', '--num-envs', '8'])
    with pytest.raises(RuntimeError, match='restore inspected before GPU'):
        atari_vector.main()
