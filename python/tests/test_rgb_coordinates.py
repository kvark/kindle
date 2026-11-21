import copy
import json
import sys
from types import SimpleNamespace

import pytest

import atari_vector
import kindle


@pytest.mark.parametrize('restore', [False, True])
def test_coordinates_reject_old_native_before_environment(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--rgb-coordinates']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(rgb_coordinates=True))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert 'requires native RGB coordinate support' in capsys.readouterr().err


@pytest.mark.parametrize('args,message', [
    (['--restore', 'unused'], 'training overrides'),
    (['--evaluate'], 'requires fresh training'),
    (['--encoder-checkpoint', 'unused'], 'requires fresh training'),
])
def test_coordinates_reject_incompatible_runs(monkeypatch, tmp_path, capsys, args, message):
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), '--rgb-coordinates', *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('side', [32, 48, 64])
@pytest.mark.parametrize('cdp', [False, True])
def test_coordinates_change_only_encoder_option(monkeypatch, tmp_path, side, cdp):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct, SUPPORTS_RGB_COORDINATES=True))
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
            '--image-size', str(side), *(['--cdp', '--cdp-centered'] if cdp else []),
            *(['--rgb-coordinates'] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, coordinates = configs
    assert not plain.get('rgb_coordinates', False)
    assert coordinates.pop('rgb_coordinates') is True
    assert coordinates == plain
    description = atari_vector.rgb_preprocessing(dict(plain, rgb_coordinates=True))
    assert description == atari_vector.rgb_preprocessing(plain) + (
        '; encoder appends fixed x/y pixel-center coordinates in [-0.5,0.5]')
