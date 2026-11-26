import copy
import json
import sys
from types import SimpleNamespace

import pytest
import atari_vector
import kindle


@pytest.mark.parametrize('args,message', [
    (['--latent-shape', '0', '8'], 'variables > 0'),
    (['--latent-shape', '-1', '8'], 'variables > 0'),
    (['--latent-shape', '64', '1'], 'classes > 1'),
    (['--latent-shape', '64', '0'], 'classes > 1'),
    (['--latent-shape', str(2**64), '8'], 'unsigned 64-bit'),
    (['--latent-shape', '64', str(2**64)], 'unsigned 64-bit'),
    (['--latent-shape', '64', '8'], 'requires centered CDP'),
    (['--latent-shape', '64', '8', '--cdp'], 'requires centered CDP'),
    (['--latent-shape', '64', '8', '--restore', 'unused'], 'requires fresh training'),
    (['--latent-shape', '64', '8', '--evaluate'], 'requires fresh training'),
])
def test_invalid_shape_stops_before_construction(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('restore', [False, True])
def test_shape_rejects_old_native(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = ['--cdp', '--cdp-centered', '--latent-shape', '64', '8']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(latent_shape=dict(variables=64, classes=8)))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'never.jsonl'), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert 'requires native latent shape support' in capsys.readouterr().err


@pytest.mark.parametrize('shape', [(32, 16), (64, 8)])
def test_shape_only_changes_declared_config(monkeypatch, tmp_path, shape):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18), reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct, SUPPORTS_LATENT_SHAPE=True))
    for enabled in [False, True]:
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
                                        '--cdp', '--cdp-centered',
                                        *(['--latent-shape', *map(str, shape)] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, grouped = configs
    assert 'latent_shape' not in plain
    assert grouped.pop('latent_shape') == dict(variables=shape[0], classes=shape[1])
    assert plain == grouped
