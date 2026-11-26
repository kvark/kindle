import copy
import json
import sys
from types import SimpleNamespace

import pytest
import atari_vector
import kindle


VALID = ['ALE/Breakout-v5', '--atari-protocol', 'published', '--cdp', '--cdp-centered',
         '--guidance-imitation-scale', '100', '--guidance-world-gradient',
         '--breakout-guidance-probability', '.2', '--breakout-guidance-max-actions', '1920',
         '--breakout-corrective-labels']


@pytest.mark.parametrize('args,message', [
    (['--actor-residual'], 'requires centered CDP'),
    (VALID + ['--actor-residual', '--actor-dynamics-scale', '.5'], 'without other actor losses'),
    (VALID + ['--actor-residual', '--actor-critic-gradient'], 'without other actor losses'),
    (VALID + ['--actor-residual', '--guidance-imitation-decay', '1', '2'], 'constant world expert'),
    (VALID + ['--actor-residual', '--restore', 'unused'], 'requires fresh training'),
    (VALID + ['--actor-residual', '--evaluate'], 'requires fresh training'),
])
def test_residual_invalid_scope_stops_before_construction(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    output = tmp_path / 'never.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


class Stub:
    SUPPORTS_EXPERT_WORLD_GRADIENT = True

    def act(self, action_overrides=None, *, expert_actions=None, expert_labels=None):
        raise AssertionError('native inference forbidden')


@pytest.mark.parametrize('restore', [False, True])
def test_residual_rejects_old_native(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, 'VectorAgent', Stub)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('environment must not start'))
    args = VALID + ['--actor-residual']
    if restore:
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(config=dict(actor_residual=True))))
        args = ['--evaluate', '--restore', str(checkpoint), '--observation-size', 'native']
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / 'never.jsonl'), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert 'requires native residual actor support' in capsys.readouterr().err


def test_residual_changes_only_declared_config(monkeypatch, tmp_path):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  action_meanings=['NOOP'] + [str(i) for i in range(1, 18)],
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
    configs = []

    class Current(Stub):
        SUPPORTS_ACTOR_RESIDUAL = True

        @staticmethod
        def learned_rgb(streams, config):
            configs.append(copy.deepcopy(config))
            raise RuntimeError('config inspected before GPU')

    monkeypatch.setattr(kindle, 'VectorAgent', Current)
    for enabled in [False, True]:
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(tmp_path / f'{enabled}.jsonl'),
                                        *VALID, *(['--actor-residual'] if enabled else [])])
        with pytest.raises(RuntimeError, match='config inspected before GPU'):
            atari_vector.main()
    plain, residual = configs
    assert 'actor_residual' not in plain
    assert residual.pop('actor_residual') is True
    assert residual == plain
