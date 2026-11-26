"""Check the CLI configuration boundary without constructing a learner."""
import sys
from types import SimpleNamespace

import pytest
import kindle
import atari_vector


def command(tmp_path, extra):
    return ['atari_vector.py', 'ALE/Breakout-v5', '--output', str(tmp_path/'run.jsonl'),
            '--breakout-guidance-probability', '.2', '--breakout-guidance-max-actions', '1920',
            *extra]


@pytest.mark.parametrize('extra,message', [
    (['--guidance-fire-weight', '4'], 'requires imitation'),
    *[(['--guidance-imitation-scale', '100', '--guidance-fire-weight', value], 'positive finite F32')
      for value in ('0', '-1', 'nan', 'inf', '1e300', '1e-300', '1e38')],
    (['--guidance-imitation-scale', '100', '--guidance-fire-weight', '4', '--evaluate'], 'frozen evaluation'),
    (['--guidance-imitation-scale', '100', '--guidance-fire-weight=4', '--restore', 'unused'], 'training overrides'),
])
def test_invalid_weight_or_frozen_override_fails_before_environment(monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment created'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, extra))
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path/'run.jsonl').exists()


@pytest.mark.parametrize('fire_weight', [None, 1., 4.])
@pytest.mark.parametrize('world_gradient', [False, True])
def test_exact_fire_weight_and_legacy_default_reach_native_boundary(monkeypatch, tmp_path, fire_weight, world_gradient):
    # Deliberately move FIRE: the adapter must use meanings, not a guessed index.
    meanings = ['NOOP'] + [str(i) for i in range(1, 18)]
    meanings[7] = 'FIRE'
    env = SimpleNamespace(action_space=SimpleNamespace(n=18), action_meanings=meanings,
                          reset=lambda **kw: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **kw: env)
    def construct(streams, config):
        weights = [.05] + [1.] * 17
        weights[7] = 1. if fire_weight is None else fire_weight
        expected = dict(scale=100., action_weights=weights)
        if world_gradient:
            expected['world_gradient'] = True
        assert config['expert_imitation'] == expected
        raise RuntimeError('checked native boundary')
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct,
        SUPPORTS_EXPERT_WORLD_GRADIENT=True,
        act=lambda greedy=False, action_overrides=None, expert_actions=None: None))
    extra = ['--guidance-imitation-scale', '100', '--guidance-noop-weight', '.05']
    if fire_weight is not None:
        extra += ['--guidance-fire-weight', str(fire_weight)]
    if world_gradient:
        extra += ['--guidance-world-gradient']
    monkeypatch.setattr(sys, 'argv', command(tmp_path, extra))
    with pytest.raises(RuntimeError, match='checked native boundary'):
        atari_vector.main()
