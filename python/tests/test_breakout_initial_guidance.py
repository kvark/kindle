"""Initial episode guidance preserves per-lane draws and releases at the normal cap."""
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import atari_vector
import kindle
from kindle._breakout_guidance import (
    BreakoutGuidance, CORRECTIVE_GUIDANCE_KIND, GUIDANCE_SEED_XOR,
    INITIAL_CORRECTIVE_GUIDANCE_KIND,
)


def config():
    return dict(kind=INITIAL_CORRECTIVE_GUIDANCE_KIND, probability=.2, seed=1009,
                max_actions=2, initial_episodes=2)


@pytest.mark.parametrize('value', [None, 0, -1, True, 1.5, 2**32])
def test_invalid_initial_episode_count(value):
    with pytest.raises(ValueError, match='positive u32'):
        BreakoutGuidance(dict(config(), initial_episodes=value), 2, 18)


@pytest.mark.parametrize('change', [dict(block_length=32), dict(extra=1),
                                  dict(kind=CORRECTIVE_GUIDANCE_KIND)])
def test_unrecognized_and_mixed_schedule_fields_rejected(change):
    with pytest.raises(ValueError, match='configuration fields'):
        BreakoutGuidance(dict(config(), **change), 2, 18)


def test_async_episode_boundary_preserves_draws_counts_and_other_lanes():
    guide = BreakoutGuidance(config(), 3, 18)
    rngs = [random.Random((1009 + lane) ^ GUIDANCE_SEED_XOR) for lane in range(3)]
    episodes, selected_counts = [0] * 3, [0] * 3
    selected = [False] * 3
    for resets in ([0, 1, 2], [0], [0], [2], [1, 2], [2], [1], [0, 1], [2]):
        if any(episodes):
            before = [(guide._policies[lane], guide._rngs[lane].getstate(),
                       guide._episode_actions[lane]) for lane in range(3)]
            guide.reset(resets)
            for lane in set(range(3)) - set(resets):
                assert before[lane] == (guide._policies[lane], guide._rngs[lane].getstate(),
                                        guide._episode_actions[lane])
        for lane in resets:
            drawn = rngs[lane].random() < .2
            episodes[lane] += 1
            selected[lane] = episodes[lane] <= 2 or drawn
            selected_counts[lane] += selected[lane]
        assert guide.selected_episodes == selected
        assert guide.episode_counts == dict(started_episodes=episodes, guided_episodes=selected_counts)
        assert [rng.getstate() for rng in guide._rngs] == [rng.getstate() for rng in rngs]
        guide.verify_actions([1 if active else None for active in guide.active_guidance])
    assert min(episodes) > 2 and not all(selected)


def test_cap_releases_forced_episode_but_all_state_labels_and_feedback_continue():
    guide = BreakoutGuidance(config(), 2, 18)
    frames = [np.zeros((210, 160, 3), dtype=np.uint8)] * 2
    for expected in ([1, 1], [1, 1], [None, None]):
        assert guide.actions(frames) == expected
        assert guide.expert_labels == [1, 1]
        assert all(guide.frame_sha256)
        guide.feedback([1 if action is not None else 3 for action in expected])
    guide.reset([0])
    assert guide.actions(frames) == [1, None]
    guide.feedback([1, 4])
    assert guide.overridden_actions == [3, 2]
    assert guide.episode_counts == dict(started_episodes=[2, 1], guided_episodes=[2, 1])


@pytest.mark.parametrize('mutation', [None, 'count', 'override', 'kind'])
def test_replay_reader_reconstructs_initial_schedule(monkeypatch, tmp_path, mutation):
    import test_breakout_corrections as fixtures
    from test_vector import write_log
    from kindle._vector_audit import audit
    monkeypatch.setattr(fixtures, 'config', lambda: dict(config(), max_actions=1))
    rows = fixtures.corrective_events()
    if mutation == 'count':
        rows[-1]['guidance_episodes']['guided_episodes'][0] += 1
    elif mutation == 'override':
        next(row for row in rows if row['event'] == 'transition')['action_overrides'][0] = None
    elif mutation == 'kind':
        rows[0]['exploration']['kind'] = CORRECTIVE_GUIDANCE_KIND
    if mutation is None:
        result = audit(write_log(tmp_path, rows))
        assert result['actions'] == 6 and result['exploration_schedule_verified']
    else:
        with pytest.raises(ValueError):
            audit(write_log(tmp_path, rows))


def command(tmp_path, extra):
    return ['atari_vector', 'ALE/Breakout-v5', '--output', str(tmp_path / 'run.jsonl'),
            '--breakout-guidance-probability', '.2', '--breakout-guidance-max-actions', '1920',
            '--guidance-imitation-scale', '100', '--breakout-corrective-labels',
            '--breakout-guidance-initial-episodes', '2', *extra]


@pytest.mark.parametrize('extra,message', [
    (['--breakout-guidance-initial-episodes', '0'], 'positive u32'),
    (['--breakout-guidance-initial-episodes', str(2**32)], 'positive u32'),
    (['--breakout-guidance-block-length', '32'], 'cannot combine'),
    (['--evaluate'], 'frozen evaluation'),
    (['--restore', 'unused'], 'training overrides'),
    (['--breakout-guidance-probability', '0'], 'requires positive'),
])
def test_cli_rejects_invalid_combinations_before_construction(monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment created'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, extra))
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err


def test_cli_requires_corrective_labels(monkeypatch, tmp_path, capsys):
    args = command(tmp_path, [])
    args.remove('--breakout-corrective-labels')
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment created'))
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert 'requires corrective labels' in capsys.readouterr().err


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_preserves_native_config_and_passes_exact_schedule(monkeypatch, tmp_path, enabled):
    env = SimpleNamespace(action_space=SimpleNamespace(n=18),
        action_meanings=['NOOP', 'FIRE'] + [str(i) for i in range(2, 18)],
        reset=lambda **kw: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: env)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **kw: env)
    seen = []
    # The runner uses this name both as constructor and in isinstance.
    class CapturedGuidance(BreakoutGuidance):
        def __init__(self, spec, *args, **kwargs):
            seen.append(spec)
            super().__init__(spec, *args, **kwargs)
    monkeypatch.setattr(atari_vector, 'BreakoutGuidance', CapturedGuidance)
    def construct(streams, native_config):
        assert 'initial_episodes' not in native_config
        assert native_config['expert_imitation']['scale'] == 100
        assert seen == [dict(kind=INITIAL_CORRECTIVE_GUIDANCE_KIND if enabled else CORRECTIVE_GUIDANCE_KIND,
            probability=.2, seed=0, max_actions=1920, **(dict(initial_episodes=2) if enabled else {}))]
        raise RuntimeError('checked native boundary')
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace(learned_rgb=construct,
        act=lambda greedy=False, action_overrides=None, expert_actions=None, expert_labels=None: None))
    args = command(tmp_path, [])
    if not enabled:
        position = args.index('--breakout-guidance-initial-episodes')
        del args[position:position + 2]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(RuntimeError, match='checked native boundary'):
        atari_vector.main()
