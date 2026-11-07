"""Four-action guidance keeps full18 teacher state and strict log identities."""

import copy
import numpy as np
import pytest

import kindle._breakout_guidance as module
from kindle._vector_audit import audit
from test_breakout_corrections import config, corrective_events
from test_vector import write_log


@pytest.mark.parametrize('count,meanings', [
    (4, None), (18, module.MINIMAL_ACTION_MEANINGS), (True, module.MINIMAL_ACTION_MEANINGS),
    (4, ['NOOP', 'FIRE', 'LEFT', 'RIGHT']), (4, ['NOOP', 'FIRE', 'RIGHT', 'RIGHT']),
    (4, ['NOOP', 'FIRE', 'RIGHT']), (4, 'NOOP'), (4, 4),
])
def test_minimal_requires_explicit_exact_vocabulary(count, meanings):
    with pytest.raises(ValueError):
        module.BreakoutGuidance(config(), 2, count, action_meanings=meanings)


def test_minimal_feedback_rejects_entire_invalid_batch_before_state_change():
    guide = module.BreakoutGuidance(config(), 2, 4, action_meanings=module.MINIMAL_ACTION_MEANINGS)
    frames = [np.zeros((210, 160, 3), np.uint8)] * 2
    assert guide.actions(frames) == [1, 1]
    for bad in (-1, 4, 17, True, 1.0, None):
        with pytest.raises(ValueError):
            guide.feedback([2, bad])
        assert [p.pending for p in guide._policies] == [1, 1]
        assert [p.policy.rejected for p in guide._policies] == [0, 0]
    guide.feedback([2, 3])
    assert [p.policy.teacher.previous_command for p in guide._policies] == [1, -1]
    assert guide.actions(frames) == [None, None]
    assert guide.expert_labels == [1, 1]


def minimal_events():
    rows = copy.deepcopy(corrective_events())
    header = rows[0]
    header.update(atari_protocol='published-minimal', full_action_space=False,
                  action_meanings=list(module.MINIMAL_ACTION_MEANINGS))
    header['config']['action_count'] = 4
    header['config']['expert_imitation']['action_weights'] = [.05, 1., 1., 1.]
    for row in rows:
        if row['event'] == 'transition':
            row['actions'] = [3 if a == 17 else module.MINIMAL_TO_FULL.index(a) for a in row['actions']]
    return rows


@pytest.mark.parametrize('mutation', [None, 'protocol', 'flag', 'order', 'count', 'label', 'override', 'game'])
def test_minimal_audit_rejects_mixed_vocabularies(tmp_path, mutation):
    rows = minimal_events()
    header = rows[0]
    transition = next(row for row in rows if row['event'] == 'transition')
    if mutation == 'protocol': header['atari_protocol'] = 'published'
    elif mutation == 'flag': header['full_action_space'] = True
    elif mutation == 'order': header['action_meanings'][2:] = ['LEFT', 'RIGHT']
    elif mutation == 'count': header['config']['action_count'] = 18
    elif mutation == 'label': transition['expert_labels'][0] = 4
    elif mutation == 'override': transition['action_overrides'][0] = 4
    elif mutation == 'game': header['environment'] = 'ALE/Qbert-v5'
    path = write_log(tmp_path, rows)
    if mutation is None:
        assert audit(path)['actions'] == 6
    else:
        with pytest.raises(ValueError):
            audit(path)


@pytest.mark.parametrize('corrective', [False, True])
def test_full_and_minimal_guidance_match_on_live_pixels(corrective):
    import ale_py
    import gymnasium as gym
    import atari

    gym.register_envs(ale_py)
    spec = config(probability=1, cap=3)
    if not corrective:
        spec['kind'] = module.GUIDANCE_KIND
    full = module.BreakoutGuidance(spec, 1, 18)
    minimal = module.BreakoutGuidance(spec, 1, 4, action_meanings=module.MINIMAL_ACTION_MEANINGS)
    environments = [atari.DreamerAtariPreprocessing(gym.make('ALE/Breakout-v5', frameskip=1,
        repeat_action_probability=.25, full_action_space=flag), noop_max=0,
        max_episode_frames=100000, screen_size=None) for flag in (True, False)]
    try:
        frames = [env.reset(seed=1009)[0] for env in environments]
        for tick in range(128):
            assert np.array_equal(*frames)
            first, second = full.actions([frames[0]])[0], minimal.actions([frames[1]])[0]
            assert first == (None if second is None else module.MINIMAL_TO_FULL[second])
            assert full.frame_sha256 == minimal.frame_sha256
            if corrective:
                assert full.expert_labels == [module.MINIMAL_TO_FULL[minimal.expert_labels[0]]]
            action = second if second is not None else tick % 4
            if corrective:
                full.feedback([module.MINIMAL_TO_FULL[action]])
                minimal.feedback([action])
                assert full._policies[0].teacher.last_debug == minimal._policies[0].policy.teacher.last_debug
                assert full._policies[0].teacher.previous_command == minimal._policies[0].policy.teacher.previous_command
            results = [environments[0].step(module.MINIMAL_TO_FULL[action]), environments[1].step(action)]
            assert results[0][1:4] == results[1][1:4]
            assert environments[0].executed_action_frames == environments[1].executed_action_frames
            frames = [result[0] for result in results]
            if results[0][2] or results[0][3] or tick == 63:
                frames = [env.reset()[0] for env in environments]
                full.reset([0]); minimal.reset([0])
            assert full.overridden_actions == minimal.overridden_actions
            assert full.episode_counts == minimal.episode_counts
    finally:
        for env in environments:
            env.close()
