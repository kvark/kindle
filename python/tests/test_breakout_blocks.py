import hashlib
import random
import sys

import numpy as np
import pytest

import atari_vector
from kindle import _breakout_guidance as module
from kindle._vector_audit import audit
from test_breakout_corrections import corrective_events
from test_vector import write_log


def config(length=2, cap=7, probability=.5):
    return dict(kind=module.BLOCK_CORRECTIVE_GUIDANCE_KIND, probability=probability,
                seed=1009, max_actions=cap, block_length=length)


@pytest.mark.parametrize('value', [None, False, 0, -1, 1.5, 2**32])
def test_invalid_block_length(value):
    with pytest.raises(ValueError, match='block_length'):
        module.BreakoutGuidance(config(value), 2, 18)


def test_blocks_keep_lane_rng_feedback_and_teacher_history_through_native_actions():
    guide = module.BreakoutGuidance(config(), 3, 18)
    verifier = module.BreakoutGuidance(config(), 3, 18)
    rngs = [random.Random((1009 + lane) ^ module.GUIDANCE_SEED_XOR) for lane in range(3)]
    choices = [rng.random() < .5 for rng in rngs]
    ages = [0] * 3
    totals = [0] * 3
    ever = list(choices)
    policies = list(guide._policies)
    frame = np.zeros((210, 160, 3), np.uint8)
    guided_episodes = list(map(int, choices))
    blocks = [1] * 3
    guided_blocks = list(map(int, choices))
    for tick in range(15):
        if tick == 4:  # Reset exactly at a block boundary: no unused block draw.
            guide.reset([1]); verifier.reset([1])
            choices[1] = rngs[1].random() < .5
            ages[1] = 0; ever[1] = choices[1]
            guided_episodes[1] += choices[1]
            blocks[1] += 1; guided_blocks[1] += choices[1]
            assert guide._policies[1] is not policies[1]
            policies[1] = guide._policies[1]
        for lane in range(3):
            if ages[lane] and ages[lane] < 7 and ages[lane] % 2 == 0:
                choices[lane] = rngs[lane].random() < .5
                blocks[lane] += 1; guided_blocks[lane] += choices[lane]
                guided_episodes[lane] += choices[lane] and not ever[lane]
                ever[lane] |= choices[lane]
        expected = [1 if choice and age < 7 else None for choice, age in zip(choices, ages)]
        assert guide.actions([frame] * 3) == expected
        verifier.verify_actions(expected)
        assert guide.expert_labels == [1] * 3
        assert guide.frame_sha256 == [hashlib.sha256(frame.tobytes()).hexdigest()] * 3
        guide.feedback([action if action is not None else 17 for action in expected])
        assert guide._policies == policies
        assert [p.teacher.previous_command for p in policies] == [0 if a else -1 for a in expected]
        totals = [total + (a is not None) for total, a in zip(totals, expected)]
        ages = [age + 1 for age in ages]
        assert guide.overridden_actions == verifier.overridden_actions == totals
        assert guide.selected_episodes == verifier.selected_episodes == ever
        assert guide.episode_counts == verifier.episode_counts == dict(
            started_episodes=[1, 2 if tick >= 4 else 1, 1], guided_episodes=guided_episodes,
            started_blocks=blocks, guided_blocks=guided_blocks)
    assert guide.config == config()


@pytest.mark.parametrize('mutation', [None, 'override', 'mask', 'counter', 'cap', 'imitation'])
def test_block_audit_checks_schedule_and_all_state_labels(tmp_path, mutation):
    rows = corrective_events()
    spec = config(length=1, cap=2)
    rows[0]['exploration'] = spec
    guide = module.BreakoutGuidance(spec, 2, 18)
    frames = [np.zeros((210, 160, 3), np.uint8)] * 2
    for row in rows[1:]:
        if row['event'] == 'transition':
            row['action_overrides'] = guide.actions(frames)
            row['actions'] = [17 if a is None else a for a in row['action_overrides']]
            guide.feedback(row['actions'])
        elif row['event'] == 'reset':
            guide.reset(row['streams'])
        elif row['event'] in ('progress', 'run_end'):
            row['overridden_actions'] = guide.overridden_actions
            row['guidance_episodes'] = guide.episode_counts
    transition = next(row for row in rows if row['event'] == 'transition')
    if mutation == 'override':
        transition['action_overrides'][0] = None if transition['action_overrides'][0] else 1
    elif mutation == 'mask': transition['expert_action_mask'][0] = False
    elif mutation == 'counter': rows[-1]['guidance_episodes']['started_blocks'][0] += 1
    elif mutation == 'cap': rows[0]['exploration']['max_actions'] = 0
    elif mutation == 'imitation': del rows[0]['config']['expert_imitation']
    if mutation:
        with pytest.raises(ValueError): audit(write_log(tmp_path, rows))
    else:
        assert audit(write_log(tmp_path, rows))['actions'] == 6


@pytest.mark.parametrize('args,message', [
    (['--breakout-guidance-block-length', '0'], 'positive u32'),
    (['--breakout-guidance-block-length', str(2**32)], 'positive u32'),
    (['--breakout-guidance-block-length', '32'], 'require corrective'),
    (['--breakout-guidance-block-length', '32', '--breakout-corrective-labels'], 'require Breakout'),
])
def test_cli_rejects_invalid_blocks_before_environment(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', ['atari_vector', 'ALE/Breakout-v5', '--output', str(tmp_path/'run'), *args])
    with pytest.raises(SystemExit) as error: atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err
