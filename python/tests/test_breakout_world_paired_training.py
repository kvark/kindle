from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
import breakout_world_paired_training as paired

CONFIG = dict(seed=41001, batch_size=8, batch_length=32)


def streams(short=4):
    result = []
    for _ in range(16):
        action = np.zeros(100, np.int32); action[[0, 73]] = -1
        terminal = np.zeros(100, bool); terminal[70] = True
        result.append(dict(action=action, terminal=terminal, reward=terminal.astype(float),
            source_wait=np.arange(100) < 40, rgb=np.zeros((100, 1, 1, 1), np.uint8)))
    for draw in range(2):
        for lane, length in enumerate([short, 13, 32, 32]):
            action = np.r_[np.full(32, 3), np.full(length, lane)]
            terminal = np.zeros(32+length, bool)
            if lane != 2: terminal[-1] = True
            result.append(dict(action=action, terminal=terminal, reward=terminal.astype(float),
                rgb=np.zeros((32+length, 1, 1, 1), np.uint8)))
    return result


GROUPS = np.arange(16, 24).reshape(2, 4)


@pytest.mark.parametrize('short', [1, 4, 16, 32])
def test_common_offsets_keep_short_terminal_and_never_invent_a_future(short):
    data = streams(short)
    plan, _ = paired.paired_plan(data, GROUPS, CONFIG, 512)
    again, _ = paired.paired_plan(data, GROUPS, CONFIG, 512)
    np.testing.assert_array_equal(plan, again)
    for step, row in enumerate(plan):
        assert any(np.array_equal(row[:4, 0], group) for group in GROUPS)
        assert np.unique(row[:4, 1]).size == 1
        limit = min(len(data[lane]['action'])-32 for lane in row[:4, 0])
        offset = row[0, 1]
        assert 0 <= offset <= limit
        if step % 2 == 0: assert offset == min(16, limit)
        assert (row[4:6, 0] < 8).all() and 8 <= row[6, 0] < 16
        lane, start = row[6]; assert data[lane]['source_wait'][start:start+32].all()
        lane, start = row[7]; assert data[lane]['terminal'][start+16:start+32].any()
        for lane, start in row:
            assert start+32 <= len(data[lane]['action'])
    terminal_lanes = {lane for lane in range(16, 24) if data[lane]['terminal'].any()}
    assert terminal_lanes.issubset(set(plan[:, 7, 0]))


def test_exposure_counts_match_actual_context_resets_targets_and_unique_arrivals():
    data = streams()
    plan, expected = paired.paired_plan(data, GROUPS, CONFIG, 512)
    actual = dict(terminal_rows=0, prior_terminal_rows=0, positive_reward_rows=0,
                  context_reset_rows=0, source_reset_rows=0, unique_source_arrivals=0)
    seen = set()
    for row in plan:
        batch = paired.corpus.training_windows([data[lane] for lane, _ in row], row[:, 1],
            32, allow_resets=True, retain_initial_targets=True)
        keep = paired.corpus.rollout_validity(batch['action'], 16)
        actual['terminal_rows'] += int(batch['terminal'].sum())
        actual['prior_terminal_rows'] += int((batch['terminal'][-16:] & keep).sum())
        actual['positive_reward_rows'] += int((batch['reward'] > 0).sum())
        actual['context_reset_rows'] += int((batch['action'] == -1).sum())
        for lane, start in row:
            actual['source_reset_rows'] += int((data[lane]['action'][start:start+32] == -1).sum())
            seen.update((int(lane), int(index)) for index in range(start, start+32))
    actual['unique_source_arrivals'] = len(seen)
    assert actual == expected
    assert expected['context_reset_rows'] > expected['source_reset_rows']


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'reset', 'terminal', 'geometry'])
def test_rejects_incomplete_groups_invalid_branches_and_wrong_native_shape(defect):
    data, groups, config = streams(), GROUPS.copy(), dict(CONFIG)
    if defect == 'missing': groups = groups[:1]
    if defect == 'duplicate': groups[1, 0] = groups[0, 0]
    if defect == 'reset': data[16]['action'][33] = -1
    if defect == 'terminal': data[16]['terminal'][33] = True
    if defect == 'geometry': config['batch_length'] = 16
    with pytest.raises(ValueError): paired.paired_plan(data, groups, config, 1)
