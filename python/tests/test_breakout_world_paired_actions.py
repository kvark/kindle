from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
import breakout_world_paired_actions as paired


def prefix():
    result = {key: np.zeros(32) for key in paired.FIELDS}
    result.update(action=np.full(32, 3, np.int32), terminal=np.zeros(32, bool),
                  frames=4*np.arange(100, 132), episode=np.full(32, 2))
    return result


def future(length):
    result = {key: np.zeros(length) for key in paired.FIELDS}
    result.update(action=np.full(length, 4, np.int32), terminal=np.zeros(length, bool),
                  frames=4*np.arange(132, 132+length), episode=np.full(length, 2))
    result['terminal'][-1] = True
    return result


def test_plans_keep_action_meanings_and_reverse_at_fifth_arrival():
    plans = dict(paired.plans('freeflight'))
    assert plans['noop'].tolist() == [0]*32
    assert plans['left'].tolist() == [4]*32
    assert plans['right'].tolist() == [3]*32
    assert plans['reverse'].tolist() == [3]*4+[4]*28
    assert {name: values.tolist() for name, values in paired.plans('serve')} == {
        'noop': [0]*32, 'fire': [1]*32, 'leftfire': [12]*32, 'rightfire': [11]*32}


@pytest.mark.parametrize('length', [1, 17, 32])
def test_short_terminal_branch_keeps_actual_prefix_actions_and_terminal(length):
    before, after = prefix(), future(length)
    result = paired.combine(before, after)
    for key in paired.FIELDS:
        np.testing.assert_array_equal(result[key][:32], before[key])
        np.testing.assert_array_equal(result[key][32:], after[key])
    assert len(result['action']) == 32+length and result['action'][0] == 3
    assert result['terminal'].sum() == 1 and result['terminal'][-1]


@pytest.mark.parametrize('defect', ['reset', 'post_terminal', 'frame', 'episode'])
def test_rejects_invalid_branch_alignment(defect):
    before, after = prefix(), future(3)
    if defect == 'reset': after['action'][0] = -1
    if defect == 'post_terminal': after['terminal'][0] = True
    if defect == 'frame': after['frames'][0] = before['frames'][-1]
    if defect == 'episode': after['episode'][0] += 1
    with pytest.raises(AssertionError): paired.combine(before, after)


def test_native_camera_uses_one_batched_resize_and_preserves_uint8():
    class Camera:
        def preprocess(self, frames):
            assert len(frames) == 8
            return np.broadcast_to(np.arange(8, dtype=np.float32)[:, None, None, None]/255-.5,
                                   (8, 3, 64, 64)).copy().tobytes()
    result = paired.resize(Camera(), [object() for _ in range(8)])
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[:, 0, 0, 0], np.arange(8))
