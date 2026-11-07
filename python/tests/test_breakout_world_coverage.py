from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import breakout_world_coverage as coverage


def streams():
    result=[]
    for _ in range(16):
        action=np.zeros(100,np.int32);action[[0,73]]=-1
        terminal=np.zeros(100,bool);terminal[70]=True
        source_wait=np.arange(100)<40
        result.append(dict(action=action,terminal=terminal,reward=np.zeros(100),source_wait=source_wait))
    return result


def test_all_waiting_commands_are_nonfire_and_ignore_pending_random_blocks():
    rng=np.random.default_rng(1)
    for lane in range(8):
        for tick in range(128):
            action,remaining,command=coverage.source_action(lane,tick,1,True,5,rng,10,17)
            assert action not in coverage.FIRE and remaining==10 and command==17


def test_real_teacher_manual_feedback_and_serve_launch_pairing():
    from kindle._breakout_guidance import FeedbackPixelPolicy
    teacher=FeedbackPixelPolicy();rng=np.random.default_rng(1)
    frame=np.zeros((210,160,3),np.uint8)
    for waiting in [True,False]:
        proposal=teacher.propose(frame)
        action,_,_=coverage.source_action(4,0,proposal,waiting,5,rng,0,0)
        teacher.feedback(int(action))
        assert action==(0 if waiting else 1) and teacher.pending is None


def test_pools_keep_terminal_context_and_exclude_fire_and_reset_waits():
    data=streams();data[8]['action'][20]=1;data[9]['action'][20]=-1
    waits,endings=coverage.sampling_pools(data)
    assert set(waits[:,0])==set(range(10,16))
    assert len(endings)==16*16
    for lane,start in endings:
        assert 16<=70-start<=31 and not (data[lane]['action'][start+1:71]==-1).any()
    data[0]['action'][60]=-1
    _,endings=coverage.sampling_pools(data)
    assert 0 not in endings[:,0]


def test_plan_is_deterministic_keeps_four_two_one_one_recipe_and_exposure_counts():
    data=streams();config=dict(seed=41001,batch_size=8,batch_length=32)
    plan,counts=coverage.coverage_plan(data,config,64)
    again,other=coverage.coverage_plan(data,config,64)
    np.testing.assert_array_equal(plan,again);assert counts==other
    assert (plan[:,:4,0]<8).all() and (plan[:,4:7,0]>=8).all()
    direct_terminal=0
    for row in plan:
        lane,start=row[6];assert data[lane]['source_wait'][start:start+32].all()
        lane,start=row[7];assert data[lane]['terminal'][start+16:start+32].any()
        direct_terminal+=sum(int(data[l]['terminal'][s:s+32].sum()) for l,s in row)
    assert counts['terminal_rows']==direct_terminal and direct_terminal>=64
    assert counts['prior_terminal_rows']==0


def test_sampling_rejects_missing_fitting_pools_and_wrong_native_geometry():
    data=streams();config=dict(seed=41001,batch_size=8,batch_length=32)
    with pytest.raises(ValueError):coverage.coverage_plan(data[:8],config,2)
    with pytest.raises(ValueError):coverage.coverage_plan(data,{**config,'batch_length':16},2)
    for stream in data:stream['terminal'][:]=False
    with pytest.raises(ValueError):coverage.coverage_plan(data,config,2)
