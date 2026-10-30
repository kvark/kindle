import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

import kindle._breakout_guidance as module
import kindle._pixel_guidance as schedule_module
from kindle._breakout_guidance import BreakoutGuidance, GUIDANCE_KIND, GUIDANCE_SEED_XOR
from kindle._exploration import EXPLORATION_PROTOCOL
from kindle._vector_audit import audit
from test_vector import version_two_events, write_log
import atari_vector
import replay_qbert_guidance


def config(probability=1, cap=2):
    return dict(kind=GUIDANCE_KIND, probability=probability, seed=1009, max_actions=cap)


@pytest.mark.parametrize('field,value', [('probability',0),('probability',True),('probability',float('nan')),
    ('probability',1.1),('seed',-1),('seed',True),('seed',2**64),('kind','episode-qbert-pixels-v1'),
    ('max_actions',None),('max_actions',0),('max_actions',True),('max_actions',2**32)])
def test_invalid_config(field,value):
    spec=config();spec[field]=value
    with pytest.raises(ValueError):BreakoutGuidance(spec,8,18)


def test_cap_release_and_reset_keep_independent_schedules_and_no_pixel_access():
    class Poison:
        def __getattribute__(self,name):raise AssertionError('released pixels inspected')
    frame=np.zeros((210,160,3),dtype=np.uint8)
    guide=BreakoutGuidance(config(),2,18)
    assert guide.actions([frame,frame])==[1,1]
    assert guide.actions([frame,frame])==[1,1]
    assert guide.actions([Poison(),Poison()])==[None,None]
    assert guide.frame_sha256==[None,None]
    previous=guide._policies[1]
    guide.reset([0])
    assert guide._policies[1] is previous
    assert guide.actions([frame,Poison()])==[1,None]
    assert guide.overridden_actions==[3,2]
    assert guide.episode_counts==dict(started_episodes=[2,1],guided_episodes=[2,1])


def test_independent_bernoulli_draws_and_action_vocabulary():
    guide=BreakoutGuidance(config(.3,3),3,18)
    rngs=[random.Random((1009+i)^GUIDANCE_SEED_XOR) for i in range(3)]
    expected=[rng.random()<.3 for rng in rngs]
    for i in range(100):
        assert guide.selected_episodes==expected
        lane=i%3;guide.reset([lane]);expected[lane]=rngs[lane].random()<.3
    all_guided=BreakoutGuidance(config(),1,18)
    for bad in [2,5,True,18,None]:
        with pytest.raises(ValueError):all_guided.verify_actions([bad])
    with pytest.raises(ValueError):BreakoutGuidance(config(),2,4)


def events():
    rows=version_two_events();spec=config()
    rows[0].update(protocol=EXPLORATION_PROTOCOL,seed=1009,environment='ALE/Breakout-v5',
        atari_protocol='published',full_action_space=True,observation_size='native',
        observation_shape=[210,160,3],sticky_actions=.25,noop_max=0,action_repeat=4,
        exploration=spec,exploration_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        guidance_schedule_sha256=hashlib.sha256(Path(schedule_module.__file__).read_bytes()).hexdigest())
    rows[0]['config'].update(seed=1009,action_count=18,expert_imitation=dict(scale=1.,action_weights=[.05]+[1.]*17))
    guide=BreakoutGuidance(spec,2,18);frames=[np.zeros((210,160,3),dtype=np.uint8)]*2
    for row in rows[1:]:
        if row['event']=='transition':
            row['action_overrides']=guide.actions(frames)
            row['actions']=[2 if a is None else a for a in row['action_overrides']]
            row['guidance_frame_sha256']=guide.frame_sha256
            row['expert_action_mask']=[a is not None for a in row['action_overrides']]
        elif row['event']=='reset':guide.reset(row['streams'])
        elif row['event'] in ('progress','run_end'):
            row['overridden_actions']=guide.overridden_actions;row['guidance_episodes']=guide.episode_counts
    return rows


@pytest.mark.parametrize('mutation',[None,'game','schedule_hash','policy_hash','expert_mask','action','pixel'])
def test_accounting_and_replay_reader_reject_invalid_guidance(tmp_path,mutation):
    rows=events();first=next(r for r in rows if r['event']=='transition')
    if mutation=='game':rows[0]['environment']='ALE/Qbert-v5'
    elif mutation=='schedule_hash':rows[0]['guidance_schedule_sha256']='0'*64
    elif mutation=='policy_hash':rows[0]['exploration_sha256']='0'*64
    elif mutation=='expert_mask':first['expert_action_mask'][0]=False
    elif mutation=='action':first['action_overrides'][0]=2
    elif mutation=='pixel':first['guidance_frame_sha256'][0]=None
    path=write_log(tmp_path,rows)
    if mutation is not None:
        with pytest.raises(ValueError):audit(path)
    else:
        result=replay_qbert_guidance.read_guided_run(path)
        assert result['accounting']['exploration_schedule_verified']
        assert not result['accounting']['exploration_pixel_actions_verified']


@pytest.mark.parametrize('args,message',[
    (['--breakout-guidance-probability','nan'],'probability'),
    (['--breakout-guidance-probability','-.1'],'probability'),
    (['--breakout-guidance-max-actions','0'],'positive u32'),
    (['--breakout-guidance-max-actions',str(2**32)],'positive u32'),
    (['--breakout-guidance-probability','0'],'requires positive'),
    (['--evaluate'],'frozen evaluation'),(['--restore','unused'],'fresh run'),
    (['--qbert-guidance-probability','.05'],'cannot combine'),
    (['--exploration-probability','.05'],'cannot combine'),
    (['--observation-size','64'],'native pixels'),
    (['--atari-protocol','published-minimal'],'full18'),
    (['--encoder-checkpoint','unused'],'learned RGB')])
def test_cli_refuses_unsupported_combinations_before_environment(monkeypatch,tmp_path,capsys,args,message):
    monkeypatch.setattr(atari_vector.gym,'make',lambda *a,**kw:pytest.fail('environment constructed'))
    monkeypatch.setattr(sys,'argv',['atari_vector','ALE/Breakout-v5','--output',str(tmp_path/'run'),
        '--breakout-guidance-probability','.05','--breakout-guidance-max-actions','640',*args])
    with pytest.raises(SystemExit) as error:atari_vector.main()
    assert error.value.code==2 and message in capsys.readouterr().err


@pytest.mark.parametrize('mutation',['choice','pixels','execution'])
def test_full_pixel_replay_rejects_forged_teacher_action_before_stepping(mutation):
    frame=np.zeros((210,160,3),dtype=np.uint8);guide=BreakoutGuidance(config(),1,18)
    row=dict(event='transition',action_overrides=[1],guidance_frame_sha256=[hashlib.sha256(frame.tobytes()).hexdigest()],actions=[1])
    if mutation=='choice':row['action_overrides']=[3]
    elif mutation=='pixels':row['guidance_frame_sha256']=['0'*64]
    else:row['actions']=[3]
    with pytest.raises(ValueError):replay_qbert_guidance.replay_rows([json.dumps(row)],[],[],[frame],guide,float('inf'))
