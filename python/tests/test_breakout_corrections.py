import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import kindle
import kindle._breakout_guidance as module
from kindle._vector_audit import audit
import atari_vector
import replay_qbert_guidance
from test_vector import write_log
from test_breakout_guidance import events as original_events


def config(probability=1, cap=1):
    return dict(kind=module.CORRECTIVE_GUIDANCE_KIND, probability=probability, seed=1009, max_actions=cap)


def test_feedback_uses_every_full18_horizontal_command_and_preserves_observations():
    meanings = ('NOOP','FIRE','UP','RIGHT','LEFT','DOWN','UPRIGHT','UPLEFT','DOWNRIGHT','DOWNLEFT',
                'UPFIRE','RIGHTFIRE','LEFTFIRE','DOWNFIRE','UPRIGHTFIRE','UPLEFTFIRE','DOWNRIGHTFIRE','DOWNLEFTFIRE')
    frame = np.zeros((210,160,3),dtype=np.uint8)
    policy = module.FeedbackPixelPolicy()
    for action,meaning in enumerate(meanings):
        assert policy.propose(frame) == 1
        with pytest.raises(ValueError,match='needs action feedback'):policy.propose(frame)
        policy.teacher.previous=(42.,130.);policy.teacher.valid_landing=(90.,4.)
        for bad in (None,True,-1,18,1.0):
            with pytest.raises(ValueError):policy.feedback(bad)
            assert policy.pending==1
        policy.feedback(action)
        assert policy.teacher.previous_command == (1 if 'RIGHT' in meaning else -1 if 'LEFT' in meaning else 0)
        assert policy.teacher.previous==(42.,130.) and policy.teacher.valid_landing==(90.,4.)
        with pytest.raises(ValueError,match='No pending proposal'):policy.feedback(action)


def test_labels_cover_released_and_unselected_states_with_atomic_feedback():
    frame=np.zeros((210,160,3),dtype=np.uint8)
    for probability in (1,1e-20):
        guidance=module.BreakoutGuidance(config(probability),2,18)
        initial=list(guidance._policies)
        for tick in range(3):
            assert guidance.actions([frame,frame])==([1,1] if probability==1 and tick==0 else [None,None])
            assert guidance.expert_labels==[1,1]
            assert guidance.frame_sha256==[hashlib.sha256(frame.tobytes()).hexdigest()]*2
            with pytest.raises(ValueError,match='feedback batch'):guidance.feedback([3,18])
            assert all(p.pending==1 and p.rejected==tick for p in guidance._policies)
            guidance.feedback([3,17])
            assert [p.teacher.previous_command for p in guidance._policies]==[1,-1]
        assert guidance.overridden_actions==([1,1] if probability==1 else [0,0])
        guidance.reset([0])
        assert guidance._policies[0] is not initial[0] and guidance._policies[1] is initial[1]
        assert guidance._policies[0].rejected==0 and guidance._policies[1].rejected==3


def corrective_events():
    rows=original_events();rows[0]['exploration']=config()
    rows[0]['exploration_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    guide=module.BreakoutGuidance(config(),2,18);frame=np.zeros((210,160,3),dtype=np.uint8)
    for row in rows[1:]:
        if row['event']=='transition':
            row['action_overrides']=guide.actions([frame,frame])
            row['actions']=[17 if a is None else a for a in row['action_overrides']]
            row['expert_labels']=list(guide.expert_labels);row['expert_action_mask']=[True,True]
            row['guidance_frame_sha256']=guide.frame_sha256;guide.feedback(row['actions'])
        elif row['event']=='reset':guide.reset(row['streams'])
        elif row['event'] in ('progress','run_end'):
            row['overridden_actions']=guide.overridden_actions;row['guidance_episodes']=guide.episode_counts
    return rows


@pytest.mark.parametrize('mutation',[None,'missing','null','qbert_action','numeric_mask','wrong_mask','override','released_hash','no_imitation','undeclared'])
def test_accounting_keeps_breakout_labels_separate_from_executed_actions(tmp_path,mutation):
    rows=corrective_events();transitions=[r for r in rows if r['event']=='transition']
    assert transitions[1]['actions']==[17,17] and transitions[1]['expert_labels']==[1,1]
    if mutation=='missing':del transitions[1]['expert_labels']
    elif mutation=='null':transitions[1]['expert_labels'][0]=None
    elif mutation=='qbert_action':transitions[1]['expert_labels'][0]=2
    elif mutation=='numeric_mask':transitions[1]['expert_action_mask'][0]=1
    elif mutation=='wrong_mask':transitions[1]['expert_action_mask'][0]=False
    elif mutation=='override':transitions[0]['expert_labels'][0]=3
    elif mutation=='released_hash':transitions[1]['guidance_frame_sha256'][0]=None
    elif mutation=='no_imitation':del rows[0]['config']['expert_imitation']
    elif mutation=='undeclared':rows[0]['exploration']['kind']=module.GUIDANCE_KIND
    if mutation is None:
        result=audit(write_log(tmp_path,rows));assert result['actions']==6 and result['updates']==3
    else:
        with pytest.raises(ValueError):audit(write_log(tmp_path,rows))


@pytest.mark.parametrize('args,message',[
    ([], 'require Breakout guidance and imitation'),
    (['--breakout-guidance-probability','1','--breakout-guidance-max-actions','1'], 'require Breakout guidance and imitation'),
    (['--guidance-imitation-scale','1'], 'fresh guided training'),
    (['--breakout-guidance-probability','1','--guidance-imitation-scale','1'], 'explicit max-actions cap'),
    (['--breakout-guidance-probability','1','--breakout-guidance-max-actions','1','--guidance-imitation-scale','1','--restore','unused'], 'fresh run'),
    (['--breakout-guidance-probability','1','--breakout-guidance-max-actions','1','--guidance-imitation-scale','1','--evaluate'], 'frozen evaluation')])
def test_cli_rejects_invalid_setup_before_environment(monkeypatch,tmp_path,capsys,args,message):
    monkeypatch.setattr(atari_vector.gym,'make',lambda *a,**kw:pytest.fail('environment constructed'))
    monkeypatch.setattr(sys,'argv',['atari_vector','ALE/Breakout-v5','--output',str(tmp_path/'run'),'--breakout-corrective-labels',*args])
    with pytest.raises(SystemExit) as error:atari_vector.main()
    assert error.value.code==2 and message in capsys.readouterr().err


def test_runner_and_replay_keep_distinct_recommendations_feedback_and_resets(monkeypatch,tmp_path):
    teachers=[];created=[];calls=[]
    class Teacher:
        pending=None
        def __init__(self):self.chosen=[];teachers.append(self)
        def propose(self,frame):
            assert self.pending is None
            self.pending=int(frame[0,0,0]);return self.pending
        def feedback(self,action):
            assert self.pending is not None
            self.chosen.append((self.pending,action));self.pending=None
    monkeypatch.setattr(module,'FeedbackPixelPolicy',Teacher)
    class Environment:
        action_space=SimpleNamespace(n=18)
        action_meanings=['NOOP']+[str(n) for n in range(1,18)]
        executed_action_frames=reset_noop_frames=emulator_resets=0
        def __init__(self):self.actions=[];self.returned=0.
        def reset(self,seed=None):
            self.length=0;self.returned=0.;self.emulator_resets+=1
            return np.full((210,160,3),0 if seed is not None else 4,dtype=np.uint8),{}
        def step(self,action):
            self.actions.append(action);self.length+=1;self.executed_action_frames+=4;self.returned+=1.
            return np.full((210,160,3),3 if self.length==1 else 4,dtype=np.uint8),1.,self.length==2,False,{}
        def close(self):pass
    class Agent:
        environment_step=learner_step=replay_len=0
        training_debt=0.
        provenance={'test_stub':True}
        gpu_device={'device_name':'CPU-test-stub','is_software_emulated':True}
        cpu_worker_threads=1
        trainable_parameter_counts={'world':0,'behavior':0}
        @classmethod
        def learned_rgb(cls,streams,config):
            obj=cls();obj.config=config;return obj
        def begin_episodes(self,ids,frames):self.replay_len+=len(ids)
        def act(self,greedy=False,action_overrides=None,*,expert_actions=None,expert_labels=None):
            assert expert_actions is None and expert_labels is not None
            calls.append((list(action_overrides),list(expert_labels)))
            return [17 if forced is None else forced for forced in action_overrides]
        def observe(self,ids,frames,rewards,terminated,truncated):
            self.environment_step+=len(ids);self.replay_len+=len(ids)
            return [[reward,0.] for reward in rewards]
        def learn_scheduled(self):return []
    def make(*args,**kwargs):
        env=Environment();created.append(env);return env
    monkeypatch.setattr(atari_vector.gym,'make',make)
    monkeypatch.setattr(atari_vector,'DreamerAtariPreprocessing',lambda env,**kw:env)
    monkeypatch.setattr(kindle,'VectorAgent',Agent)
    output=tmp_path/'run.jsonl'
    monkeypatch.setattr(sys,'argv',['atari_vector','ALE/Breakout-v5','--output',str(output),'--num-envs','2',
        '--steps','6','--train-ratio','0','--breakout-guidance-probability','1','--breakout-guidance-max-actions','1',
        '--guidance-imitation-scale','1','--breakout-corrective-labels'])
    atari_vector.main()
    rows=[json.loads(line) for line in output.open()]
    assert rows[0]['gpu_device']==Agent.gpu_device and rows[0]['model_provenance']==Agent.provenance
    assert rows[0]['trainable_parameter_counts']==Agent.trainable_parameter_counts
    assert calls==[([0,0],[0,0]),([None,None],[3,3]),([4,4],[4,4])]
    assert [teacher.chosen for teacher in teachers]==[[(0,0),(3,17)],[(0,0),(3,17)],[(4,4)],[(4,4)]]
    assert [env.actions for env in created]==[[0,17,4],[0,17,4]]
    assert audit(output)['actions']==6
    envs=[Environment(),Environment()]
    frames=[env.reset(seed=1009+i)[0] for i,env in enumerate(envs)]
    monitors=[SimpleNamespace(task=SimpleNamespace(result=lambda *_,env=env:dict(episode_score=env.returned))) for env in envs]
    episodes,_=replay_qbert_guidance.replay_rows([json.dumps(row) for row in rows[1:]],envs,monitors,frames,
        module.BreakoutGuidance(rows[0]['exploration'],2,18),float('inf'))
    assert len(episodes)==2 and [env.actions for env in envs]==[[0,17,4],[0,17,4]]
