import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import kindle
import kindle._qbert_guidance as module
from kindle._vector_audit import audit
import atari_vector
import replay_qbert_guidance
from test_qbert_guidance import guided_events, test_cli_refuses_invalid_guidance_before_gpu as reject_cli
from test_vector import write_log


def config(probability=1, cap=1):
    return dict(kind=module.CORRECTIVE_GUIDANCE_KIND, probability=probability, seed=1009, max_actions=cap)


def test_feedback_rejects_invalid_calls_without_losing_observed_state():
    policy = module.FeedbackPixelPolicy()
    frame = np.zeros((210,160,3), dtype=np.uint8)
    assert policy.propose(frame) == 0
    with pytest.raises(ValueError, match='needs action feedback'):
        policy.propose(frame)
    movement = policy.teacher.policy
    movement.target, movement.age, movement.command, movement.origin = (1,0), 7, 5, (0,0)
    observed = dict(movement.lit), movement.previous, movement.stable
    for invalid in (None, True, -1, 18, 1.0):
        with pytest.raises(ValueError):
            policy.feedback(invalid)
        assert policy.pending == 0 and movement.target == (1,0)
    policy.feedback(3)
    assert (movement.target, movement.age, movement.command, movement.origin) == (None,0,0,None)
    assert (movement.lit, movement.previous, movement.stable) == observed
    assert policy.rejected == 1
    with pytest.raises(ValueError, match='No pending proposal'):
        policy.feedback(3)


def test_corrective_guidance_labels_released_and_unselected_states():
    frame = np.zeros((210,160,3), dtype=np.uint8)
    for probability in (1, 1e-20):
        guidance = module.QbertGuidance(config(probability), 2, 18)
        initial = list(guidance._policies)
        for tick in range(3):
            assert guidance.actions([frame,frame]) == ([0,0] if probability == 1 and tick == 0 else [None,None])
            assert guidance.expert_labels == [0,0]
            assert guidance.frame_sha256 == [hashlib.sha256(frame.tobytes()).hexdigest()]*2
            with pytest.raises(ValueError, match='feedback batch'):
                guidance.feedback([3,18])
            assert all(policy.pending == 0 and policy.rejected == tick for policy in guidance._policies)
            guidance.feedback([3,4])
        assert guidance.overridden_actions == ([1,1] if probability == 1 else [0,0])
        guidance.reset([0])
        assert guidance._policies[0] is not initial[0] and guidance._policies[1] is initial[1]
        assert guidance._policies[0].rejected == 0 and guidance._policies[1].rejected == 3


@pytest.mark.parametrize('mutation', [None, 'missing', 'null', 'invalid_action', 'numeric_mask',
    'wrong_mask', 'override', 'released_hash', 'no_imitation', 'undeclared'])
def test_corrective_accounting_separates_labels_from_actual_actions(tmp_path, mutation):
    rows = guided_events(config())
    transitions = [row for row in rows if row['event']=='transition']
    assert transitions[1]['actions'] == [1,1] and transitions[1]['expert_labels'] == [0,0]
    if mutation == 'missing': del transitions[1]['expert_labels']
    elif mutation == 'null': transitions[1]['expert_labels'][0] = None
    elif mutation == 'invalid_action': transitions[1]['expert_labels'][0] = 6
    elif mutation == 'numeric_mask': transitions[1]['expert_action_mask'][0] = 1
    elif mutation == 'wrong_mask': transitions[1]['expert_action_mask'][0] = False
    elif mutation == 'override': transitions[0]['expert_labels'][0] = 2
    elif mutation == 'released_hash': transitions[1]['guidance_frame_sha256'][0] = None
    elif mutation == 'no_imitation': del rows[0]['config']['expert_imitation']
    elif mutation == 'undeclared': rows[0]['exploration']['kind'] = module.CONTINUATION_GUIDANCE_KIND
    if mutation is None:
        result = audit(write_log(tmp_path,rows))
        assert result['actions'] == 6 and result['updates'] == 3
        assert result['overridden_actions'] == [2,1]
    else:
        with pytest.raises(ValueError):
            audit(write_log(tmp_path,rows))


def test_replay_rejects_a_forged_released_label_before_stepping():
    frame=np.zeros((210,160,3),dtype=np.uint8)
    digest=hashlib.sha256(frame.tobytes()).hexdigest()
    class Environment:
        executed_action_frames=0
        def step(self,action):
            self.executed_action_frames+=4
            return frame,0.,False,False,{}
    env=Environment()
    rows=[dict(event='transition',run_step=tick,actions=[0 if tick==1 else 7],
        action_overrides=[0 if tick==1 else None],expert_labels=[0 if tick==1 else 2],
        guidance_frame_sha256=[digest],rewards=[0.],terminated=[False],truncated=[False],
        executed_action_frames=[tick*4]) for tick in (1,2)]
    with pytest.raises(ValueError, match='corrective expert labels differ'):
        replay_qbert_guidance.replay_rows([json.dumps(row) for row in rows],[env],[],[frame],
            module.QbertGuidance(config(),1,18),float('inf'))
    assert env.executed_action_frames==4


@pytest.mark.parametrize('args,message', [
    ([], 'require continuation guidance and imitation'),
    (['--qbert-guidance-policy','continuation','--qbert-guidance-max-actions','1'], 'require continuation guidance and imitation'),
    (['--guidance-imitation-scale','1'], 'require continuation guidance and imitation'),
    (['--qbert-guidance-policy','continuation','--guidance-imitation-scale','1'], 'explicit max-actions cap'),
    (['--qbert-guidance-policy','continuation','--qbert-guidance-max-actions','1','--guidance-imitation-scale','1','--restore','unused'], 'fresh run'),
    (['--qbert-guidance-policy','continuation','--qbert-guidance-max-actions','1','--guidance-imitation-scale','1','--evaluate'], 'fresh guided training'),
])
def test_corrective_cli_rejects_invalid_setup_before_native(monkeypatch,tmp_path,capsys,args,message):
    reject_cli(monkeypatch,tmp_path,capsys,['--qbert-corrective-labels',*args],message)


def test_corrective_cli_rejects_old_native(monkeypatch,tmp_path,capsys):
    def act(self,action_overrides=None,expert_actions=None):
        raise AssertionError('No inference expected')
    monkeypatch.setattr(kindle,'VectorAgent',SimpleNamespace(act=act))
    reject_cli(monkeypatch,tmp_path,capsys,['--qbert-corrective-labels','--qbert-guidance-policy','continuation',
        '--qbert-guidance-max-actions','1','--guidance-imitation-scale','1'],'native passive-label support')


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
            return np.full((210,160,3),0 if seed is not None else 3,dtype=np.uint8),{}
        def step(self,action):
            self.actions.append(action);self.length+=1;self.executed_action_frames+=4;self.returned+=1.
            return np.full((210,160,3),2 if self.length==1 else 4,dtype=np.uint8),1.,self.length==2,False,{}
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
    monkeypatch.setattr(sys,'argv',['atari_vector','ALE/Qbert-v5','--output',str(output),'--num-envs','2',
        '--steps','6','--train-ratio','0','--qbert-guidance-probability','1','--qbert-guidance-max-actions','1',
        '--qbert-guidance-policy','continuation','--guidance-imitation-scale','1','--qbert-corrective-labels'])
    atari_vector.main()
    rows=[json.loads(line) for line in output.open()]
    assert rows[0]['gpu_device']==Agent.gpu_device and rows[0]['model_provenance']==Agent.provenance
    assert rows[0]['trainable_parameter_counts']==Agent.trainable_parameter_counts
    assert calls==[([0,0],[0,0]),([None,None],[2,2]),([3,3],[3,3])]
    assert [teacher.chosen for teacher in teachers]==[[(0,0),(2,17)],[(0,0),(2,17)],[(3,3)],[(3,3)]]
    assert [env.actions for env in created]==[[0,17,3],[0,17,3]]
    assert audit(output)['actions']==6
    envs=[Environment(),Environment()]
    frames=[env.reset(seed=1009+i)[0] for i,env in enumerate(envs)]
    monitors=[SimpleNamespace(task=SimpleNamespace(result=lambda *_,env=env:dict(episode_score=env.returned))) for env in envs]
    episodes,_=replay_qbert_guidance.replay_rows([json.dumps(row) for row in rows[1:]],envs,monitors,frames,
        module.QbertGuidance(rows[0]['exploration'],2,18),float('inf'))
    assert len(episodes)==2 and [env.actions for env in envs]==[[0,17,3],[0,17,3]]
