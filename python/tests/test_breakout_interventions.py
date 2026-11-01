from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
from probe_breakout_interventions import comparison, continuation, select_origins


def test_life_and_reward_outcomes_remain_separate_and_ties_are_retained():
    def row(policy_loss, teacher_loss, policy_reward, teacher_reward, confidence=.8):
        return dict(lane=6, teacher=3, choices=dict(candidate=4), confidence=dict(candidate=confidence),
            outcomes={'4':dict(life_lost=policy_loss,reward=policy_reward),
                      '3':dict(life_lost=teacher_loss,reward=teacher_reward)})
    rows = [row(True,False,4,0), row(False,True,0,2), row(False,False,0,0,.6)]
    result = comparison(rows, 'candidate')
    assert result['policy_saves_life'] == result['teacher_saves_life'] == result['same_life_outcome'] == 1
    assert result['policy_more_reward'] == result['teacher_more_reward'] == result['same_reward'] == 1
    assert result['mean_policy_minus_teacher_reward'] == pytest.approx(2/3)
    assert comparison(rows,'candidate',confidence_min=.75)['count'] == 2
    assert comparison(rows,'candidate',lane=7)['mean_policy_minus_teacher_reward'] is None


def test_identical_actions_are_not_counted_as_disagreements():
    row = dict(lane=6,teacher=3,choices=dict(candidate=3),confidence=dict(candidate=.9),
               outcomes={'3':dict(life_lost=False,reward=2)})
    assert comparison([row],'candidate')['same_reward'] == 1
    assert comparison([row],'candidate',disagreements_only=True)['count'] == 0


def test_origin_selection_uses_predeclared_split_and_phase_not_outcomes():
    rows = [dict(episode=0,step=i,split=split,phase=phase,action=3,input_pixel_sha256=str(i))
            for i,(split,phase) in enumerate([('test','near_descending'),('train','near_descending'),('test','ascending')])]
    corpus = dict(streams=[rows])
    phases = [dict(stream=0,label=4,**row) for row in rows]
    p = np.full((3,18),1/18)
    data = dict(teacher=np.array([4]*3),lane=np.array([0]*3),index=np.arange(3),
                split=np.array([r['split'] for r in rows]),control=p,candidate=p)
    selected = select_origins(corpus,data,phases)
    assert [r['index'] for r in selected] == [0]
    for r in rows:
        r['reward'] = 999
    assert select_origins(corpus,data,phases) == selected
    phases[0]['label'] = 3
    with pytest.raises(ValueError,match='identity'):
        select_origins(corpus,data,phases)


def test_single_action_then_feedback_driven_teacher_stops_at_first_life_loss():
    class Environment:
        executed_action_frames = 100
        remaining = 3
        actions = []
        @property
        def unwrapped(self): return self
        @property
        def ale(self): return self
        def lives(self): return self.remaining
        def step(self,action):
            self.actions.append(action)
            self.executed_action_frames += 4
            if len(self.actions) == 3:
                self.remaining -= 1
            return np.zeros((1,1,3),np.uint8),1.0,False,False,{}
    class Teacher:
        pending = 3
        feedbacks = []
        def feedback(self,action):
            self.feedbacks.append(action)
            self.pending = None
        def propose(self,frame):
            self.pending = 4 if self.feedbacks[-1] == 0 else 3
            return self.pending
    environment,teacher = Environment(),Teacher()
    result = continuation(environment,teacher,np.zeros((1,1,3),np.uint8),0)
    assert environment.actions == teacher.feedbacks == [0,4,3]
    assert result['actions'] == 3 and result['frames'] == 12
    assert result['life_lost'] and result['reward'] == 3
    assert not result['terminated'] and not result['truncated']
