"""Native schedule configuration and one-based report accounting, without a GPU."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import kindle

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import atari_vector
from kindle._vector_audit import audit
from test_breakout_corrections import corrective_events
from test_vector import write_log


def command(tmp_path, extra):
    return ['atari_vector.py','ALE/Breakout-v5','--output',str(tmp_path/'run.jsonl'),
            '--breakout-guidance-probability','.2','--breakout-guidance-max-actions','1920',*extra]


@pytest.mark.parametrize('extra,message',[
    (['--guidance-imitation-decay','1','2'],'requires imitation'),
    *[(['--guidance-imitation-scale','100','--guidance-imitation-decay',a,b],'within u64')
      for a,b in [('-1','2'),('2','2'),('3','2'),('0',str(2**64))]],
    (['--guidance-imitation-scale','100','--guidance-imitation-decay','1','2','--restore','unused'],'training overrides'),
    (['--guidance-imitation-scale','100','--guidance-imitation-decay','1','2','--evaluate'],'frozen evaluation'),
])
def test_invalid_or_frozen_schedule_fails_before_environment(monkeypatch,tmp_path,capsys,extra,message):
    monkeypatch.setattr(atari_vector.gym,'make',lambda *a,**kw:pytest.fail('environment constructed'))
    monkeypatch.setattr(sys,'argv',command(tmp_path,extra))
    with pytest.raises(SystemExit) as error:atari_vector.main()
    assert error.value.code==2 and message in capsys.readouterr().err


@pytest.mark.parametrize('decay',[None,(0,1),(6227,9352)])
def test_exact_native_boundary_and_legacy_omission(monkeypatch,tmp_path,decay):
    env=SimpleNamespace(action_space=SimpleNamespace(n=18),action_meanings=['NOOP']+[str(i) for i in range(1,18)],
                        reset=lambda **kw:(None,{}),close=lambda:None)
    monkeypatch.setattr(atari_vector.gym,'make',lambda *a,**kw:env)
    monkeypatch.setattr(atari_vector,'DreamerAtariPreprocessing',lambda env,**kw:env)
    monkeypatch.setitem(kindle.__dict__,'default_config',lambda actions,size:dict(action_count=actions,loss_scales={}))
    def construct(streams,config):
        expected=dict(scale=100.,action_weights=[1.]*18,world_gradient=True)
        if decay is not None:expected['decay']=dict(start_update=decay[0],end_update=decay[1])
        assert config['expert_imitation']==expected
        raise RuntimeError('checked native boundary')
    monkeypatch.setitem(kindle.__dict__,'VectorAgent',SimpleNamespace(learned_rgb=construct,
        SUPPORTS_EXPERT_WORLD_GRADIENT=True,SUPPORTS_EXPERT_IMITATION_DECAY=True,
        act=lambda action_overrides=None,expert_actions=None:None))
    extra=['--guidance-imitation-scale','100','--guidance-world-gradient']
    if decay is not None:extra+=['--guidance-imitation-decay',*map(str,decay)]
    monkeypatch.setattr(sys,'argv',command(tmp_path,extra))
    with pytest.raises(RuntimeError,match='checked native boundary'):atari_vector.main()


def test_old_native_cannot_silently_ignore_decay(monkeypatch,tmp_path,capsys):
    monkeypatch.setitem(kindle.__dict__,'VectorAgent',SimpleNamespace(act=lambda action_overrides=None,expert_actions=None:None))
    monkeypatch.setattr(atari_vector.gym,'make',lambda *a,**kw:pytest.fail('environment constructed'))
    monkeypatch.setattr(sys,'argv',command(tmp_path,['--guidance-imitation-scale','100','--guidance-imitation-decay','1','2']))
    with pytest.raises(SystemExit):atari_vector.main()
    assert 'requires native decay support' in capsys.readouterr().err


@pytest.mark.parametrize('mutation',[None,'one_step_late','missing_scale','negative_scale','nonzero_after_end','equal_bounds','boolean_bound','unknown_field'])
def test_audit_checks_each_applied_scale_against_report_counter(tmp_path,mutation):
    rows=corrective_events()
    expert=dict(scale=100.,action_weights=[1.]*18,decay=dict(start_update=1,end_update=3))
    rows[0]['config']['expert_imitation']=expert
    learners=[row['report'] for row in rows if row['event']=='learner']
    assert len(learners)==3
    for report,scale in zip(learners,[100.,50.,0.]):
        report['behavior']=dict(expert_imitation=dict(loss=.5,labeled_rows=1,weight_sum=1.,scale=scale))
    if mutation=='one_step_late':learners[1]['behavior']['expert_imitation']['scale']=100.
    elif mutation=='missing_scale':del learners[0]['behavior']['expert_imitation']['scale']
    elif mutation=='negative_scale':learners[2]['behavior']['expert_imitation']['scale']=-1.
    elif mutation=='nonzero_after_end':learners[2]['behavior']['expert_imitation']['scale']=1e-8
    elif mutation=='equal_bounds':expert['decay']['end_update']=1
    elif mutation=='boolean_bound':expert['decay']['start_update']=True
    elif mutation=='unknown_field':expert['decay']['extra']=0
    if mutation is None:
        assert audit(write_log(tmp_path,rows))['updates']==3
    else:
        with pytest.raises(ValueError,match='expert imitation'):audit(write_log(tmp_path,rows))
