"""The bounded H32 adapter profile; no GPU or emulator construction."""
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import breakout_world_coverage as coverage
import breakout_world_corpus as corpus


def streams():
    data=[]
    for lane in range(16):
        action=np.zeros(160,np.int32);action[[0,121]]=-1
        terminal=np.zeros(160,bool);terminal[118]=True
        data.append(dict(action=action,terminal=terminal,reward=np.zeros(160),
            source_wait=np.arange(160)<64,rgb=np.zeros((160,3,64,64),np.uint8),
            position=np.zeros((160,3)),lives=np.full(160,5)))
    return data


def config():return dict(seed=41001,batch_size=8,batch_length=48,rgb_spatial_action=True,
    loss_scales=dict(reconstruction=100.,dynamics=1.,representation=.1,reward=1.,continuation=1.))


def test_h32_pools_cover_all_future_terminal_offsets_and_keep_waits_clean():
    data=streams();waits,endings=coverage.sampling_pools(data,48)
    assert set(118-endings[:,1])==set(range(16,48))
    for lane,start in endings:
        assert data[lane]['terminal'][start+16:start+48].any()
        assert not (data[lane]['action'][start+1:119]==-1).any()
    for lane,start in waits:
        assert lane>=8 and start+48<=160
        assert data[lane]['source_wait'][start:start+48].all()
        assert not data[lane]['terminal'][start:start+48].any()
        assert (data[lane]['action'][start+1:start+48]>=0).all()


def test_h32_plan_is_deterministic_and_recounts_the_new_geometry():
    data=streams();c=config();a,counts=coverage.coverage_plan(data,c,64,32);b,other=coverage.coverage_plan(data,c,64,32)
    np.testing.assert_array_equal(a,b);assert counts==other and a.shape==(64,8,2)
    assert (a[:,:4,0]<8).all() and (a[:,4:7,0]>=8).all()
    terminals=prior=0
    for row in a:
        lane,start=row[6];assert data[lane]['source_wait'][start:start+48].all()
        lane,start=row[7];assert data[lane]['terminal'][start+16:start+48].any()
        batch=corpus.training_windows([data[l] for l,_ in row],[int(s) for _,s in row],48,allow_resets=True,retain_initial_targets=True)
        keep=corpus.rollout_validity(batch['action'],32)
        terminals+=int(batch['terminal'].sum());prior+=int((batch['terminal'][-32:]&keep).sum())
    assert counts['terminal_rows']==terminals and counts['prior_terminal_rows']==prior and prior>=64


@pytest.mark.parametrize('length,horizon,action',[(32,32,True),(48,16,True),(48,32,False),(64,32,True)])
def test_unselected_temporal_profiles_are_rejected(length,horizon,action):
    with pytest.raises(ValueError):coverage.coverage_plan(streams(),{**config(),'batch_length':length,'rgb_spatial_action':action},1,horizon)


def test_h32_actual_fresh_training_loop_preserves_reports_masks_and_counts(tmp_path,monkeypatch):
    import kindle
    from safetensors.numpy import save_file
    c=config();data=streams();plan,counts=coverage.coverage_plan(data,c,2,32)
    class Learner:
        updates=0;config=c;pixel_loss='motion_event_rollout'
        def save(self,path):
            path=Path(path);path.mkdir();save_file({'world.parameter':np.array([self.updates],np.float32)},path/'world.safetensors')
        def learn(self,observed,action,reward,terminal):
            assert len(observed)==48*8*3*64*64*4
            assert len(action)==len(reward)==len(terminal)==48*8
            self.updates+=1;keep=corpus.rollout_validity(np.asarray(action).reshape(48,8),32)
            v=[0.,1.,2.,3.,4.,5.,0.,2.,0.,6.,float(keep.mean()),.2,.3]
            v[0]=corpus.reported_total(c,v,self.pixel_loss,32)
            return v
    learner=Learner();output=tmp_path/'train';output.mkdir();source=tmp_path/'initial';learner.save(str(source))
    args=SimpleNamespace(output=output,reference=None,steps=2,prediction_length=32,pixel_loss='motion_event_rollout',eval_every=1,dense_fitting=1,dense_development=1)
    evaluated=[]
    monkeypatch.setattr(corpus,'require_device',lambda _:None)
    monkeypatch.setattr(corpus,'evaluate_group',lambda model,*args,**kwargs:evaluated.append(model.updates))
    monkeypatch.setitem(kindle.__dict__,'WorldModelLearner',lambda *a,**kw:learner)
    corpus.train(args,c,data,data[:8],sampler=lambda *_:(plan,counts),initial_checkpoint=source)
    result=json.loads((output/'result.json').read_text());assert result['updates']==2 and result['start_updates']==0 and result['new_updates']==2 and result['prediction_length']==32
    assert result['prior_terminal_rows']==counts['prior_terminal_rows']
    assert set(evaluated)=={0,1,2}
    rows=[json.loads(s) for s in (output/'learning.jsonl').read_text().splitlines()];assert len(rows)==2 and all(len(row['losses'])==13 for row in rows)


@pytest.mark.parametrize('mode,objective',[('collect','motion_event_rollout'),('train','motion_cross_entropy'),('restore-check','motion_clean_prior')])
def test_h32_cli_rejects_other_uses_before_creating_output(tmp_path,monkeypatch,mode,objective):
    p=tmp_path/'config.json';p.write_text(json.dumps(config()));out=tmp_path/'out'
    argv=['coverage',mode,'--config',str(p),'--output',str(out),'--prediction-length','32','--pixel-loss',objective,
        '--checkpoint','unused','--reference','unused','--dataset','unused','--supplement','unused','--development','unused','--initial-reference','unused','--sampling-plan','unused']
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(SystemExit) as error:coverage.main()
    assert error.value.code==2 and not out.exists()
