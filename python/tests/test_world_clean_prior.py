"""Causal-prior objective/header/report checks without GPU discovery."""
import json
from pathlib import Path
import sys

import kindle
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))

def metadata():
    c=kindle.default_config(18,'tiny')
    c.update(observation_kind='rgb32',rgb_spatial_decoder=True,rgb_spatial_fusion=True,world_backprop_length=c['batch_length'],world_microbatch_size=c['batch_size'],disagreement_bonus=False)
    for n in ('policy','value','replay_value'):c['loss_scales'][n]=0.
    return dict(format='kindle-world-only-v14',config=c,pixel_loss='motion_clean_prior',prediction_length=16,updates=0)

@pytest.mark.parametrize('version',range(1,14))
def test_clean_prior_requires_v14_before_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}';(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='clean prior objective requires checkpoint format v14'):kindle.WorldModelLearner.restore(str(tmp_path))

@pytest.mark.parametrize('field',['rgb_spatial_decoder','rgb_spatial_fusion'])
def test_clean_prior_requires_both_fusion_flags(tmp_path,field):
    m=metadata();m['config'][field]=False;(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='invalid clean prior|invalid spatial fusion'):kindle.WorldModelLearner.restore(str(tmp_path))

def test_clean_prior_rejects_an_empty_suffix_before_gpu():
    with pytest.raises(ValueError,match='positive prediction length'):kindle.WorldModelLearner(metadata()['config'],pixel_loss='motion_clean_prior')

def test_clean_prior_names_identify_changed_readouts_and_exclude_fraction_from_total():
    import breakout_world_corpus as corpus
    names=corpus.report_names('motion_clean_prior',16)
    assert len(names)==14 and names[1]=='clean_prior_rgb' and names[4:6]==('clean_prior_reward','clean_prior_continuation') and names[13]=='clean_prior_valid_fraction'
    values=list(range(14));scales=dict(reconstruction=100,dynamics=1,representation=.1,reward=2,continuation=3)
    assert corpus.reported_total(dict(loss_scales=scales),values,'motion_clean_prior',16)==100*(1+9)+2+.1*3+2*(4+11)+3*(5+12)
    with pytest.raises(ValueError,match='positive suffix'):corpus.report_names('motion_clean_prior',0)
