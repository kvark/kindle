"""Standalone centroid checkpoint guards and independent report accounting."""
import json
import kindle
import pytest
import breakout_world_corpus as corpus


def metadata():
    c=kindle.default_config(18,'tiny')
    c.update(observation_kind='rgb32',batch_length=4,world_backprop_length=4,
             world_microbatch_size=c['batch_size'],disagreement_bonus=False,
             color_centroid_supervision=dict(scale=1000.,channel=0,background_scale=1.5,
                 region=[.0625,.4375,.9375,.875]))
    for key in ('policy','value','replay_value'):c['loss_scales'][key]=0.
    return dict(format='kindle-world-only-v22',config=c,pixel_loss='motion_cross_entropy',prediction_length=2,updates=0)


@pytest.mark.parametrize('version',[1,2,21])
def test_centroid_rejects_older_headers_before_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='world centroid supervision requires checkpoint format v22'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective,length',[('squared_error',2),('motion_cross_entropy',0)])
def test_centroid_constructor_rejects_other_objectives_before_gpu(objective,length):
    with pytest.raises(ValueError,match='world centroid requires a plain categorical motion RGB suffix'):
        kindle.WorldModelLearner(metadata()['config'],pixel_loss=objective,prediction_length=length)


def test_centroid_reports_separate_geometry_scale_from_rgb_scale():
    c=dict(loss_scales=dict(reconstruction=2.,dynamics=3.,representation=.1,reward=4.,continuation=5.),color_centroid_supervision=dict(scale=7.))
    values=[0.,1.,2.,3.,4.,5.,0.,0.,0.,6.,.75,8.,.25,9.,.5]
    names=corpus.report_names('motion_cross_entropy',16,color_centroid=True)
    assert len(names)==15 and names[11:]==('observed_centroid','observed_centroid_valid_fraction','future_centroid','future_centroid_valid_fraction')
    assert corpus.reported_total(c,values,'motion_cross_entropy',16)==2*(1+6)+3*2+.1*3+4*4+5*5+7*(8+9)
    assert corpus.report_names('motion_cross_entropy',16)==corpus.LOSS_NAMES
    with pytest.raises(ValueError):corpus.report_names('motion_cross_entropy',0,color_centroid=True)
