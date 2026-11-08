"""Local action architecture contracts must reject before GPU construction."""
import json
import kindle
import pytest


def metadata():
    c=kindle.default_config(18,'tiny')
    c.update(observation_kind='rgb32',rgb_spatial_decoder=True,rgb_spatial_fusion=True,
        rgb_spatial_action=True,batch_length=32,world_backprop_length=32,
        world_microbatch_size=c['batch_size'],disagreement_bonus=False)
    for name in ('policy','value','replay_value'):c['loss_scales'][name]=0.
    return dict(format='kindle-world-only-v18',config=c,prediction_length=16,
        pixel_loss='motion_event_rollout',updates=0)


@pytest.mark.parametrize('version',range(1,18))
def test_local_action_rejects_older_headers_before_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='spatial action requires checkpoint format v18'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field',['rgb_spatial_action','rgb_spatial_fusion','rgb_spatial_decoder'])
def test_local_action_rejects_missing_architecture(tmp_path,field):
    m=metadata();m['config'][field]=False
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError):kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective',['motion_cross_entropy','motion_clean_prior','motion_fixed_teacher_events'])
def test_local_action_requires_its_qualified_objective(objective):
    with pytest.raises(ValueError):
        kindle.WorldModelLearner(metadata()['config'],prediction_length=16,pixel_loss=objective)


def test_local_action_default_off_and_incompatible_residual_or_zero_suffix():
    assert 'rgb_spatial_action' not in kindle.default_config(18,'tiny')
    m=metadata();m['config']['rgb_spatial_residual']=True
    with pytest.raises(ValueError):kindle.WorldModelLearner(m['config'],prediction_length=16,pixel_loss='motion_event_rollout')
    with pytest.raises(ValueError):kindle.WorldModelLearner(metadata()['config'],prediction_length=0,pixel_loss='motion_event_rollout')
    m=metadata();m['config']['rgb_spatial_transport']=True
    with pytest.raises(ValueError):kindle.WorldModelLearner(m['config'],prediction_length=16,pixel_loss='motion_event_rollout')
