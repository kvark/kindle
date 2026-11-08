"""Combined architecture headers/objectives reject before GPU construction."""
import json
import kindle
import pytest


def metadata():
    c=kindle.default_config(18,'tiny')
    c.update(observation_kind='rgb32',rgb_spatial_decoder=True,rgb_spatial_fusion=True,
        rgb_spatial_action=True,rgb_spatial_transport=True,batch_length=32,
        world_backprop_length=32,world_microbatch_size=c['batch_size'],disagreement_bonus=False)
    for name in ('policy','value','replay_value'):c['loss_scales'][name]=0.
    return dict(format='kindle-world-only-v19',config=c,prediction_length=16,
        pixel_loss='motion_event_rollout',updates=0)


@pytest.mark.parametrize('version',range(1,19))
def test_action_transport_requires_v19_header(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='combined action transport requires checkpoint format v19'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field',['rgb_spatial_action','rgb_spatial_transport','rgb_spatial_fusion','rgb_spatial_decoder'])
def test_action_transport_rejects_missing_architecture(tmp_path,field):
    m=metadata();m['config'][field]=False
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError):kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective',['motion_cross_entropy','motion_clean_prior','motion_fixed_teacher_events'])
def test_action_transport_requires_suffix_events(objective):
    with pytest.raises(ValueError):kindle.WorldModelLearner(metadata()['config'],prediction_length=16,pixel_loss=objective)


def test_action_transport_rejects_residual_and_zero_horizon():
    c=metadata()['config'];c['rgb_spatial_residual']=True
    with pytest.raises(ValueError):kindle.WorldModelLearner(c,prediction_length=16,pixel_loss='motion_event_rollout')
    with pytest.raises(ValueError):kindle.WorldModelLearner(metadata()['config'],prediction_length=0,pixel_loss='motion_event_rollout')
