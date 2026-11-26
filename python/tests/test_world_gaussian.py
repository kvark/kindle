"""Gaussian architecture rejection checks; never construct a GPU or emulator."""
import json
import kindle
import pytest


def metadata():
    c=kindle.default_config(18,'tiny');c['world_gaussian']=True
    return dict(format='kindle-world-only-v8',config=c,pixel_loss='motion_cross_entropy',prediction_length=16,updates=0)


def test_gaussian_stays_off_in_default_config():
    assert 'world_gaussian' not in kindle.default_config(18,'tiny')
    assert 'world_gaussian_dim' not in kindle.default_config(18,'tiny')


@pytest.mark.parametrize('version',range(1,9))
def test_compact_gaussian_rejects_old_formats_before_gpu(tmp_path,version):
    m=metadata();m['config']['world_gaussian_dim']=4;m['format']=f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='compact Gaussian world requires checkpoint format v9'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('dim',[None,0,-1,1.5,'32'])
def test_v9_rejects_invalid_dimensions_before_gpu(tmp_path,dim):
    m=metadata();m['format']='kindle-world-only-v9';m['config']['world_gaussian_dim']=dim
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='Gaussian dimension|Gaussian checkpoint architecture|format v8'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('dim',[0,17])
def test_compact_gaussian_rejects_invalid_constructor_dimensions(dim):
    c=metadata()['config'];c['world_gaussian_dim']=dim
    c.update(observation_kind='rgb32',disagreement_bonus=False,batch_length=32,world_backprop_length=32)
    c['loss_scales'].update(policy=0.0,value=0.0,replay_value=0.0)
    with pytest.raises(ValueError,match='world_gaussian_dim'):
        kindle.WorldModelLearner(c,prediction_length=16,pixel_loss='motion_cross_entropy')


@pytest.mark.parametrize('version',range(1,8))
def test_gaussian_rejects_old_formats_before_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}';(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='Gaussian world requires checkpoint format v8'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field,value',[('world_gaussian',False),('rgb_latent_only',True),('rgb_previous_frame',True)])
def test_v8_requires_plain_gaussian_architecture(tmp_path,field,value):
    m=metadata();m['config'][field]=value;(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='invalid Gaussian checkpoint architecture'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_gaussian_rejects_unsupported_training_objective():
    with pytest.raises(ValueError,match='Gaussian world requires the motion RGB objective'):
        kindle.WorldModelLearner(metadata()['config'],prediction_length=16,pixel_loss='motion_recurrent_energy')


@pytest.mark.parametrize('phase',['start_fixed_teacher_phase','start_fixed_teacher_image_phase','start_frozen_visual_phase'])
def test_gaussian_cannot_be_source_for_categorical_phases(tmp_path,phase):
    m=metadata();m.update(updates=8192,prediction_length=0);(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='categorical latents|experienced single-frame all-weight H0 motion world'):
        getattr(kindle.WorldModelLearner,phase)(str(tmp_path))
