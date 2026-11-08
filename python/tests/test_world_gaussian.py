"""Gaussian architecture rejection checks; never construct a GPU or emulator."""
import json
import kindle
import pytest


def metadata():
    c=kindle.default_config(18,'tiny');c['world_gaussian']=True
    return dict(format='kindle-world-only-v8',config=c,pixel_loss='motion_cross_entropy',prediction_length=16,updates=0)


def test_gaussian_stays_off_in_default_config():
    assert 'world_gaussian' not in kindle.default_config(18,'tiny')


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
