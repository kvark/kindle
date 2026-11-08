"""Spatial-decoder config/header rejection; no GPU or emulator construction."""
import json
import kindle
import pytest


def metadata():
    c=kindle.default_config(18,'tiny');c['rgb_spatial_decoder']=True
    return dict(format='kindle-world-only-v10',config=c,pixel_loss='motion_cross_entropy',prediction_length=16,updates=0)


def test_spatial_decoder_is_absent_by_default():
    assert 'rgb_spatial_decoder' not in kindle.default_config(18,'tiny')


@pytest.mark.parametrize('version',range(1,10))
def test_spatial_decoder_rejects_old_headers_before_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='spatial decoder requires checkpoint format v10'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field,value',[('rgb_spatial_decoder',False),('world_gaussian',True),('rgb_latent_only',True),('rgb_previous_frame',True)])
def test_v10_requires_the_selected_architecture(tmp_path,field,value):
    m=metadata();m['config'][field]=value
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='invalid spatial decoder checkpoint architecture'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_spatial_decoder_rejects_other_training_objectives():
    with pytest.raises(ValueError,match='spatial decoder requires the motion RGB objective'):
        kindle.WorldModelLearner(metadata()['config'],prediction_length=16,pixel_loss='motion_recurrent_energy')


@pytest.mark.parametrize('phase',['start_frozen_visual_phase','start_fixed_teacher_phase','start_fixed_teacher_image_phase'])
def test_spatial_decoder_cannot_enter_original_decoder_phases(tmp_path,phase):
    m=metadata();m.update(updates=8192,prediction_length=0)
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='original RGB decoder|experienced single-frame all-weight H0 motion world'):
        getattr(kindle.WorldModelLearner,phase)(str(tmp_path))
