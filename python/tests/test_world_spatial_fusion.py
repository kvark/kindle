"""Fusion header/config rejection with no GPU or emulator construction."""
import json

import kindle
import pytest


def metadata():
    config = kindle.default_config(18, 'tiny')
    config.update(rgb_spatial_decoder=True, rgb_spatial_fusion=True, observation_kind='rgb32',
                  world_backprop_length=config['batch_length'],
                  world_microbatch_size=config['batch_size'], disagreement_bonus=False)
    for key in ('policy', 'value', 'replay_value'):
        config['loss_scales'][key] = 0.
    return dict(format='kindle-world-only-v12', config=config,
                pixel_loss='motion_cross_entropy', prediction_length=16, updates=0)


def test_spatial_fusion_is_absent_by_default():
    assert 'rgb_spatial_fusion' not in kindle.default_config(18, 'tiny')


@pytest.mark.parametrize('version', range(1, 12))
def test_spatial_fusion_rejects_older_headers_before_gpu(tmp_path, version):
    value = metadata()
    value['format'] = f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='spatial fusion requires checkpoint format v12'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field', ['rgb_spatial_decoder', 'rgb_spatial_fusion'])
def test_v12_requires_both_architecture_flags(tmp_path, field):
    value = metadata()
    value['config'][field] = False
    (tmp_path/'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='invalid spatial fusion checkpoint'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_spatial_fusion_cannot_use_image_feedback():
    config = metadata()['config']
    config['world_image_feedback'] = True
    with pytest.raises(ValueError, match='spatial fusion requires'):
        kindle.WorldModelLearner(config, prediction_length=16, pixel_loss='motion_image_feedback')


@pytest.mark.parametrize('phase', ['start_frozen_visual_phase', 'start_fixed_teacher_phase',
                                  'start_fixed_teacher_image_phase'])
def test_spatial_fusion_cannot_enter_original_decoder_phases(tmp_path, phase):
    value = metadata()
    value.update(updates=8192, prediction_length=0)
    (tmp_path/'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='original RGB decoder|experienced single-frame all-weight H0 motion world'):
        getattr(kindle.WorldModelLearner, phase)(str(tmp_path))
