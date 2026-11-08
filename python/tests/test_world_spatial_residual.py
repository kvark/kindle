"""Decoder mode/header errors must be reported without GPU construction."""
import json

import kindle
import pytest


def metadata():
    c = kindle.default_config(18, 'tiny')
    c.update(observation_kind='rgb32', rgb_spatial_decoder=True,
             rgb_spatial_fusion=True, rgb_spatial_residual=True,
             batch_length=32, world_backprop_length=32,
             world_microbatch_size=c['batch_size'], disagreement_bonus=False)
    for name in ('policy', 'value', 'replay_value'):
        c['loss_scales'][name] = 0.
    return dict(format='kindle-world-only-v15', config=c, prediction_length=16,
                pixel_loss='motion_event_rollout', updates=0)


@pytest.mark.parametrize('version', [1, 13, 14])
def test_spatial_residual_rejects_old_header(tmp_path, version):
    m = metadata()
    m['format'] = f'kindle-world-only-v{version}'
    (tmp_path / 'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='spatial residual requires checkpoint format v15'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field', ['rgb_spatial_residual', 'rgb_spatial_fusion', 'rgb_spatial_decoder'])
def test_spatial_residual_rejects_missing_architecture(tmp_path, field):
    m = metadata()
    m['config'][field] = False
    (tmp_path / 'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='spatial residual|spatial fusion'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective', ['motion_clean_prior', 'motion_cross_entropy'])
def test_spatial_residual_rejects_other_objective(objective):
    with pytest.raises(ValueError, match='spatial residual requires the suffix event objective'):
        kindle.WorldModelLearner(metadata()['config'], prediction_length=16, pixel_loss=objective)


def test_spatial_residual_is_default_off_and_requires_fusion():
    assert 'rgb_spatial_residual' not in kindle.default_config(18, 'tiny')
    c = metadata()['config']
    c['rgb_spatial_fusion'] = False
    with pytest.raises(ValueError, match='spatial residual requires the standalone fused RGB decoder'):
        kindle.WorldModelLearner(c, prediction_length=16, pixel_loss='motion_event_rollout')
