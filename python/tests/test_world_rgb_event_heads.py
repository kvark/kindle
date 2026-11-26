"""RGB event-head scope and headers must reject invalid restores before GPU use."""
import json

import kindle
import pytest


def metadata(features=False):
    config = kindle.default_config(18, '1m')
    config.update(observation_kind='rgb64', batch_length=32, world_backprop_length=32,
                  world_microbatch_size=config['batch_size'], disagreement_bonus=False,
                  rgb_spatial_decoder=True, rgb_spatial_fusion=True,
                  rgb_spatial_action=True, rgb_event_heads=True)
    for name in ('policy', 'value', 'replay_value'):
        config['loss_scales'][name] = 0.
    value = dict(format='kindle-world-only-v28', config=config, updates=0,
                 prediction_length=16, pixel_loss='motion_event_rollout',
                 rgb_event_head_base_format='kindle-world-only-v18',
                 rgb_event_head_geometry=dict(latent=640, visual=256, inputs=896, units=64))
    if features:
        config['world_action_features'] = [[0.] * 18 for _ in range(18)]
        value.update(rgb_event_head_base_format='kindle-world-only-v25',
                     action_feature_base_format='kindle-world-only-v18')
    return value


@pytest.mark.parametrize('features', [False, True])
@pytest.mark.parametrize('field,replacement', [
    ('format', 'kindle-world-only-v18'),
    ('format', 'kindle-world-only-v25'),
    ('rgb_event_head_base_format', 'kindle-world-only-v2'),
    ('rgb_event_head_geometry', dict(latent=640, visual=256, inputs=640, units=64)),
    ('training_scope', 'all'),
    ('phase_start_updates', 1),
    ('prediction_length', 8),
    ('pixel_loss', 'motion_posterior_history'),
])
def test_rgb_event_header_rejects_disguises_before_tensor_loading(tmp_path, features, field, replacement):
    value = metadata(features)
    value[field] = replacement
    (tmp_path / 'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='RGB event head'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field', ['rgb_event_heads', 'rgb_event_head_geometry', 'rgb_event_head_base_format'])
def test_rgb_event_header_rejects_missing_declarations_before_tensor_loading(tmp_path, field):
    value = metadata()
    if field == 'rgb_event_heads':
        del value['config'][field]
    else:
        del value[field]
    (tmp_path / 'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='RGB event head'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field', ['rgb_wide_motion', 'rgb_late_action', 'world_image_feedback',
                                  'world_mean_latents', 'rgb_previous_frame', 'rgb_coordinates'])
def test_rgb_event_heads_reject_unqualified_compositions_before_gpu(field):
    config = metadata()['config']
    config[field] = True
    with pytest.raises(ValueError, match='RGB event heads'):
        kindle.WorldModelLearner(config, prediction_length=16, pixel_loss='motion_event_rollout')


def test_rgb_event_default_is_off():
    assert 'rgb_event_heads' not in kindle.default_config(18, '1m')
