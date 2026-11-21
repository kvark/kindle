"""Late command conditioning must reject incompatible headers before GPU work."""
import json

import kindle
import pytest


def metadata(features=False):
    config = kindle.default_config(18, 'tiny')
    config.update(observation_kind='rgb32', batch_length=4, world_backprop_length=4,
                  world_microbatch_size=config['batch_size'], disagreement_bonus=False,
                  rgb_spatial_decoder=True, rgb_spatial_fusion=True,
                  rgb_spatial_action=True, rgb_late_action=True)
    for name in ('policy', 'value', 'replay_value'):
        config['loss_scales'][name] = 0.
    value = dict(format='kindle-world-only-v26', config=config, updates=0,
                 prediction_length=2, pixel_loss='motion_event_rollout',
                 late_action_base_format='kindle-world-only-v18')
    if features:
        config['world_action_features'] = [[0.] * 18 for _ in range(18)]
        value.update(late_action_base_format='kindle-world-only-v25',
                     action_feature_base_format='kindle-world-only-v18')
    return value


@pytest.mark.parametrize('features', [False, True])
@pytest.mark.parametrize('field,replacement', [
    ('format', 'kindle-world-only-v18'),
    ('format', 'kindle-world-only-v25'),
    ('late_action_base_format', 'kindle-world-only-v2'),
    ('training_scope', 'all'),
    ('phase_start_updates', 1),
    ('prediction_length', 0),
    ('pixel_loss', 'motion_posterior_history'),
])
def test_late_header_rejects_disguises_before_tensor_loading(tmp_path, features, field, replacement):
    value = metadata(features)
    value[field] = replacement
    (tmp_path / 'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='late action'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field', ['rgb_spatial_action', 'rgb_spatial_fusion', 'rgb_spatial_decoder'])
def test_late_action_requires_local_fused_decoder(field):
    config = metadata()['config']
    config[field] = False
    with pytest.raises(ValueError):
        kindle.WorldModelLearner(config, prediction_length=2, pixel_loss='motion_event_rollout')


def test_default_omits_late_option():
    assert 'rgb_late_action' not in kindle.default_config(18, 'tiny')
