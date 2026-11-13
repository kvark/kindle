"""Persisted command features must reject incompatible inputs before GPU work."""
import json

import kindle
import pytest


def metadata():
    config = kindle.default_config(18, 'tiny')
    config.update(observation_kind='rgb32', batch_length=4,
                  world_backprop_length=4, world_microbatch_size=config['batch_size'],
                  disagreement_bonus=False,
                  world_action_features=[[0.] * 18 for _ in range(18)])
    for name in ('policy', 'value', 'replay_value'):
        config['loss_scales'][name] = 0.
    return dict(format='kindle-world-only-v25', config=config,
                action_feature_base_format='kindle-world-only-v2',
                prediction_length=2, pixel_loss='motion_cross_entropy', updates=0)


def restore(tmp_path, value):
    (tmp_path / 'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='action feature'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('version', [1, 2, 18, 24])
def test_feature_table_requires_its_own_checkpoint_version(tmp_path, version):
    value = metadata()
    value['format'] = f'kindle-world-only-v{version}'
    restore(tmp_path, value)


@pytest.mark.parametrize('field,replacement', [
    ('action_feature_base_format', 'kindle-world-only-v3'),
    ('action_feature_base_format', 'kindle-world-only-v18'),
    ('training_scope', 'frozen_visual'),
    ('phase_start_updates', 1),
    ('prediction_length', 0),
])
def test_incompatible_feature_header_fails_before_tensor_or_gpu_loading(tmp_path, field, replacement):
    value = metadata()
    value[field] = replacement
    restore(tmp_path, value)


@pytest.mark.parametrize('table', [[], [[0.] * 18] * 17, [[0.] * 17] * 18,
                                 [[2.] * 18] * 18])
def test_invalid_command_table_fails_before_gpu_construction(table):
    config = metadata()['config']
    config['world_action_features'] = table
    with pytest.raises(ValueError, match='action features'):
        kindle.WorldModelLearner(config, prediction_length=2,
                                 pixel_loss='motion_cross_entropy')


def test_default_omits_table_and_new_header_requires_table(tmp_path):
    assert 'world_action_features' not in kindle.default_config(18, 'tiny')
    value = metadata()
    del value['config']['world_action_features']
    restore(tmp_path, value)
