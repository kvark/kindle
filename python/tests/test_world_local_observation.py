"""Malformed local H0 checkpoints must fail before constructing a GPU learner."""
import json

import kindle
import pytest


def metadata():
    c = kindle.default_config(18, 'tiny')
    c.update(observation_kind='rgb32', rgb_spatial_decoder=True, rgb_spatial_fusion=True,
             rgb_spatial_action=True, batch_length=32, world_backprop_length=32,
             world_microbatch_size=c['batch_size'], disagreement_bonus=False)
    for name in ('policy', 'value', 'replay_value'):
        c['loss_scales'][name] = 0.
    return dict(format='kindle-world-only-v20', config=c, prediction_length=0,
                pixel_loss='motion_cross_entropy', updates=0)


@pytest.mark.parametrize('objective,length', [
    ('motion_event_rollout', 0), ('motion_event_rollout', 16),
    ('motion_cross_entropy', 16), ('squared_error', 0), ('cross_entropy', 0),
    ('motion_cross_entropy', None), ('motion_cross_entropy', -1),
])
def test_local_observation_rejects_wrong_objective_before_gpu(tmp_path, objective, length):
    m = metadata()
    m.update(pixel_loss=objective, prediction_length=length)
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='invalid spatial action checkpoint objective or geometry'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('version', [18, 19])
def test_local_observation_cannot_disguise_its_header_as_an_old_runtime(tmp_path, version):
    m = metadata()
    m['format'] = f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_local_observation_rejects_phase_scope_before_gpu(tmp_path):
    m = metadata()
    m['training_scope'] = 'all'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='world training scope requires checkpoint format v3'):
        kindle.WorldModelLearner.restore(str(tmp_path))
