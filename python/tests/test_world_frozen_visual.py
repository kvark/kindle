import json

import pytest

import kindle


def checkpoint(tmp_path, **fields):
    metadata = dict(format='kindle-world-only-v2', config=kindle.default_config(3, 'tiny'), updates=8192,
                    prediction_length=0, pixel_loss='motion_cross_entropy')
    metadata.update(fields)
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    return str(tmp_path)


@pytest.mark.parametrize('fields', [
    dict(updates=0), dict(prediction_length=16), dict(pixel_loss='squared_error'),
    dict(format='kindle-world-only-v3', training_scope='frozen_visual',
         phase_start_updates=8192, prediction_length=16),
])
def test_phase_rejects_ineligible_headers_before_tensor_or_gpu_loading(tmp_path, fields):
    path = checkpoint(tmp_path, **fields)
    with pytest.raises(RuntimeError, match='requires an experienced all-weight H0'):
        kindle.WorldModelLearner.start_frozen_visual_phase(path)


@pytest.mark.parametrize('fields', [
    dict(training_scope='frozen_visual'),
    dict(format='kindle-world-only-v3', training_scope='all', phase_start_updates=8192, prediction_length=16),
    dict(format='kindle-world-only-v3', training_scope='frozen_visual', phase_start_updates=0, prediction_length=16),
    dict(format='kindle-world-only-v3', training_scope='frozen_visual', phase_start_updates=8193, prediction_length=16),
    dict(format='kindle-world-only-v3', training_scope='frozen_visual', phase_start_updates=8192, prediction_length=0),
])
def test_restore_rejects_bad_phase_schema_before_tensor_or_gpu_loading(tmp_path, fields):
    path = checkpoint(tmp_path, **fields)
    with pytest.raises(RuntimeError, match='(training scope requires|invalid frozen visual)'):
        kindle.WorldModelLearner.restore(path)


def test_phase_has_no_fresh_configuration_override(tmp_path):
    with pytest.raises(TypeError):
        kindle.WorldModelLearner.start_frozen_visual_phase(str(tmp_path), prediction_length=0)
