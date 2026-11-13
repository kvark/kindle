import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import kindle

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
import breakout_world_frozen_visual as phase


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


def test_config_comparison_preserves_exact_native_f32_values_across_json_forms():
    config = dict(learning_rate=.0004, loss_scales=dict(representation=.1), batch_size=8)
    metadata = dict(config=phase.config_f32(config), updates=8192)
    learner = SimpleNamespace(config=config, updates=8192, phase_start_updates=8192,
        training_scope='frozen_visual', prediction_length=16, pixel_loss='motion_cross_entropy')
    assert config != metadata['config']
    phase.assert_phase(learner, metadata)


@pytest.mark.parametrize('defect', ['rate_ulp', 'nested_ulp', 'shape', 'nonfinite'])
def test_config_comparison_rejects_even_one_f32_ulp_and_structural_changes(defect):
    config = dict(learning_rate=.0004, loss_scales=dict(representation=.1), batch_size=8)
    metadata = dict(config=phase.config_f32(config), updates=8192)
    if defect == 'rate_ulp': config['learning_rate'] = float(np.nextafter(np.float32(.0004), np.float32(np.inf)))
    if defect == 'nested_ulp': config['loss_scales']['representation'] = float(np.nextafter(np.float32(.1), np.float32(np.inf)))
    if defect == 'shape': config['batch_size'] = 16
    if defect == 'nonfinite': config['learning_rate'] = float('nan')
    learner = SimpleNamespace(config=config, updates=8192, phase_start_updates=8192,
        training_scope='frozen_visual', prediction_length=16, pixel_loss='motion_cross_entropy')
    with pytest.raises(AssertionError): phase.assert_phase(learner, metadata)
