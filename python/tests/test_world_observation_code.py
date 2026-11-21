"""Invalid architecture headers must be rejected before native construction."""
import json

import kindle
import pytest


def test_observation_code_remains_off_in_default_configuration():
    assert 'rgb_latent_only' not in kindle.default_config(18, 'tiny')


@pytest.mark.parametrize('version', ['kindle-world-only-v1', 'kindle-world-only-v2', 'kindle-world-only-v3'])
def test_legacy_headers_cannot_disguise_observation_only_architecture(tmp_path, version):
    config = kindle.default_config(18, 'tiny')
    config['rgb_latent_only'] = True
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format=version, config=config,
        updates=0, prediction_length=16, pixel_loss='motion_latent_overshooting')))
    with pytest.raises(RuntimeError, match='requires world-only checkpoint format v4'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('flag', [False, None])
def test_v4_requires_its_explicit_architecture_before_loading_tensors(tmp_path, flag):
    config = kindle.default_config(18, 'tiny')
    if flag is not None: config['rgb_latent_only'] = flag
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format='kindle-world-only-v4', config=config,
        updates=0, prediction_length=16, pixel_loss='motion_latent_overshooting')))
    with pytest.raises(RuntimeError, match='invalid latent-only RGB checkpoint architecture'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_observation_only_architecture_rejects_frozen_visual_phase_before_loading(tmp_path):
    config = kindle.default_config(18, 'tiny')
    config['rgb_latent_only'] = True
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format='kindle-world-only-v4', config=config,
        updates=8192, prediction_length=0, pixel_loss='motion_cross_entropy')))
    with pytest.raises(RuntimeError, match='requires recurrent-conditioned RGB'):
        kindle.WorldModelLearner.start_frozen_visual_phase(str(tmp_path))
