"""Denoiser configuration and incompatible restores fail before GPU construction."""
import json
import pytest
import kindle

CONFIG = dict(action_count=18, batch_size=8, train_batch_size=32, side=64,
              value_bins=255, learning_rate=1e-4, seed=41001, horizon=333)

@pytest.mark.parametrize('field,value', [('action_count',0),('batch_size',0),
    ('train_batch_size',0),('side',48),('value_bins',4),('learning_rate',0),('horizon',1)])
def test_invalid_configuration_without_gpu(field,value):
    with pytest.raises(ValueError):
        kindle.DenoisingWorldLearner(CONFIG | {field:value})

@pytest.mark.parametrize('field', ['format','architecture','sampler','observed_reconstruction'])
def test_incompatible_header_without_gpu(tmp_path,field):
    header=dict(format='kindle-denoising-world-v1',config=CONFIG,updates=0,
        architecture='conditional-unet16-32-64-history4',sampler='edm-euler3-sigma5-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    header[field]='legacy'
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

def test_unknown_setting_is_rejected_without_gpu():
    with pytest.raises(ValueError,match='unknown field'):
        kindle.DenoisingWorldLearner(CONFIG | {'target_conditioning':True})
