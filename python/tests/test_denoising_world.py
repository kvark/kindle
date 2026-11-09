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

@pytest.mark.parametrize('scale', [-1., 1001.])
def test_invalid_context_scale_without_gpu(scale):
    with pytest.raises(ValueError, match='invalid denoising'):
        kindle.DenoisingWorldLearner(CONFIG | {'context_motion_scale':scale})

@pytest.mark.parametrize('version,scale', [('v1',100.),('v2',0.)])
def test_context_objective_requires_matching_header_without_gpu(tmp_path,version,scale):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':scale},updates=0,
        architecture='conditional-unet16-32-64-history4',sampler='edm-euler3-sigma5-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

@pytest.mark.parametrize('version,residual,sampler', [
    ('v1',True,'edm-euler3-sigma5-quantized'),
    ('v2',True,'edm-euler3-sigma5-quantized'),
    ('v3',False,'edm-euler3-sigma5-last-frame-residual-quantized'),
    ('v3',True,'edm-euler3-sigma5-quantized')])
def test_residual_mode_requires_matching_version_and_sampler_without_gpu(tmp_path,version,residual,sampler):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':residual},updates=0,
        architecture='conditional-unet16-32-64-history4',sampler=sampler,
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

@pytest.mark.parametrize('version,half', [('v1',True),('v2',True),('v3',True),('v4',False)])
def test_half_max_noise_requires_matching_version_without_gpu(tmp_path,version,half):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':True,'half_max_noise':half},updates=0,
        architecture='conditional-unet16-32-64-history4',sampler='edm-euler3-sigma5-last-frame-residual-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

@pytest.mark.parametrize('scale', [0., .009, 2.01])
def test_invalid_sigma_data_without_gpu(scale):
    with pytest.raises(ValueError,match='invalid denoising'):
        kindle.DenoisingWorldLearner(CONFIG | {'sigma_data':scale})

@pytest.mark.parametrize('version,scale', [('v1',.05),('v2',.05),('v3',.05),('v4',.05),('v5',.5)])
def test_sigma_data_requires_matching_header_without_gpu(tmp_path,version,scale):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':True,'half_max_noise':True,'sigma_data':scale},updates=0,
        architecture='conditional-unet16-32-64-history4',sampler='edm-euler3-sigma5-last-frame-residual-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

@pytest.mark.parametrize('mean', [-6.01, 1.01])
def test_invalid_log_noise_mean_without_gpu(mean):
    with pytest.raises(ValueError,match='invalid denoising'):
        kindle.DenoisingWorldLearner(CONFIG | {'log_noise_mean':mean})

@pytest.mark.parametrize('version,mean', [(f'v{i}',-3.) for i in range(1,6)]+[('v6',-1.2)])
def test_log_noise_mean_requires_matching_header_without_gpu(tmp_path,version,mean):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':True,
                       'half_max_noise':True,'sigma_data':.05,'log_noise_mean':mean},updates=0,
        architecture='conditional-unet16-32-64-history4',
        sampler='edm-euler3-sigma5-last-frame-residual-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

@pytest.mark.parametrize('limit', [-1.,0.,1000001.])
def test_invalid_reward_limit_without_gpu(limit):
    with pytest.raises(ValueError,match='invalid denoising'):
        kindle.DenoisingWorldLearner(CONFIG | {'reward_limit':limit})

@pytest.mark.parametrize('version,limit',[(f'v{i}',32.) for i in range(1,7)]+[('v7',None)])
def test_reward_limit_requires_matching_header_without_gpu(tmp_path,version,limit):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':True,
                       'half_max_noise':True,'sigma_data':.05,'reward_limit':limit},updates=0,
        architecture='conditional-unet16-32-64-history4',
        sampler='edm-euler3-sigma5-last-frame-residual-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

@pytest.mark.parametrize('field,value', [('format','kindle-denoising-world-v7'),
    ('sampler','edm-euler3-sigma5-last-frame-residual-quantized'),
    ('observed_reconstruction','single-denoise-sigma0.5-not-rssm-posterior'),
    ('deterministic_frame',False)])
def test_direct_frame_rejects_stale_metadata_without_gpu(tmp_path,field,value):
    header=dict(format='kindle-denoising-world-v8',
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':True,
                       'half_max_noise':True,'sigma_data':.05,'reward_limit':32.,
                       'deterministic_frame':True},updates=0,
        architecture='conditional-unet16-32-64-history4',sampler='direct-history4-quantized',
        observed_reconstruction='causal-single-pass-not-observed-denoising')
    if field=='deterministic_frame':header['config'][field]=value
    else:header[field]=value
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))

def test_target_motion_requires_nonzero_scale_without_gpu():
    with pytest.raises(ValueError,match='invalid denoising'):
        kindle.DenoisingWorldLearner(CONFIG | {'target_motion_loss':True})

@pytest.mark.parametrize('version,enabled',[('v7',True),('v8',True),('v9',False)])
def test_target_motion_requires_matching_header_without_gpu(tmp_path,version,enabled):
    header=dict(format='kindle-denoising-world-'+version,
        config=CONFIG | {'context_motion_scale':100.,'residual_frame':True,
                       'half_max_noise':True,'sigma_data':.05,'reward_limit':32.,
                       'target_motion_loss':enabled},updates=0,
        architecture='conditional-unet16-32-64-history4',
        sampler='edm-euler3-sigma5-last-frame-residual-quantized',
        observed_reconstruction='single-denoise-sigma0.5-not-rssm-posterior')
    (tmp_path/'metadata.json').write_text(json.dumps(header))
    with pytest.raises(RuntimeError,match='checkpoint header'):
        kindle.DenoisingWorldLearner.restore(str(tmp_path))
