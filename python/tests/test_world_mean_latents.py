"""Mean-latent world metadata and incompatible modes reject before GPU."""
import json
import kindle
import pytest


def config():
    c = kindle.default_config(18, 'tiny')
    c.update(observation_kind='rgb32', batch_length=4, world_backprop_length=4,
             world_microbatch_size=c['batch_size'], disagreement_bonus=False,
             world_mean_latents=True)
    for key in ('policy', 'value', 'replay_value'):
        c['loss_scales'][key] = 0.
    return c


@pytest.mark.parametrize('version', [1, 2, 23])
def test_mean_latents_reject_older_headers_before_gpu(tmp_path, version):
    metadata = dict(format=f'kindle-world-only-v{version}', config=config(),
                    pixel_loss='motion_cross_entropy', prediction_length=2, updates=0)
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='mean-latent world requires checkpoint format v24'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective,length', [('squared_error', 2), ('motion_cross_entropy', 0)])
def test_mean_latents_reject_other_objectives_before_gpu(objective, length):
    with pytest.raises(ValueError, match='mean-latent world requires a plain motion RGB suffix'):
        kindle.WorldModelLearner(config(), pixel_loss=objective, prediction_length=length)


def test_mean_latents_default_off():
    assert 'world_mean_latents' not in kindle.default_config(18, 'tiny')
