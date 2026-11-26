"""Loss-only RGB color attention must reject incompatible restores before GPU."""
import json
import kindle
import pytest


def config():
    c = kindle.default_config(18, 'tiny')
    c.update(observation_kind='rgb32', batch_length=4, world_backprop_length=4,
             world_microbatch_size=c['batch_size'], disagreement_bonus=False,
             rgb_color_attention=dict(scale=1000., channel=0, background_scale=1.5,
                                      region=[.0625, .4375, .9375, .875]))
    for key in ('policy', 'value', 'replay_value'):
        c['loss_scales'][key] = 0.
    return c


@pytest.mark.parametrize('version', [1, 2, 22])
def test_color_attention_rejects_older_headers_before_gpu(tmp_path, version):
    metadata = dict(format=f'kindle-world-only-v{version}', config=config(),
                    pixel_loss='motion_cross_entropy', prediction_length=2, updates=0)
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='color attention requires checkpoint format v23'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective,length', [('squared_error', 2), ('motion_cross_entropy', 0)])
def test_color_attention_rejects_other_objectives_before_gpu(objective, length):
    with pytest.raises(ValueError, match='color attention requires a plain categorical motion RGB suffix'):
        kindle.WorldModelLearner(config(), pixel_loss=objective, prediction_length=length)


def test_color_attention_is_absent_by_default():
    assert 'rgb_color_attention' not in kindle.default_config(18, 'tiny')
