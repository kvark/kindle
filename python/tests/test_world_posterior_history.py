"""Explicit posterior-history metadata and report math, before GPU construction."""
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
    return dict(format='kindle-world-only-v21', config=c, prediction_length=16,
                pixel_loss='motion_posterior_history', updates=0)


@pytest.mark.parametrize('version', [18, 20])
def test_posterior_history_rejects_legacy_headers_before_gpu(tmp_path, version):
    m = metadata()
    m['format'] = f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='posterior history requires checkpoint format v21'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective,length,reason', [
    ('motion_event_rollout', 16, 'posterior history requires checkpoint format v21'),
    ('motion_cross_entropy', 0, 'posterior history requires checkpoint format v21'),
    ('motion_posterior_history', 0, 'invalid spatial action checkpoint objective or geometry'),
])
def test_posterior_history_rejects_changed_objective_before_gpu(tmp_path, objective, length, reason):
    m = metadata()
    m.update(pixel_loss=objective, prediction_length=length)
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match=reason):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_posterior_history_constructor_rejects_h0_before_gpu():
    with pytest.raises(ValueError, match='posterior history requires local action'):
        kindle.WorldModelLearner(metadata()['config'], pixel_loss='motion_posterior_history')


def test_posterior_history_reports_weight_only_the_extra_rgb_term():
    import breakout_world_corpus as corpus
    names = corpus.report_names('motion_posterior_history', 16)
    assert len(names) == 11 and names[9:] == ('posterior_history_reconstruction', 'posterior_history_valid_fraction')
    scales = dict(reconstruction=2., dynamics=3., representation=.1, reward=4., continuation=5.)
    values = list(range(11))
    assert corpus.reported_total(dict(loss_scales=scales), values, 'motion_posterior_history', 16) == 2*(1+9)+3*2+.1*3+4*4+5*5
    with pytest.raises(ValueError, match='positive suffix'):
        corpus.report_names('motion_posterior_history', 0)
