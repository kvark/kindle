"""Suffix-event objectives reject incompatible headers before any GPU creation."""
import json
from pathlib import Path
import sys

import kindle
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))


def metadata():
    config = kindle.default_config(18, 'tiny')
    config.update(observation_kind='rgb32', rgb_spatial_decoder=True, rgb_spatial_fusion=True,
                  world_backprop_length=config['batch_length'], world_microbatch_size=config['batch_size'],
                  disagreement_bonus=False)
    for name in ('policy', 'value', 'replay_value'):
        config['loss_scales'][name] = 0.
    return dict(format='kindle-world-only-v13', config=config, prediction_length=16,
                pixel_loss='motion_event_rollout', updates=0)


@pytest.mark.parametrize('version', range(1, 13))
def test_suffix_event_objective_requires_v13(tmp_path, version):
    value = metadata()
    value['format'] = f'kindle-world-only-v{version}'
    (tmp_path / 'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='suffix event objective requires checkpoint format v13'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field', ['rgb_spatial_decoder', 'rgb_spatial_fusion'])
def test_suffix_event_header_requires_fusion(tmp_path, field):
    value = metadata()
    value['config'][field] = False
    (tmp_path / 'metadata.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='invalid suffix event|invalid spatial fusion'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_suffix_events_require_positive_length_and_fusion():
    with pytest.raises(ValueError, match='positive prediction length'):
        kindle.WorldModelLearner(metadata()['config'], pixel_loss='motion_event_rollout')
    config = metadata()['config']
    config['rgb_spatial_fusion'] = False
    with pytest.raises(ValueError, match='suffix event objective requires spatial fusion'):
        kindle.WorldModelLearner(config, prediction_length=16, pixel_loss='motion_event_rollout')


def test_suffix_event_report_order_and_total_preserve_the_original_terms():
    import breakout_world_corpus as corpus
    names = corpus.report_names('motion_event_rollout', 16)
    assert len(names) == 13 and names[-2:] == ('rollout_future_reward', 'rollout_future_continuation')
    scales = dict(reconstruction=100, dynamics=1, representation=.1, reward=2, continuation=3)
    values = list(range(13))
    assert corpus.reported_total(dict(loss_scales=scales), values, 'motion_event_rollout', 16) == 100*(1+9)+2+.1*3+2*(4+11)+3*(5+12)
    with pytest.raises(ValueError, match='positive suffix'):
        corpus.report_names('motion_event_rollout', 0)
