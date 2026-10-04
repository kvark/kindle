import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import summarize_atari_comparison as summary
from summarize_cdp import recipe
from summarize_representation_learning import TINY_CHECKPOINTS
from test_cdp_summary import header


def tiny_header():
    value = header('rgb')
    value['environment'] = 'ALE/Freeway-v5'
    value['config']['observation_kind'] = 'features'
    value['config']['loss_scales'].update(reconstruction=0, future_prediction=.25)
    value['model_provenance']['perception'] = dict(kind='levjepa-tiny',
        checkpoint_sha256=TINY_CHECKPOINTS['pretrained_tiny'])
    value['learned_rgb_preprocessing'] = None
    return value


def test_five_game_recipe_is_shared_only_after_declared_method_differences():
    tiny = tiny_header()
    expected = recipe(tiny, 'pretrained_tiny', environment='ALE/Freeway-v5')
    for method in ('cdp', 'rgb'):
        value = header(method)
        value['environment'] = 'ALE/Freeway-v5'
        assert recipe(value, method, environment='ALE/Freeway-v5') == expected


@pytest.mark.parametrize('change', ['untrained', 'joint', 'pixels', 'loss', 'game', 'aid'])
def test_wrong_tiny_or_task_recipe_is_rejected(change):
    value = tiny_header()
    if change == 'untrained':
        value['model_provenance']['perception']['checkpoint_sha256'] = TINY_CHECKPOINTS['initial_tiny']
    elif change == 'joint':
        value['config']['video_encoder'] = 'joint'
    elif change == 'pixels':
        value['learned_rgb_preprocessing'] = 'RGB64 upscale'
    elif change == 'loss':
        value['config']['loss_scales']['future_prediction'] = 500
    elif change == 'game':
        value['environment'] = 'ALE/Venture-v5'
    else:
        value['exploration'] = {'enabled': True}
    with pytest.raises(ValueError):
        recipe(value, 'pretrained_tiny', environment='ALE/Freeway-v5')


def episode(stream, score, *, truncated=False):
    return dict(stream=stream, episode_return=score, terminated=not truncated, truncated=truncated)


def test_frozen_cohort_weights_streams_equally_and_keeps_excess_and_cutoffs():
    rows = [episode(0, 1), episode(0, 3, truncated=True), episode(0, 100),
            episode(1, -1), episode(1, 5)]
    result = summary.cohort(rows, streams=2, target=2)
    assert result['complete'] and result['score'] == 2
    assert len(result['selected']) == 4 and result['excess'] == [rows[2]]
    assert result['natural']['completed_episodes'] == 3
    assert result['truncated']['completed_episodes'] == 1


def test_capped_cohort_never_fabricates_a_complete_score():
    result = summary.cohort([episode(0, 100)], streams=2, target=1)
    assert not result['complete'] and result['score'] is None
    assert result['per_stream_counts'] == [1, 0] and len(result['selected']) == 1


def pair(method, seed, score, *, complete=True):
    return dict(name=f'freeway-{method}-{seed}', game='Freeway', method=method,
                training=dict(seed=seed, curve=[dict(actions=2000, seconds=1, score=score),
                                               dict(actions=200000, seconds=10, score=score)],
                              final=dict(elapsed_seconds=10)),
                evaluation=dict(cohort=dict(complete=complete, score=score if complete else None)))


def test_partial_seeds_and_capped_evaluations_never_become_complete_evidence():
    rows = [pair(method, seed, 10 if method == 'cdp' else 4)
            for method in summary.METHODS for seed in summary.SEEDS]
    partial = summary.aggregate(rows[:2])
    assert partial['completed_pairs'] == 2 and partial['status'] == 'partial'
    assert partial['paired'] == [] and all(g['aggregate'] is None for g in partial['results'])
    complete = summary.aggregate(rows)
    assert complete['status'] == 'partial' and complete['completed_pairs'] == 9
    assert len(complete['paired']) == 2
    assert all(p['online_difference']['mean'] == p['frozen_difference']['mean'] == 6 for p in complete['paired'])
    capped = copy.deepcopy(rows)
    capped[0]['evaluation']['cohort'] = dict(complete=False, score=None)
    result = summary.aggregate(capped)
    assert all(p['frozen_difference'] is None for p in result['paired'])
    assert result['results'][0]['frozen_score'] is None
    with pytest.raises(ValueError, match='duplicate'):
        summary.aggregate(rows + [rows[0]])


def test_only_original_five_games_and_declared_seeds_are_admitted():
    assert summary.run_name('Qbert', 'cdp', 3019) == 'qbert-cdp-3019'
    for selection in [('Venture', 'cdp', 1009), ('Pong', 'joint_tiny', 1009), ('Pong', 'rgb', 42)]:
        with pytest.raises(ValueError):
            summary.run_name(*selection)
