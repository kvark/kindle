from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
from probe_action_ranking import ranking_metrics, select_origins, validate_ranking_checkpoint_config


def test_model_ties_receive_uniform_outcome_and_half_pairwise_credit():
    result = ranking_metrics([0, 0, 4, 8], [0, 0, 0, 0])
    assert result['chosen_actual'] == result['uniform_actual'] == 3
    assert result['gain_over_uniform'] == 0
    assert result['pairwise_agreement'] == .5 and result['unequal_pairs'] == 5
    assert result['best_action_fraction'] == .25


def test_equal_actual_outcomes_do_not_count_as_discrimination():
    result = ranking_metrics([2, 2, 2], [1, 0, 3])
    assert not result['informative'] and result['pairwise_agreement'] is None
    assert result['regret'] == 0 and result['gain_over_uniform'] == 0


def test_correct_and_reversed_preferences_with_actual_ties():
    for predicted, agreement, regret in [([1, 2, 2], 1, 0), ([3, 1, 1], 0, 5)]:
        result = ranking_metrics([0, 5, 5], predicted)
        assert result['pairwise_agreement'] == agreement
        assert result['regret'] == regret and result['unequal_pairs'] == 2


@pytest.mark.parametrize('truth,prediction', [([1], [1]), ([1,2], [1]),
    ([[1,2]], [[1,2]]), ([1,np.nan], [1,2]), ([1,2], [1,np.inf])])
def test_invalid_candidate_scores_fail(truth, prediction):
    with pytest.raises(ValueError):
        ranking_metrics(truth, prediction)


def test_origin_selection_ignores_rewards_and_excludes_reset_crossings():
    rows = [dict(episode=i//256, split=0 if i<256 else 1,
                 phase=int(i%256>=128), reward=0) for i in range(768)]
    selected = select_origins([rows])
    assert len(selected) == 16
    assert all(r['episode'] in (1,2) and r['index']%16==0 for r in selected)
    assert all(rows[r['index']]['episode']==rows[r['index']+14]['episode'] for r in selected)
    for i,row in enumerate(rows):
        row['reward'] = 999 if i%7==0 else -999
    assert select_origins([rows]) == selected


def test_declared_seed_comparison_preserves_both_original_configs():
    source = dict(seed=10007, observation_kind='rgb64', batch_size=16)
    checkpoint = dict(source, seed=12011)
    validate_ranking_checkpoint_config(source, checkpoint, 12011)
    assert source['seed'] == 10007 and checkpoint['seed'] == 12011
    validate_ranking_checkpoint_config(source, dict(source))


def test_uniform_origins_work_without_a_second_game_milestone():
    rows = [dict(episode=i//256, split=0 if i<256 else 1, phase=0, reward=i)
            for i in range(768)]
    selected = select_origins([rows], 'episode-uniform')
    assert len(selected) == 16 and len({r['index'] for r in selected}) == 16
    assert all(r['episode'] in (1,2) and r['index']%16==0 for r in selected)
    assert all(rows[r['index']]['episode']==rows[r['index']+14]['episode'] for r in selected)
    for row in rows:
        row['reward'] = -999
    assert select_origins([rows], 'episode-uniform') == selected
    with pytest.raises(ValueError):
        select_origins([rows])
    with pytest.raises(ValueError):
        select_origins([rows], 'unknown')


def test_declared_dynamics_comparison_preserves_original_configs():
    source = dict(seed=1009, observation_kind='rgb64', dynamics_learning_rate=4e-4, learning_rate=4e-5)
    checkpoint = dict(source, dynamics_learning_rate=1e-4)
    validate_ranking_checkpoint_config(source, checkpoint, comparison_dynamics_rate=1e-4)
    assert source['dynamics_learning_rate'] == 4e-4
    assert checkpoint['dynamics_learning_rate'] == 1e-4


@pytest.mark.parametrize('rate,change,seed', [
    (None, {}, None), (4e-4, {}, None), (2e-4, {}, None),
    (True, {}, None), (0.0, {}, None), (-1.0, {}, None),
    (float('nan'), {}, None), (float('inf'), {}, None),
    (1e-4, {'learning_rate': 8e-5}, None),
    (1e-4, {'seed': 4013}, None), (1e-4, {'seed': 4013}, 4013),
])
def test_dynamics_comparison_rejects_undeclared_or_additional_changes(rate, change, seed):
    source = dict(seed=1009, observation_kind='rgb64', dynamics_learning_rate=4e-4, learning_rate=4e-5)
    checkpoint = dict(source, dynamics_learning_rate=1e-4, **change)
    with pytest.raises(ValueError):
        validate_ranking_checkpoint_config(source, checkpoint, seed, rate)


@pytest.mark.parametrize('comparison,change', [
    (None, {}), (10007, {}), (12012, {}), (True, {}), (-1, {}), (2**64, {}),
    (12011, {'batch_size': 8}), (12011, {'observation_kind': 'rgb48'}),
    (12011, {'learning_rate': 1e-5}),
])
def test_seed_comparison_rejects_undeclared_seed_or_other_recipe_changes(comparison, change):
    source = dict(seed=10007, observation_kind='rgb64', batch_size=16)
    checkpoint = dict(source, seed=12011, **change)
    with pytest.raises(ValueError):
        validate_ranking_checkpoint_config(source, checkpoint, comparison)
