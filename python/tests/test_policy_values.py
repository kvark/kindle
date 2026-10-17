import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from probe_policy_values import discounted_returns, episode_samples, summarize, value_metrics
from kindle._vector_audit import audit
from test_vector import fixture_events, write_log


def test_discounted_targets_include_arrival_reward_without_bootstrapping():
    assert discounted_returns([2, 4, 8], .5).tolist() == [6, 8, 8]
    for rewards, discount in (([np.nan], .5), ([[1]], .5), ([1], 0), ([1], np.nan)):
        with pytest.raises(ValueError):
            discounted_returns(rewards, discount)


def test_live_value_episode_alignment_and_held_out_constant():
    rows = [dict(event='run_start', environment_seeds=[0])]
    selected, outcomes = [], {}
    for episode in range(3):
        for step in range(2):
            rows.append(dict(event='transition', rewards=[2+step], stored_rewards=[[2+step, 0]],
                behavior_intrinsic=[.1*(step+1)], behavior_values=[10+step],
                executed_action_frames=[episode*8+(step+1)*4]))
        row = dict(event='episode', stream=0, episode=episode, episode_length=2,
                   episode_return=5, terminated=True, truncated=False)
        rows.append(row)
        selected.append(row)
        outcomes[0, episode] = dict(first_frame=episode*8, task_outcome=dict(first_milestone_frame=4))
    rows.append(dict(event='transition', rewards=[999], stored_rewards=[[999, 0]],
        behavior_intrinsic=[100], behavior_values=[999], executed_action_frames=[28]))
    samples = episode_samples(rows, selected, .5, outcomes)
    assert [s['split'] for s in samples] == [0, 1, 1]
    assert samples[0]['target'].tolist() == [3.7, 3.2]
    assert samples[0]['phase'].tolist() == [False, True]
    result = summarize(samples)
    assert result['calibration_mean'] == pytest.approx(3.45)
    assert result['slices']['all']['critic']['count'] == 4
    assert result['slices']['all']['critic']['bias'] == pytest.approx(7.05)
    assert result['bonus_max'] == .2
    assert value_metrics(np.array([]), np.array([])) is None
    selected[0]['truncated'] = True
    with pytest.raises(ValueError, match='natural'):
        episode_samples(rows, selected, .5, outcomes)


def frozen_probe_events():
    rows = [r for r in fixture_events() if r['event'] != 'learner']
    rows[0].update(mode='evaluate_sample', behavior_probe='live-posterior-v1')
    rows[-1].update(learner_step=0, learner_updates=0)
    for row in rows:
        if row['event'] == 'transition':
            row.update(behavior_values=[10, 20], behavior_intrinsic=[.1, .2])
    return rows


def test_probe_ledger_is_frozen_and_batched(tmp_path):
    assert audit(write_log(tmp_path, frozen_probe_events()))['updates'] == 0


@pytest.mark.parametrize('mutation', [
    lambda rows: rows[0].update(mode='train'),
    lambda rows: rows[0].pop('behavior_probe'),
    lambda rows: rows[1].update(behavior_values=[1]),
    lambda rows: rows[1].update(behavior_intrinsic=[-1, 0]),
    lambda rows: rows[1].update(behavior_values=[None, 0]),
    lambda rows: rows[1].update(behavior_intrinsic=[float('nan'), 0]),
])
def test_probe_ledger_rejects_unreliable_values(tmp_path, mutation):
    rows = frozen_probe_events()
    mutation(rows)
    with pytest.raises(ValueError):
        audit(write_log(tmp_path, rows))
