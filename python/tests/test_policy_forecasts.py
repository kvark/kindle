from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
from probe_policy_forecasts import forecast_targets, replay_reset, reward_metrics, stream_records, summarize


def transition(frame, terminal=False):
    return dict(event='transition', actions=[2], rewards=[25.0], terminated=[terminal],
                truncated=[False], executed_action_frames=[frame])


def test_stream_records_preserve_resets_splits_and_causal_milestone_phase():
    selected = [dict(stream=0, episode=i, terminated=True, truncated=False) for i in range(3)]
    replay = [dict(stream=0, episode=i, first_frame=i*8,
                   task_outcome=dict(first_milestone_frame=4)) for i in range(3)]
    rows = [transition(4), transition(8, True), dict(event='episode', stream=0, episode=0),
            dict(event='reset', streams=[0]), transition(12), transition(16, True),
            dict(event='episode', stream=0, episode=1), dict(event='reset', streams=[0]), transition(20)]
    records, = stream_records(rows, 1, selected, replay)
    assert [r['episode'] for r in records] == [0, 0, 1, 1, 2]
    assert [r['split'] for r in records] == [0, 0, 1, 1, 1]
    assert [r['phase'] for r in records] == [0, 1, 0, 1, 0]
    assert [r['reset_after'] for r in records] == [False, True, False, True, False]


def test_cutoffs_cannot_enter_calibration_or_test():
    with pytest.raises(ValueError, match='natural'):
        stream_records([], 1, [dict(stream=0, episode=0, terminated=False, truncated=True)], [])


@pytest.mark.parametrize('invalid', [dict(event='reset', streams=[0]),
                                   dict(event='episode', stream=0, episode=1), dict(event='learner')])
def test_invalid_source_events_fail(invalid):
    with pytest.raises(ValueError):
        stream_records([transition(4), invalid], 1, [], [])


def example_data():
    count = 48
    data = dict(reward=np.arange(count, dtype=np.float64), episode=np.repeat(np.arange(3), 16),
        terminated=np.arange(count) % 16 == 15, split=np.repeat([0, 1, 1], 16),
        phase=np.tile(np.repeat([0, 1], 8), 3), origins15=np.array([0, 8, 16, 32, 40]))
    for name in ('actual_reward', 'unrelated_reward'):
        data[name+'1'] = data['reward'].copy()
        data[name+'15'] = np.stack([np.arange(i, i+15) for i in data['origins15']])
    for name in ('actual_continue', 'unrelated_continue'):
        data[name+'1'] = (~data['terminated']).astype(float)
        data[name+'15'] = np.ones((5, 15))
    data['posterior_reward'] = data['reward'].copy()
    return data


def test_forecast_alignment_excludes_cross_reset_and_short_tail_without_horizon_change():
    data = example_data()
    result = forecast_targets(data, 15)
    assert result['origins'].tolist() == [0, 16, 32]
    assert result['endpoints'].tolist() == [14, 30, 46]
    assert result['reward'].tolist() == [14, 30, 46]
    assert result['previous_reward'].tolist() == [0, 0, 0]
    assert result['excluded'] == 2
    assert result['reward_sum'].tolist() == [sum(range(i, i+15)) for i in [0, 16, 32]]


def test_one_step_keeps_terminal_endpoint_and_resets_persistence():
    data = example_data()
    result = forecast_targets(data, 1)
    assert result['terminal'].sum() == 3 and result['excluded'] == 0
    assert result['previous_reward'][15:18].tolist() == [14, 0, 16]


def test_calibration_is_not_fitted_to_test_and_sparse_terminal_evidence_is_explicit():
    data = example_data()
    result = summarize([data])
    one = result['1']
    assert one['calibration_reward_mean'] == 7.5
    assert one['calibration_terminal_frequency'] == 1/16
    assert one['calibration_examples'] == 16 and one['test_examples'] == 32
    assert one['slices']['all']['reward']['all']['predictions']['actual_reward']['mae'] == 0
    terminal = one['slices']['all']['terminal']
    assert terminal['count'] == 2 and terminal['limited']
    assert terminal['predictions']['actual']['brier'] == 0
    assert result['15']['excluded_reset_crossings_or_short_tails'] == 2
    assert result['15']['slices']['all']['reward_sum']['all']['predictions']['actual']['rmse'] == 0


def test_empty_event_classes_are_not_successful_zero_error_results():
    result = reward_metrics(np.zeros(3), dict(actual=np.ones(3)))
    assert result['positive'] == dict(count=0, target_mean=None, predictions={})
    assert result['negative'] == dict(count=0, target_mean=None, predictions={})
    assert result['positive_auc']['actual'] is None


def test_missing_calibration_or_test_stops_analysis():
    data = example_data()
    data['split'][:] = -1
    with pytest.raises(ValueError, match='support'):
        summarize([data])


@pytest.mark.parametrize('continuing', [False, True])
def test_recorded_reset_at_stream_end_does_not_open_an_unused_actor_episode(continuing):
    class Environment:
        resets = 0

        def reset(self):
            self.resets += 1
            return 'reset-frame', {}

    class Actor:
        needs_reset = True
        arrivals = []

        def begin_episode(self, frame):
            assert self.needs_reset, 'reset requires an episode boundary'
            self.needs_reset = False
            self.arrivals.append(frame)

    environment, actor = Environment(), Actor()
    replay_reset(environment, actor, continuing=continuing)
    assert environment.resets == 1
    assert actor.arrivals == (['reset-frame'] if continuing else [])
    if not continuing:
        actor.begin_episode('next-stream')
        assert actor.arrivals == ['next-stream']
