"""CPU analysis of frozen live-belief values, including the critic's intrinsic target.

Collect with atari_vector.py --evaluate --probe-behavior. Complete natural
episodes provide Monte Carlo targets; unfinished tails and cutoffs are excluded,
not bootstrapped with the critic being tested. No fitting or actor updates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_atari import read_run
from summarize_atari_comparison import cohort


def discounted_returns(rewards, discount):
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all() or not 0 < discount <= 1:
        raise ValueError('finite reward vector and discount in (0, 1] required')
    result = np.empty_like(rewards)
    value = 0.0
    for index in range(len(rewards)-1, -1, -1):
        value = float(rewards[index]) + discount * value
        result[index] = value
    return result


def value_metrics(target, prediction):
    if not len(target):
        return None
    error = prediction-target
    variance = float(np.var(target))
    return dict(count=len(target), target_mean=float(np.mean(target)),
        prediction_mean=float(np.mean(prediction)), bias=float(np.mean(error)),
        mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
        r2=float(1-np.mean(error**2)/variance) if variance else None,
        correlation=float(np.corrcoef(target, prediction)[0, 1])
                    if variance and np.var(prediction) else None)


def episode_samples(rows, selected, discount, outcomes):
    """Align V(s_t), bonus(s_t,a_t), and the next arrival's external reward."""
    count = len(rows[0]['environment_seeds'])
    episodes, ranks, buffers = [0]*count, [0]*count, [[] for _ in range(count)]
    keys = {(row['stream'], row['episode']) for row in selected}
    samples = []
    last_frames = [0]*count
    for row in rows:
        if row['event'] == 'transition':
            for stream in range(count):
                if row['stored_rewards'][stream][1] != 0:
                    raise ValueError('adapter intrinsic rewards are not covered by this diagnostic')
                buffers[stream].append((row['rewards'][stream], row['behavior_intrinsic'][stream],
                                        row['behavior_values'][stream], last_frames[stream]))
                last_frames[stream] = row['executed_action_frames'][stream]
        elif row['event'] == 'episode':
            stream = row['stream']
            if row['episode'] != episodes[stream]:
                raise ValueError('episode ledger changed')
            key = (stream, row['episode'])
            if key in keys:
                if not row['terminated'] or row['truncated']:
                    raise ValueError('complete natural episodes required')
                data = np.asarray(buffers[stream], dtype=np.float64)
                if len(data) != row['episode_length'] or data[:, 0].sum() != row['episode_return']:
                    raise ValueError('episode rewards or length differ')
                outcome = outcomes[key]
                milestone = outcome['task_outcome']['first_milestone_frame']
                phase = (data[:, 3] >= outcome['first_frame']+milestone) if milestone is not None else np.zeros(len(data), bool)
                external = discounted_returns(data[:, 0], discount)
                intrinsic = discounted_returns(data[:, 1], discount)
                samples.append(dict(stream=stream, episode=row['episode'], split=int(ranks[stream] > 0),
                    external=external, intrinsic=intrinsic, target=external+intrinsic,
                    value=data[:, 2], bonus=data[:, 1], phase=phase))
                ranks[stream] += 1
            buffers[stream] = []
            episodes[stream] += 1
    if len(samples) != len(keys) or min(ranks) < 2:
        raise ValueError('missing calibration/test episodes')
    return samples


def summarize(samples):
    calibration = [part for part in samples if part['split'] == 0]
    test = [part for part in samples if part['split'] == 1]
    if not calibration or not test:
        raise ValueError('missing calibration/test split')
    constant = float(np.concatenate([part['target'] for part in calibration]).mean())
    joined = {key: np.concatenate([part[key] for part in test])
              for key in ('target', 'external', 'intrinsic', 'value', 'bonus', 'phase')}
    slices = {}
    for name, mask in (('all', np.ones(len(joined['target']), bool)),
                       ('before_first_pyramid', ~joined['phase']), ('after_first_pyramid', joined['phase'])):
        truth = joined['target'][mask]
        slices[name] = dict(critic=value_metrics(truth, joined['value'][mask]),
            calibration_mean=value_metrics(truth, np.full(len(truth), constant)),
            zero=value_metrics(truth, np.zeros(len(truth))),
            extrinsic_return_mean=float(joined['external'][mask].mean()) if mask.any() else None,
            intrinsic_return_mean=float(joined['intrinsic'][mask].mean()) if mask.any() else None)
    return dict(calibration_episodes=len(calibration), test_episodes=len(test),
        calibration_mean=constant, slices=slices,
        bonus_mean=float(joined['bonus'].mean()), bonus_max=float(joined['bonus'].max()),
        episode_starts=[dict(stream=part['stream'], episode=part['episode'], split=part['split'],
            value=float(part['value'][0]), external_return=float(part['external'][0]),
            intrinsic_return=float(part['intrinsic'][0])) for part in samples])


def analyze(log, replay):
    run = read_run(log, allow_capped_evaluation=True)
    header = run['start']
    if header.get('behavior_probe') != 'live-posterior-v1' or run['accounting']['updates'] != 0:
        raise ValueError('declared frozen live-belief diagnostic required')
    selected = cohort(run['episodes'], natural_only=True)
    if not selected['complete']:
        raise ValueError('incomplete natural cohort')
    rows = [json.loads(line) for line in Path(log).read_text().splitlines()]
    outcomes = {(row['stream'], row['episode']): row for row in replay['episodes']}
    discount = float(np.float32(1)-np.float32(1)/np.float32(header['config']['horizon']))
    samples = episode_samples(rows, selected['selected'], discount, outcomes)
    return dict(discount=discount, actions=run['end']['run_step'], actor_updates=0,
                score=selected['score'], **summarize(samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('replay', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.log, json.loads(args.replay.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
