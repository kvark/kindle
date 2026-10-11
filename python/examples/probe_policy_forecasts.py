"""Frozen reward/terminal forecasts on recorded policy trajectories, not new acting.

Run capture under gpu_host_guard.py. Scalar re-observation does not reproduce the
original vector actor's categorical draws. RAM/task labels are analysis-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from atari import checkpoint_identity, sha256_file
from audit_atari import read_run
from kindle._reward_probe import roc_auc
from replay_atari import verify_replay_identity
from summarize_atari_comparison import cohort


PROTOCOL = 'kindle-policy-forecasts-v1'
FIELDS = ('action', 'reward', 'terminated', 'truncated', 'frames', 'episode', 'split', 'phase', 'reset_after')


def stream_records(rows, count, selected, replayed):
    """Split a vector ledger without merging histories or selecting cutoffs."""
    splits, ranks = {}, [0] * count
    for row in selected:
        if not row['terminated'] or row['truncated']:
            raise ValueError('calibration/test episodes must be natural')
        stream = row['stream']
        splits[stream, row['episode']] = int(ranks[stream] > 0)
        ranks[stream] += 1
    outcomes = {(r['stream'], r['episode']): r for r in replayed}
    streams, episodes = [[] for _ in range(count)], [0] * count
    for row in rows:
        if row['event'] == 'transition':
            for stream, records in enumerate(streams):
                episode = episodes[stream]
                previous_frame = records[-1]['frames'] if records else 0
                outcome = outcomes.get((stream, episode))
                milestone = outcome['task_outcome']['first_milestone_frame'] if outcome else None
                phase = int(milestone is not None and previous_frame >= outcome['first_frame'] + milestone)
                records.append(dict(action=row['actions'][stream], reward=row['rewards'][stream],
                    terminated=row['terminated'][stream], truncated=row['truncated'][stream],
                    frames=row['executed_action_frames'][stream], episode=episode,
                    split=splits.get((stream, episode), -1), phase=phase, reset_after=False))
        elif row['event'] == 'episode':
            if row['episode'] != episodes[row['stream']]:
                raise ValueError('episode ledger changed')
        elif row['event'] == 'reset':
            for stream in row['streams']:
                last = streams[stream][-1]
                if not (last['terminated'] or last['truncated']):
                    raise ValueError('reset without a boundary')
                last['reset_after'] = True
                episodes[stream] += 1
        elif row['event'] == 'learner':
            raise ValueError('source must be frozen')
    return streams


def forecast_targets(data, horizon):
    origins = np.arange(len(data['reward'])) if horizon == 1 else data['origins15']
    endpoints = origins + horizon - 1
    bounded = endpoints < len(data['reward'])
    selected = np.flatnonzero(bounded)
    same = data['episode'][origins[selected]] == data['episode'][endpoints[selected]]
    selected = selected[same]
    origins, endpoints = origins[selected], endpoints[selected]
    return dict(rows=selected, origins=origins, endpoints=endpoints,
        reward=data['reward'][endpoints], terminal=data['terminated'][endpoints],
        reward_sum=np.array([data['reward'][i:j+1].sum() for i, j in zip(origins, endpoints)]),
        previous_reward=np.array([data['reward'][i-1] if i and data['episode'][i-1] == data['episode'][i]
                                  else 0.0 for i in origins]),
        split=data['split'][origins], phase=data['phase'][origins],
        excluded=int(len(bounded)-len(selected)))


def reward_metrics(target, predictions):
    result = {}
    for label, mask in (('all', np.ones(len(target), bool)), ('positive', target > 0),
                        ('zero', target == 0), ('negative', target < 0)):
        result[label] = dict(count=int(mask.sum()), target_mean=float(target[mask].mean()) if mask.any() else None,
            predictions={name: dict(mae=float(np.abs(value[mask]-target[mask]).mean()),
                rmse=float(np.sqrt(np.square(value[mask]-target[mask]).mean())))
                for name, value in predictions.items()} if mask.any() else {})
    result['positive_auc'] = {name: roc_auc((target > 0).tolist(), values.tolist())
                              for name, values in predictions.items()}
    return result


def summarize(files):
    result = {}
    for horizon in (1, 15):
        parts = []
        excluded = 0
        for data in files:
            target = forecast_targets(data, horizon)
            excluded += target.pop('excluded')
            rows, origins, endpoints = (target.pop(k) for k in ('rows', 'origins', 'endpoints'))
            suffix = '1' if horizon == 1 else '15'
            for name in ('actual_reward', 'unrelated_reward', 'actual_continue', 'unrelated_continue'):
                values = data[name+suffix][rows]
                target[name] = values if horizon == 1 else values[:, -1]
            if horizon == 15:
                target['actual_sum'] = data['actual_reward15'][rows].sum(axis=1)
                target['unrelated_sum'] = data['unrelated_reward15'][rows].sum(axis=1)
            target['posterior_reward'] = data['posterior_reward'][endpoints]
            parts.append(target)
        joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        calibration, test = joined['split'] == 0, joined['split'] == 1
        if not calibration.any() or not test.any():
            raise ValueError('missing calibration/test support')
        reward_mean = float(joined['reward'][calibration].mean())
        terminal_mean = float(joined['terminal'][calibration].mean())
        slices = {}
        for name, selected in (('all', test), ('before_first_pyramid', test & (joined['phase'] == 0)),
                               ('after_first_pyramid', test & (joined['phase'] == 1))):
            truth = joined['reward'][selected]
            predictions = {key: joined[key][selected] for key in ('actual_reward', 'unrelated_reward', 'posterior_reward')}
            predictions.update(zero=np.zeros(len(truth)), calibration_mean=np.full(len(truth), reward_mean),
                               persistence=joined['previous_reward'][selected])
            death = joined['terminal'][selected]
            risks = dict(actual=1-joined['actual_continue'][selected], unrelated=1-joined['unrelated_continue'][selected],
                         calibration_mean=np.full(len(death), terminal_mean), always_continuing=np.zeros(len(death)))
            row = dict(reward=reward_metrics(truth, predictions), terminal=dict(count=int(death.sum()),
                examples=len(death), limited=int(death.sum()) < 20,
                predictions={key: dict(brier=float(np.square(value-death).mean()) if len(death) else None,
                                       auc=roc_auc(death.tolist(), value.tolist())) for key, value in risks.items()}))
            if horizon == 15:
                row['reward_sum'] = reward_metrics(joined['reward_sum'][selected],
                    dict(actual=joined['actual_sum'][selected], unrelated=joined['unrelated_sum'][selected],
                         zero=np.zeros(len(truth)), persistence=15*joined['previous_reward'][selected],
                         calibration_mean=np.full(len(truth), joined['reward_sum'][calibration].mean())))
            slices[name] = row
        result[str(horizon)] = dict(calibration_examples=int(calibration.sum()), test_examples=int(test.sum()),
            excluded_reset_crossings_or_short_tails=excluded, calibration_reward_mean=reward_mean,
            calibration_terminal_frequency=terminal_mean, slices=slices)
    return result


def replay_reset(environment, agent, *, continuing):
    """Preserve real reset accounting without opening an unused actor episode."""
    frame, _ = environment.reset()
    if continuing:
        agent.begin_episode(frame)


def capture(log, checkpoint, manifest, replay_file, output, smoke):
    import ale_py
    import gymnasium as gym
    import kindle
    from atari import DreamerAtariPreprocessing
    from check_atari_adapter import rom_identity
    from fit_atari_probes import checked_memory
    from probe_fixed_latents import assert_frozen_tensors

    output.mkdir(exist_ok=False)
    source = read_run(log, allow_capped_evaluation=True)
    header = source['start']
    selected = cohort(source['episodes'], natural_only=True)
    replay = json.loads(replay_file.read_text())
    if not selected['complete'] or source['accounting']['updates'] or replay['source_log_sha256'] != source['sha256']:
        raise ValueError('requires completed frozen source and its matching full replay')
    gym.register_envs(ale_py)
    verify_replay_identity(header, json.loads(manifest.read_text()), rom_identity(header['environment']))
    with log.open() as stream:
        streams = stream_records(map(json.loads, stream), header['num_envs'], selected['selected'], replay['episodes'])
    expected_actions = 128 if smoke else source['end']['run_step']
    if sum(map(len, streams)) != source['end']['run_step']:
        raise ValueError('incomplete action ledger')
    started = time.monotonic()
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    if agent.config != header['config'] or agent.config['observation_kind'] != 'rgb64':
        raise ValueError('checkpoint configuration differs from source policy')
    initial_updates, initial_actions = agent.learner_step, agent.environment_step
    result = dict(protocol=PROTOCOL, status='running', smoke=smoke, checkpoint=checkpoint_identity(checkpoint),
        source_log=str(log), source_log_sha256=source['sha256'], replay_sha256=sha256_file(replay_file),
        environment=header['environment'], seed=header['config']['seed'], unrelated_seed_xor=0xF04ECA57,
        horizons=[1, 15], long_horizon_stride=16, calibration='first selected natural episode per stream',
        native_sha256=sha256_file(kindle._native.__file__), source_sha256=sha256_file(__file__),
        files=[], diagnostic_actions=0, actor_updates=0, collection_cuts=0, prefix_checks=0,
        belief='scalar frozen re-observation; not original vector categorical draws', gpu_device=agent.gpu_device)

    def save():
        (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    def memory_check():
        memory = checked_memory(agent)
        headroom = memory['budget_bytes']-memory['usage_bytes']
        result['minimum_sampled_headroom_bytes'] = min(headroom, result.get('minimum_sampled_headroom_bytes', headroom))

    class FrameHash(gym.Wrapper):
        def __init__(self, environment):
            super().__init__(environment)
            self.digest = hashlib.sha256()
        def step(self, action):
            row = self.env.step(action)
            self.digest.update(row[0].tobytes(order='C'))
            return row

    memory_check()
    save()
    try:
        for stream, records in enumerate(streams[:1] if smoke else streams):
            records = records[:128] if smoke else records
            frame_hash = FrameHash(gym.make(header['environment'], frameskip=1,
                repeat_action_probability=header['sticky_actions'], full_action_space=header['full_action_space']))
            environment = DreamerAtariPreprocessing(frame_hash, noop_max=0, max_episode_frames=header['max_episode_frames'], screen_size=None)
            data = {key: np.array([row[key] for row in records]) for key in FIELDS}
            for key in ('actual_reward1', 'unrelated_reward1', 'actual_continue1', 'unrelated_continue1', 'posterior_reward'):
                data[key] = np.zeros(len(records), np.float64)
            long = {key: [] for key in ('origins15', 'actual_reward15', 'unrelated_reward15', 'actual_continue15', 'unrelated_continue15')}
            control = random.Random(header['environment_seeds'][stream] ^ 0xF04ECA57)
            unrelated = [control.randrange(header['config']['action_count']) for _ in records]
            frame, _ = environment.reset(seed=header['environment_seeds'][stream])
            agent.begin_episode(frame)
            collection_started = time.monotonic()
            try:
                for index, row in enumerate(records):
                    horizon = 15 if index % 16 == 0 and index+15 <= len(records) else 1
                    actions = data['action'][index:index+horizon].tolist()
                    before = (agent.latent_feature, agent.posterior_action_probabilities(), agent.environment_step, agent.learner_step) if index == 0 else None
                    rewards, continuations, _ = agent.prior_behavior_rollout(actions)
                    other_rewards, other_continuations, _ = agent.prior_behavior_rollout(unrelated[index:index+horizon])
                    if before is not None:
                        one = agent.prior_behavior_rollout(actions[:1])
                        again = agent.prior_behavior_rollout(actions)
                        assert one[0][0] == rewards[0] and one[1][0] == continuations[0]
                        assert again[0] == rewards and again[1] == continuations
                        assert before == (agent.latent_feature, agent.posterior_action_probabilities(), agent.environment_step, agent.learner_step)
                        result['prefix_checks'] += 1
                    if any(len(values) != horizon for values in (rewards, continuations, other_rewards, other_continuations)):
                        raise RuntimeError('incomplete prior prediction')
                    if not np.isfinite([rewards, continuations, other_rewards, other_continuations]).all():
                        raise RuntimeError('nonfinite prior prediction')
                    if not all(0 <= value <= 1 for value in continuations+other_continuations):
                        raise RuntimeError('invalid continuation probability')
                    for key, values in (('actual_reward', rewards), ('unrelated_reward', other_rewards),
                                        ('actual_continue', continuations), ('unrelated_continue', other_continuations)):
                        data[key+'1'][index] = values[0]
                        if horizon == 15:
                            long[key+'15'].append(values)
                    if horizon == 15:
                        long['origins15'].append(index)
                    action = int(row['action'])
                    if agent.act(action_mask=[i == action for i in range(header['config']['action_count'])]) != action:
                        raise RuntimeError('forced source action changed')
                    frame, reward, terminal, cutoff, _ = environment.step(action)
                    if (reward, terminal, cutoff, environment.executed_action_frames) != (row['reward'], row['terminated'], row['truncated'], row['frames']):
                        raise RuntimeError(f'replay differs at stream={stream}, action={index}')
                    cut = index+1 == len(records) and not terminal and not cutoff
                    agent.observe(frame, extrinsic_reward=float(reward), terminated=terminal, truncated=cutoff or cut)
                    data['posterior_reward'][index] = agent.posterior_reward_prediction()
                    result['diagnostic_actions'] += 1
                    result['collection_cuts'] += int(cut)
                    if row['reset_after']:
                        replay_reset(environment, agent, continuing=index+1 < len(records))
                    if index % 256 == 0:
                        memory_check()
                if not smoke:
                    assert environment.executed_action_frames == source['end']['executed_action_frames'][stream]
                    assert environment.emulator_resets == source['end']['emulator_resets'][stream]
                    if stream == 0:
                        assert frame_hash.digest.hexdigest() == replay['video']['raw_frames_sha256']
            finally:
                environment.close()
            data.update({key: np.asarray(value, np.int32 if key == 'origins15' else np.float64) for key, value in long.items()})
            if any(not np.isfinite(value).all() for value in data.values()):
                raise RuntimeError('nonfinite capture')
            path = output/f'stream-{stream}.npz'
            with path.open('xb') as file:
                np.savez_compressed(file, **data)
            result['files'].append(dict(path=str(path), sha256=sha256_file(path), stream=stream, actions=len(records),
                seconds=time.monotonic()-collection_started, raw_frames_sha256=frame_hash.digest.hexdigest()))
            save()
            print(json.dumps(dict(stream=stream, diagnostic_actions=result['diagnostic_actions'])), flush=True)
        after = output/'frozen-after'
        agent.save_checkpoint(str(after))
        result['unchanged_tensors'] = assert_frozen_tensors(checkpoint, after)
        assert sum(result['unchanged_tensors'].values()) == 346
        assert agent.learner_step == initial_updates and agent.environment_step-initial_actions == expected_actions
        assert result['diagnostic_actions'] == expected_actions
        memory_check()
        result.update(status='complete', seconds=time.monotonic()-started)
        save()
        print(json.dumps(dict(status=result['status'], actions=expected_actions, seconds=result['seconds'])), flush=True)
    except BaseException as error:
        result.update(status='failed', reason=str(error), seconds=time.monotonic()-started)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('replay', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    capture(args.log, args.checkpoint, args.manifest, args.replay, args.output, args.smoke)
