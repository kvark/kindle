"""Frozen RGB/CDP state and causal forecast probes; run under gpu_host_guard.py.

Random actions and RAM labels are diagnostics, never policy training. Four native
GPU readouts use training-only normalization and validation-only selection.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import ale_py
import gymnasium as gym
import numpy as np

import kindle
from atari import DreamerAtariPreprocessing, checkpoint_identity, sha256_file
from fit_atari_probes import checked_memory, mlp_probe
from fit_fixed_latents import HEAD_SEED, normalization, predict
from kindle._representation_probe import LABEL_SOURCE, positions, regression_metrics
from kindle._reward_probe import roc_auc
from probe_atari_dynamics import ACTION_SEED_XOR, CONTROL_ACTION_SEED_XOR
from probe_fixed_latents import SPLITS, assert_frozen_tensors


PROTOCOL = 'kindle-cdp-frozen-probes-v1'
TRACE_FIELDS = ('current', 'following', 'actions', 'rewards', 'terminated', 'truncated',
                'collection_cut', 'episodes', 'positions')


def trace_identity(data):
    digest = hashlib.sha256()
    for key in TRACE_FIELDS:
        value = np.ascontiguousarray(data[key])
        digest.update(str((key, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def collect_trace(agent, environment, seed, steps, check_memory, *, cdp, deter=512):
    rng = random.Random(seed ^ ACTION_SEED_XOR)
    controls = random.Random(seed ^ CONTROL_ACTION_SEED_XOR)
    actions = [rng.randrange(environment.action_space.n) for _ in range(steps)]
    cnn, posterior, labels = [], [], []
    current, following, rewards, terminated, truncated, episodes = [], [], [], [], [], []
    forecasts = {h: dict(origins=[], prior=[], unrelated_prior=[], reward=[], unrelated_reward=[],
                        continuation=[], unrelated_continuation=[], predicted=[], unrelated_predicted=[])
                 for h in (1, 15)}
    frames = hashlib.sha256()
    episode, max_error = 0, 0.
    start_updates, start_actions = agent.learner_step, agent.environment_step

    def arrival(frame):
        frames.update(np.asarray(frame).tobytes())
        cnn.append(np.asarray(agent.encoded_observation, np.float32))
        posterior.append(np.asarray(agent.latent_feature, np.float32))
        labels.append(positions('Seaquest', environment.unwrapped.ale.getRAM())[:2])
        return len(cnn) - 1

    frame, _ = environment.reset(seed=seed)
    agent.begin_episode(frame)
    index = arrival(frame)
    check_memory()
    for step, action in enumerate(actions):
        first_prior = None
        if step % 16 == 0:
            horizon = min(15, steps - step)
            sequence = actions[step:step + horizon]
            unrelated = [controls.randrange(environment.action_space.n) for _ in sequence]
            states, predictions = agent.forecast_states(sequence)
            other_states, other_predictions = agent.forecast_states(unrelated)
            reward, continuation, _ = agent.prior_behavior_rollout(sequence)
            other_reward, other_continuation, _ = agent.prior_behavior_rollout(unrelated)
            first_prior = np.asarray(states[0][:deter], np.float32)
            for h, endpoint in ((1, 0), (15, 1)):
                if h > horizon:
                    continue
                row = forecasts[h]
                row['origins'].append(step)
                for key, value in (('prior', states[endpoint]), ('unrelated_prior', other_states[endpoint]),
                                   ('reward', reward[h - 1]), ('unrelated_reward', other_reward[h - 1]),
                                   ('continuation', continuation[h - 1]),
                                   ('unrelated_continuation', other_continuation[h - 1])):
                    row[key].append(value)
                if cdp:
                    row['predicted'].append(predictions[endpoint])
                    row['unrelated_predicted'].append(other_predictions[endpoint])
        mask = [i == action for i in range(environment.action_space.n)]
        if agent.act(action_mask=mask) != action:
            raise RuntimeError('forced action not honored')
        frame, reward, terminal, cutoff, _ = environment.step(action)
        agent.observe(frame, extrinsic_reward=float(reward), terminated=terminal,
                      truncated=cutoff or step + 1 == steps)
        next_index = arrival(frame)
        if first_prior is not None:
            error = float(np.abs(first_prior - posterior[-1][:deter]).max())
            max_error = max(max_error, error)
            if error > 2e-5:
                raise RuntimeError(f'h1 prior/next posterior deterministic mismatch: {error}')
        current.append(index)
        following.append(next_index)
        rewards.append(reward)
        terminated.append(terminal)
        truncated.append(cutoff)
        episodes.append(episode)
        index = next_index
        if (step + 1) % 128 == 0:
            check_memory()
        if (terminal or cutoff) and step + 1 < steps:
            episode += 1
            frame, _ = environment.reset()
            agent.begin_episode(frame)
            index = arrival(frame)
    if agent.learner_step != start_updates or agent.environment_step - start_actions != steps:
        raise RuntimeError('collection changed learner or action accounting')
    data = dict(cnn=np.stack(cnn), posterior=np.stack(posterior), positions=np.stack(labels),
                current=np.asarray(current, np.int32), following=np.asarray(following, np.int32),
                actions=np.asarray(actions, np.int32), rewards=np.asarray(rewards, np.float32),
                terminated=np.asarray(terminated, bool), truncated=np.asarray(truncated, bool),
                collection_cut=np.arange(steps) == steps - 1, episodes=np.asarray(episodes, np.int32))
    for horizon, row in forecasts.items():
        origins = np.asarray(row['origins'], np.int32)
        # Only the real trajectory can determine whether a speculative rollout
        # crossed a reset. Retain terminal endpoints, exclude crossings.
        keep = data['episodes'][origins] == data['episodes'][origins + horizon - 1]
        for key, value in row.items():
            if not cdp and key in ('predicted', 'unrelated_predicted'):
                continue
            data[f'{key}_h{horizon}'] = np.asarray(value, np.int32 if key == 'origins' else np.float32)[keep]
    if any(not np.isfinite(value).all() for key, value in data.items() if key != 'positions'):
        raise RuntimeError('nonfinite captured state/forecast')
    check_memory()
    return data, dict(trace_sha256=trace_identity(data), frames_sha256=frames.hexdigest(),
                      max_h1_deter_error=max_error, actions=steps, arrivals=len(cnn),
                      positive_rewards=int(np.count_nonzero(data['rewards'] > 0)),
                      terminals=int(data['terminated'].sum()), truncations=int(data['truncated'].sum()))


def examples(data, stage, horizon):
    if horizon == 0:
        origins = np.arange(len(data['actions']))
        following = data['following']
        controls = {}
        x = data[stage][following]
    else:
        origins = data[f'origins_h{horizon}']
        following = data['following'][origins + horizon - 1]
        current = data['current'][origins]
        x = data[f'prior_h{horizon}']
        controls = dict(unrelated=data[f'unrelated_prior_h{horizon}'],
                        current_posterior=data['posterior'][current], current_cnn=data['cnn'][current],
                        future_cnn=data['cnn'][following], current_positions=data['positions'][current],
                        reward=data['rewards'][origins + horizon - 1],
                        terminated=data['terminated'][origins + horizon - 1])
        for key in ('reward', 'continuation', 'predicted'):
            if f'{key}_h{horizon}' in data:
                controls[f'forecast_{key}'] = data[f'{key}_h{horizon}']
                controls[f'unrelated_{key}'] = data[f'unrelated_{key}_h{horizon}']
    return dict(x=x, labels=data['positions'][following], origins=origins, **controls)


def load_split(root, files, split, stage, horizon):
    pieces = []
    for row in files:
        if row['split'] != split:
            continue
        with np.load(root / row['file']) as data:
            part = examples(data, stage, horizon)
        part['seeds'] = np.full(len(part['x']), row['seed'], np.int32)
        pieces.append(part)
    return {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}


def report(prediction, data):
    def metrics(p, y):
        return dict(zip(('player_x', 'player_y'), regression_metrics(p, y)))
    return dict(all=metrics(prediction, data['labels']), by_trajectory={str(seed): metrics(
        prediction[data['seeds'] == seed], data['labels'][data['seeds'] == seed]) for seed in np.unique(data['seeds'])})


def cosine_error(prediction, target):
    p, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    return 1 - (p * y).sum(-1) / np.sqrt(np.maximum((p * p).sum(-1), 1e-8)
                                        * np.maximum((y * y).sum(-1), 1e-8))


def spread(features):
    centered = features.astype(np.float64) - features.mean(0, dtype=np.float64)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(centered)), 0)
    total = eigenvalues.sum()
    weights = eigenvalues[eigenvalues > 0] / total if total > 0 else np.array([])
    return dict(mean_feature_std=float(np.sqrt(np.maximum(np.diag(centered.T @ centered / len(centered)), 0)).mean()),
                effective_rank=float(np.exp(-(weights * np.log(weights)).sum())) if len(weights) else 0.,
                centered_variance=float(total), dimensions=features.shape[1])


def forecast_report(data, train_mean):
    def metrics(mask):
        reward = data['reward'][mask]
        result = dict(count=len(reward), positive_rewards=int((reward > 0).sum()),
                      terminals=int(data['terminated'][mask].sum()), reward={})
        for key, prediction in (('prior', data['forecast_reward'][mask]),
                                ('unrelated_actions', data['unrelated_reward'][mask]), ('zero', np.zeros(len(reward)))):
            result['reward'][key] = dict(mae=float(np.abs(prediction - reward).mean()),
                rmse=float(np.sqrt(np.square(prediction - reward).mean())),
                positive_auc=roc_auc((reward > 0).tolist(), prediction.tolist()))
        if 'forecast_predicted' in data:
            target = data['future_cnn'][mask]
            result['cosine_distance'] = {key: float(cosine_error(prediction, target).mean()) for key, prediction in (
                ('prior', data['forecast_predicted'][mask]), ('unrelated_actions', data['unrelated_predicted'][mask]),
                ('persistence', data['current_cnn'][mask]), ('training_mean', np.broadcast_to(train_mean, target.shape)))}
        return result
    return dict(all=metrics(np.ones(len(data['x']), bool)),
                by_trajectory={str(seed): metrics(data['seeds'] == seed) for seed in np.unique(data['seeds'])})


def fit_readouts(root, result, save, *, steps):
    model = kindle._native.RegressionProbe(256, 2, hidden=128, batch=64, seed=HEAD_SEED)
    checked_memory(model)
    posterior_parameters = posterior_norm = train_mean = None
    for stage, horizon in (('cnn', 0), ('posterior', 0), ('prior', 1), ('prior', 15)):
        started = time.monotonic()
        train, val, test = [load_split(root, result['files'], split, stage, horizon) for split in SPLITS]
        norm = normalization(train['x'], train['labels'])
        fitted, info = mlp_probe(train['x'], train['labels'], val['x'], val['labels'], test['x'], HEAD_SEED,
                                 steps=steps, model=model, validation_interval=128)
        parameters = model.parameters()
        training = predict(model, train['x'], norm)
        readouts = dict(fitted=fitted, training_mean=np.broadcast_to(norm['y_mean'], fitted.shape))
        if horizon:
            readouts['unrelated_actions'] = predict(model, test['unrelated'], norm)
        if model.parameters() != parameters:
            raise RuntimeError('readout changed during evaluation')
        head = root / f'head-{stage}-h{horizon}.npz'
        np.savez_compressed(head, **norm, **{f'parameter_{i}': np.asarray(p, np.float32) for i, p in enumerate(parameters)})
        if stage == 'cnn':
            train_mean = train['x'].mean(0, dtype=np.float64)
            result['representation_spread'] = {split: spread(data['x']) for split, data in zip(SPLITS, (train, val, test))}
        if stage == 'posterior':
            posterior_parameters, posterior_norm = parameters, norm
        if horizon:
            model.reset(test['current_posterior'].shape[1], 2, HEAD_SEED)
            model.set_parameters(posterior_parameters)
            readouts['posterior_persistence'] = predict(model, test['current_posterior'], posterior_norm)
            if model.parameters() != posterior_parameters:
                raise RuntimeError('frozen posterior readout changed')
        evidence = root / f'evidence-{stage}-h{horizon}.npz'
        np.savez_compressed(evidence, labels=test['labels'], origins=test['origins'], seeds=test['seeds'], **readouts)
        row = dict(stage=stage, horizon=horizon, fit=info, input_width=train['x'].shape[1],
                   train_examples=len(train['x']), validation_examples=len(val['x']), test_examples=len(test['x']),
                   training=report(training, train), readouts={key: report(value, test) for key, value in readouts.items()},
                   training_normalized_mse=float(np.nanmean(np.nanmean(
                       np.square((training - train['labels']) / norm['y_scale']), axis=0))),
                   head_file=head.name, head_sha256=sha256_file(head), evidence_file=evidence.name,
                   evidence_sha256=sha256_file(evidence), seconds=time.monotonic() - started)
        if horizon:
            row['forecasts'] = forecast_report(test, train_mean)
            # Privileged persistence is a diagnostic ceiling, not an actor readout.
            valid = np.isfinite(test['current_positions']).all(1)
            row['privileged_position_persistence'] = dict(zip(('player_x', 'player_y'),
                regression_metrics(test['current_positions'][valid], test['labels'][valid])))
        result['heads'].append(row)
        save()
        print(json.dumps(dict(fitted=stage, horizon=horizon, selected_step=info['selected_step'])), flush=True)
    result['memory'].append(checked_memory(model))


def run(checkpoint, output, smoke):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    config = agent.config
    if (config['model_size'] != 'size1_m' or config['observation_kind'] != 'rgb64' or config['video_encoder'] is not None
            or config['action_count'] != 18 or config['intrinsic_reward_scale'] != 0 or config['actor_critic_gradient']):
        raise ValueError('requires unassisted Size1M RGB/CDP actor with ac_grads=false')
    cdp = config['loss_scales']['future_prediction'] > 0
    result = dict(protocol=PROTOCOL, status='running', smoke=smoke, method='cdp' if cdp else 'rgb',
                  seed=config['seed'], checkpoint=checkpoint_identity(checkpoint), config=config,
                  source_sha256=sha256_file(__file__), native_sha256=sha256_file(kindle._native.__file__),
                  wrapper_sha256=sha256_file(Path(__file__).with_name('atari.py')), ale_py_version=ale_py.__version__,
                  environment=dict(name='ALE/Seaquest-v5', observation_size='native', sticky_actions=.25,
                                   full_action_space=True, action_repeat=4, noop_max=0, max_episode_frames=100000),
                  gpu_device=agent.gpu_device, label_source=LABEL_SOURCE, labels=['player_x', 'player_y'],
                  labels_limitation='RAM diagnostic coordinates only; no bullet, enemy or full-state claim',
                  steps_per_trajectory=128 if smoke else 4096, files=[], heads=[], memory=[], actor_updates=0)

    def save():
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    def check_memory():
        memory = checked_memory(agent)
        # Keep a bounded summary rather than thousands of redundant rows.
        if not result['memory']:
            result['memory'].append(memory)
        headroom = memory['budget_bytes'] - memory['usage_bytes']
        result['minimum_sampled_headroom_bytes'] = min(headroom, result.get('minimum_sampled_headroom_bytes', headroom))

    check_memory()
    before = agent.learner_step
    gym.register_envs(ale_py)
    env = DreamerAtariPreprocessing(gym.make('ALE/Seaquest-v5', frameskip=1,
        repeat_action_probability=.25, full_action_space=True), noop_max=0,
        max_episode_frames=100000, screen_size=None)
    save()
    try:
        for split, seeds in SPLITS.items():
            for seed in seeds[:1] if smoke else seeds:
                data, row = collect_trace(agent, env, seed, result['steps_per_trajectory'], check_memory, cdp=cdp)
                if data['cnn'].shape[1] != 256 or data['posterior'].shape[1] != 640:
                    raise RuntimeError('unexpected Size1M representation dimensions')
                path = output / f'{split}-{seed}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **data)
                result['files'].append(dict(file=path.name, split=split, seed=seed, sha256=sha256_file(path), **row))
                save()
                print(json.dumps(dict(collected=path.name, actions=row['actions'])), flush=True)
    finally:
        env.close()
    after = output / 'frozen-after'
    agent.save_checkpoint(str(after))
    if agent.learner_step != before:
        raise RuntimeError('frozen actor updated')
    result.update(unchanged_tensor_counts=assert_frozen_tensors(checkpoint, after),
                  frozen_tensor_bytes_unchanged=True, actor_learner_step=before,
                  new_game_actions=sum(row['actions'] for row in result['files']),
                  collection_seconds=time.monotonic() - started)
    del agent
    save()
    fit_readouts(output, result, save, steps=32 if smoke else 2048)
    result.update(status='complete', seconds=time.monotonic() - started)
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.smoke)


if __name__ == '__main__':
    main()
