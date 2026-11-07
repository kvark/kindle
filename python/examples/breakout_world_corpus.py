"""Fixed-corpus native world training; run under gpu_host_guard.py.

Evaluation conditions on recorded future controls. It is retrospective prediction,
not a prospective planner or an Atari policy assessment. Labels are analysis-only.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from breakout_world_model import pixels, require_device, sha, write
from kindle._breakout_world import (
    evaluation_windows, event_metrics, first_contact_window, image_metrics,
    sample_locations, training_windows,
)


HISTORY = 16
HORIZONS = (1, 4, 8, 16, 32)
DRAWS = (83001, 83003, 83009, 83023)
FIELDS = ('rgb', 'action', 'reward', 'terminal', 'position', 'lives', 'frames', 'episode')
LOSS_NAMES = ('total', 'reconstruction', 'dynamics_kl', 'representation_kl', 'reward',
              'continuation', 'replay_value', 'raw_kl', 'future_prediction',
              'pixel_rollout', 'pixel_rollout_valid_fraction')


def load_dataset(root):
    manifest = json.loads((root/'manifest.json').read_text())
    streams = []
    for lane, item in enumerate(manifest['files']):
        path = root/f'lane{lane}.npz'
        if sha(path) != item['sha256']:
            raise ValueError(f'corpus changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            streams.append({key: data[key] for key in FIELDS})
    if len(streams) != 8:
        raise ValueError('eight source streams required')
    return streams, manifest


def choose_evaluation(streams, seed):
    natural = sample_locations(streams, HISTORY+1, 32, np.random.default_rng(seed))
    contacts, endings = [], []
    for lane, stream in enumerate(streams):
        try:
            contacts.append((lane, first_contact_window(stream)))
        except ValueError:
            pass
        endings.extend((lane, int(index)-HISTORY) for index in np.flatnonzero(stream['terminal']) if index >= HISTORY)
    return dict(natural=natural, contacts=contacts, endings=endings)


def evaluate_group(learner, streams, locations, output, name, draws=DRAWS):
    if not locations:
        result = dict(count=0, unsupported=True)
        write(output/f'{name}.json', result)
        return result
    batch_size = learner.config['batch_size']
    rgb_parts, posterior_parts, reward_parts, positive_parts, continuation_parts = (
        [[] for _ in draws] for _ in range(5))
    truth, context, previous, validity, rewards, terminals, positions, lives = ([] for _ in range(8))
    encoded, posterior_features, prior_features = [], [], []
    before = learner.updates
    selected = np.asarray(HORIZONS)-1
    for offset in range(0, len(locations), batch_size):
        actual = locations[offset:offset+batch_size]
        count = len(actual)
        padded = list(actual)+[actual[0]]*(batch_size-count)
        data = evaluation_windows(streams, padded, history=HISTORY, horizon=max(HORIZONS))
        observed = data['rgb'][:HISTORY].astype(np.float32)/255-.5
        truth.append(data['rgb'][HISTORY+selected, :count])
        context.append(data['rgb'][HISTORY-1, :count])
        previous.append(data['rgb'][HISTORY-2, :count])
        validity.append(data['valid'][:, :count])
        rewards.append(data['reward'][HISTORY:, :count])
        terminals.append(data['terminal'][HISTORY:, :count])
        positions.append(data['position'][:, :count])
        lives.append(data['lives'][:, :count])
        for draw, seed in enumerate(draws):
            prediction = learner.predict(pixels(observed), data['action'][:HISTORY].ravel().tolist(),
                data['action'][HISTORY:].ravel().tolist(), seed=seed, include_features=draw == 0)
            rgb = np.frombuffer(prediction['prior_rgb'], '<f4').reshape(max(HORIZONS), batch_size, 3, 64, 64)
            rgb_parts[draw].append(rgb[selected, :count])
            posterior_parts[draw].append(np.frombuffer(prediction['posterior_rgb'], '<f4').reshape(batch_size, 3, 64, 64)[:count])
            reward_parts[draw].append(np.asarray(prediction['reward'], np.float32)[:, :count])
            positive_parts[draw].append(np.asarray(prediction['reward_positive_probability'], np.float32)[:, :count])
            continuation_parts[draw].append(np.asarray(prediction['continuation'], np.float32)[:, :count])
            if draw == 0:
                encoded.append(np.frombuffer(prediction['encoder_features'], '<f4').reshape(batch_size, -1)[:count])
                posterior_features.append(np.frombuffer(prediction['posterior_features'], '<f4').reshape(batch_size, -1)[:count])
                prior_features.append(np.frombuffer(prediction['prior_features'], '<f4').reshape(max(HORIZONS), batch_size, -1)[selected, :count])
    assert learner.updates == before
    arrays = dict(horizons=np.asarray(HORIZONS), locations=np.asarray(locations),
        truth=np.concatenate(truth, axis=1), context=np.concatenate(context), previous_context=np.concatenate(previous),
        valid=np.concatenate(validity, axis=1), target_reward=np.concatenate(rewards, axis=1),
        target_terminal=np.concatenate(terminals, axis=1), position=np.concatenate(positions, axis=1),
        lives=np.concatenate(lives, axis=1), prediction=np.stack([np.concatenate(v, axis=1) for v in rgb_parts]),
        posterior=np.stack([np.concatenate(v) for v in posterior_parts]),
        reward=np.stack([np.concatenate(v, axis=1) for v in reward_parts]),
        reward_positive_probability=np.stack([np.concatenate(v, axis=1) for v in positive_parts]),
        continuation=np.stack([np.concatenate(v, axis=1) for v in continuation_parts]),
        encoder_features=np.concatenate(encoded), posterior_features=np.concatenate(posterior_features),
        prior_features=np.concatenate(prior_features, axis=1))
    for key in ('prediction', 'posterior', 'reward', 'reward_positive_probability', 'continuation',
                'encoder_features', 'posterior_features', 'prior_features'):
        assert np.isfinite(arrays[key]).all(), key
    np.savez_compressed(output/f'{name}.npz', **arrays)
    context = arrays['context'].astype(np.float32)/255-.5
    result = dict(count=len(locations), source_streams=sorted({int(lane) for lane, _ in locations}),
        scope='retrospective recorded-control forecasts; four draws are not independent episodes',
        posterior=[image_metrics(context, image, context) for image in arrays['posterior']], horizons={})
    gamma = float(np.float32(1)-np.float32(1)/np.float32(learner.config['horizon']))
    for index, horizon in enumerate(HORIZONS):
        valid = arrays['valid'][horizon-1]
        target = arrays['truth'][index, valid].astype(np.float32)/255-.5
        actual_reward = arrays['target_reward'][horizon-1, valid]
        predicted_reward = arrays['reward'][:, horizon-1, valid].mean(axis=0)
        probability = arrays['reward_positive_probability'][:, horizon-1, valid].mean(axis=0)
        terminal_probability = 1-np.clip(arrays['continuation'][:, horizon-1, valid].mean(axis=0)/gamma, 0, 1)
        result['horizons'][str(horizon)] = dict(count=int(valid.sum()), excluded=int((~valid).sum()),
            images=[image_metrics(target, image[index, valid], context[valid]) for image in arrays['prediction']],
            persistence=image_metrics(target, context[valid], context[valid]),
            reward_mae=float(np.abs(predicted_reward-actual_reward).mean()) if valid.any() else None,
            rewarded_mae=float(np.abs(predicted_reward[actual_reward > 0]-actual_reward[actual_reward > 0]).mean()) if (actual_reward > 0).any() else None,
            positive_reward=event_metrics(actual_reward > 0, probability),
            terminal=event_metrics(arrays['target_terminal'][horizon-1, valid], terminal_probability))
    write(output/f'{name}.json', result)
    print(json.dumps(dict(event='evaluation', group=name, count=len(locations),
        ball={h:[x['ball_within_1px'] for x in row['images']] for h, row in result['horizons'].items()})), flush=True)
    return result


def train(args, config, fitting, development):
    import kindle
    from safetensors.numpy import load_file
    learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length)
    require_device(learner)
    write(args.output/'actual-config.json', learner.config)
    selections = dict(fitting=choose_evaluation(fitting, 112003), development=choose_evaluation(development, 112019))
    write(args.output/'selections.json', selections)
    learner.save(str(args.output/'initial'))
    rng = np.random.default_rng(config['seed'] ^ 0x636f72707573)
    seen = [np.zeros(len(stream['action']), bool) for stream in fitting]
    total_terminal, prior_terminal, positive_rows, context_reset_rows, source_reset_rows = 0, 0, 0, 0, 0
    native_seconds = 0.
    started = time.monotonic()

    def evaluate(step):
        output = args.output/'evaluations'/str(step)
        output.mkdir(parents=True)
        for split, streams in [('fitting', fitting), ('development', development)]:
            for kind, locations in selections[split].items():
                evaluate_group(learner, streams, locations, output, f'{split}-{kind}')

    evaluate(0)
    with (args.output/'learning.jsonl').open('x') as log:
        for step in range(1, args.steps+1):
            locations = sample_locations(fitting, config['batch_length'], config['batch_size'], rng)
            streams = [fitting[lane] for lane, _ in locations]
            batch = training_windows(streams, [start for _, start in locations], config['batch_length'],
                                     allow_resets=True, retain_initial_targets=True)
            observed = pixels(batch['rgb'].astype(np.float32)/255-.5)
            tick = time.monotonic()
            values = learner.learn(observed, *[batch[key].ravel().tolist() for key in ('action', 'reward', 'terminal')])
            native_seconds += time.monotonic()-tick
            keep = np.cumprod(batch['action'][-args.prediction_length:] != -1, axis=0).astype(bool)
            assert len(values) == len(LOSS_NAMES) and values[10] == float(keep.mean())
            terminals = int(batch['terminal'].sum())
            kept_terminals = int((batch['terminal'][-args.prediction_length:] & keep).sum())
            total_terminal += terminals
            prior_terminal += kept_terminals
            positive_rows += int((batch['reward'] > 0).sum())
            context_reset_rows += int((batch['action'] == -1).sum())
            source_reset_rows += sum(int((stream['action'][start:start+config['batch_length']] == -1).sum())
                                     for stream, (_, start) in zip(streams, locations))
            for lane, start in locations: seen[lane][start:start+config['batch_length']] = True
            log.write(json.dumps(dict(step=step, locations=locations, losses=values,
                terminal_rows=terminals, prior_terminal_rows=kept_terminals, training_seconds=native_seconds))+'\n')
            if step % 128 == 0:
                log.flush()
                print(json.dumps(dict(event='learning', step=step, losses=values, training_seconds=native_seconds,
                    unique_source_arrivals=sum(int(row.sum()) for row in seen))), flush=True)
                require_device(learner)
            if step in (2048, 4096) and step != args.steps:
                learner.save(str(args.output/f'checkpoint-{step}'))
            if step == args.steps: learner.save(str(args.output/'final'))
            if step % args.eval_every == 0 or step == args.steps: evaluate(step)
    final_extra = args.output/'dense-final'
    final_extra.mkdir()
    for label, streams, count, seed in [('fitting', fitting, args.dense_fitting, 182003),
                                        ('development', development, args.dense_development, 182009)]:
        locations = sample_locations(streams, HISTORY+1, count, np.random.default_rng(seed))
        evaluate_group(learner, streams, locations, final_extra, label, draws=DRAWS[:1])
    learner.save(str(args.output/'final-frozen'))
    before, after = (load_file(args.output/phase/'world.safetensors') for phase in ('final', 'final-frozen'))
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
    assert all(np.isfinite(v).all() for v in after.values()) and learner.updates == args.steps
    write(args.output/'result.json', dict(status='complete', updates=args.steps, prediction_length=args.prediction_length,
        training_seconds=native_seconds, total_seconds=time.monotonic()-started, no_behavior=True,
        unique_source_arrivals=sum(int(row.sum()) for row in seen), source_arrivals=sum(len(row) for row in seen),
        replayed_terminal_rows=total_terminal, prior_terminal_rows=prior_terminal,
        positive_reward_rows=positive_rows, context_reset_rows=context_reset_rows,
        source_reset_rows=source_reset_rows, frozen_tensors=len(before),
        world_model_proven=False, untouched_test_scored=False))


def restore_check(args, config, development):
    import kindle
    learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
    require_device(learner)
    selection = json.loads((args.reference/'selections.json').read_text())['development']['natural'][:config['batch_size']]
    evaluate_group(learner, development, selection, args.output, 'restored')
    expected = np.load(args.reference/'evaluations'/str(learner.updates)/'development-natural.npz', allow_pickle=False)
    actual = np.load(args.output/'restored.npz', allow_pickle=False)
    axes = dict(truth=1, context=0, previous_context=0, valid=1, target_reward=1, target_terminal=1,
        position=1, lives=1, prediction=2, posterior=1, reward=2, reward_positive_probability=2,
        continuation=2, encoder_features=0, posterior_features=0, prior_features=1)
    for key, axis in axes.items():
        reference = np.take(expected[key], np.arange(len(selection)), axis=axis)
        assert np.array_equal(reference, actual[key], equal_nan=True), key
    write(args.output/'result.json', dict(status='complete', updates=learner.updates,
        exact_prediction_restore=True, checked_arrays=len(axes), source_origins=len(selection)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'restore-check'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--prediction-length', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=1024)
    parser.add_argument('--dense-fitting', type=int, default=256)
    parser.add_argument('--dense-development', type=int, default=128)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.prediction_length != 16 or min(args.dense_fitting, args.dense_development) < 1:
        parser.error('positive training/evaluation budgets and prediction-length16 required')
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if config['batch_size'] != 8 or config['batch_length'] != 32:
        parser.error('this protocol requires B8/T32')
    fitting, fit_manifest = load_dataset(args.dataset)
    development, dev_manifest = load_dataset(args.development)
    fit_seeds = {row['seed'] for row in fit_manifest['files']}
    assert not fit_seeds.intersection(row['seed'] for row in dev_manifest['files'])
    write(args.output/'declaration.json', dict(mode=args.mode, config=config, loss_names=LOSS_NAMES,
        prediction_length=args.prediction_length, steps=args.steps, eval_every=args.eval_every,
        source_sha256=sha(__file__), fitting_manifest_sha256=sha(args.dataset/'manifest.json'),
        development_manifest_sha256=sha(args.development/'manifest.json'), started_unix=time.time()))
    if args.mode == 'train': train(args, config, fitting, development)
    else:
        if args.checkpoint is None or args.reference is None: parser.error('restore-check needs checkpoint/reference')
        restore_check(args, config, development)


if __name__ == '__main__':
    main()
