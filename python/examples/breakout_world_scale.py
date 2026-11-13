"""Breakout world learning at corpus scale.

`collect` records eight lanes of GPU-resized 64x64 frames without native copies,
so corpora of hundreds of thousands of arrivals fit in memory. `train` fits the
standalone world learner on such a corpus and scores held-out forecasts with the
existing physical metrics. Run GPU modes under gpu_host_guard.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kindle._breakout_world import evaluation_windows, event_metrics, image_metrics, sample_locations, training_windows

FIELDS = ('rgb', 'action', 'reward', 'terminal', 'position', 'lives', 'frames', 'episode')
CHECKPOINT_DIVISOR = 4


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def pixels(values):
    return np.asarray(values, dtype='<f4').tobytes()


class Lane:
    """Preallocated arrival storage for one environment stream."""

    def __init__(self, capacity, side):
        self.rgb = np.zeros((capacity, 3, side, side), np.uint8)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float64)
        self.terminal = np.zeros(capacity, bool)
        self.position = np.zeros((capacity, 3), np.float32)
        self.lives = np.zeros(capacity, np.int64)
        self.frames = np.zeros(capacity, np.int64)
        self.episode = np.zeros(capacity, np.int64)
        self.count = 0

    def append(self, **values):
        if self.count == len(self.action):
            for key in FIELDS:
                array = getattr(self, key)
                setattr(self, key, np.concatenate([array, np.zeros_like(array)]))
        for key, value in values.items():
            getattr(self, key)[self.count] = value
        self.count += 1

    def arrays(self):
        return {key: getattr(self, key)[:self.count] for key in FIELDS}


def collect(args):
    import ale_py
    import gymnasium as gym
    import kindle
    from atari import DreamerAtariPreprocessing
    from breakout_world_model import require_device
    from kindle._breakout_guidance import FeedbackPixelPolicy
    from kindle._representation_probe import positions
    gym.register_envs(ale_py)
    config = json.loads(args.config.read_text())
    lanes, side = config['batch_size'], frame_side(config)
    learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length, pixel_loss=args.pixel_loss)
    require_device(learner)
    environments = [DreamerAtariPreprocessing(gym.make('ALE/Breakout-v5', frameskip=1,
        repeat_action_probability=args.sticky, full_action_space=True), noop_max=0, screen_size=None,
        max_episode_frames=108000) for _ in range(lanes)]
    teachers = [FeedbackPixelPolicy() for _ in range(lanes)]
    rngs = [np.random.default_rng(args.seed + lane * 1009 + 0x17a) for lane in range(lanes)]
    streams = [Lane(args.ticks + 1 + args.ticks // 16, side) for _ in range(lanes)]
    episodes = np.zeros(lanes, np.int64)
    blocks, command = np.zeros(lanes, int), np.zeros(lanes, int)
    frames = [environment.reset(seed=args.seed + lane * 1009)[0] for lane, environment in enumerate(environments)]
    started = time.monotonic()

    def append(indices, actions, rewards, terminals):
        rgb = np.frombuffer(learner.preprocess(frames), '<f4').reshape(lanes, 3, side, side)
        rgb8 = np.rint((rgb + .5) * 255).clip(0, 255).astype(np.uint8)
        for lane in indices:
            environment = environments[lane]
            streams[lane].append(rgb=rgb8[lane], action=actions[lane], reward=rewards[lane], terminal=terminals[lane],
                                 position=positions('Breakout', environment.unwrapped.ale.getRAM()),
                                 lives=int(environment.unwrapped.ale.lives()), frames=environment.executed_action_frames,
                                 episode=int(episodes[lane]))

    append(range(lanes), [-1] * lanes, [0.] * lanes, [False] * lanes)
    reset = []
    for tick in range(args.ticks):
        if reset:
            for lane in reset:
                frames[lane] = environments[lane].reset()[0]
                teachers[lane] = FeedbackPixelPolicy()
                episodes[lane] += 1
                blocks[lane] = 0
            append(reset, [-1] * lanes, [0.] * lanes, [False] * lanes)
        actions = []
        for lane, (teacher, rng) in enumerate(zip(teachers, rngs)):
            proposed = teacher.propose(frames[lane])
            if blocks[lane] == 0 and rng.random() < args.block_probability:
                blocks[lane], command[lane] = int(rng.integers(4, 13)), int(rng.integers(18))
            action = int(command[lane]) if blocks[lane] else int(proposed)
            if blocks[lane]:
                blocks[lane] -= 1
            teacher.feedback(action)
            actions.append(action)
        transitions = [environment.step(action) for environment, action in zip(environments, actions)]
        frames = [row[0] for row in transitions]
        rewards = [float(row[1]) for row in transitions]
        terminals = [bool(row[2]) for row in transitions]
        assert not any(row[3] for row in transitions), 'unexpected collection time limit'
        append(range(lanes), actions, rewards, terminals)
        reset = [lane for lane, terminal in enumerate(terminals) if terminal]
        if (tick + 1) % 1024 == 0:
            require_device(learner)
            print(json.dumps(dict(event='collection', ticks=tick + 1, actions=lanes * (tick + 1),
                                  seconds=time.monotonic() - started)), flush=True)
    files, action_counts = [], np.zeros(18, int)
    for lane, stream in enumerate(streams):
        data = stream.arrays()
        action_counts += np.bincount(data['action'][data['action'] >= 0], minlength=18)
        path = args.output / f'lane{lane}.npz'
        np.savez(path, **data)
        files.append(dict(path=str(path), sha256=sha(path), seed=args.seed + lane * 1009, arrivals=int(stream.count),
                          reward=float(data['reward'].sum()), game_overs=int(data['terminal'].sum()),
                          episodes=int(data['episode'].max()) + 1))
    write(args.output / 'manifest.json', dict(status='complete', name=args.output.name, side=side, sticky=args.sticky, actions=lanes * args.ticks,
        emulator_frames=int(sum(environment.executed_action_frames for environment in environments)),
        block_probability=args.block_probability, action_counts=action_counts.tolist(), files=files,
        seconds=time.monotonic() - started))


def frame_side(config):
    return int(config['observation_kind'].removeprefix('rgb'))


def evaluate_group(learner, streams, locations, output, name, draws, *, history=16, horizons=(1, 4, 8, 16, 32)):
    """Side-agnostic copy of breakout_world_corpus.evaluate_group: forecasts from
    recorded history and future actions, scored with the physical image metrics."""
    if not locations:
        write(output / f'{name}.json', dict(count=0, unsupported=True))
        return dict(count=0, unsupported=True)
    batch_size, side = learner.config['batch_size'], frame_side(learner.config)
    horizon = max(horizons)
    selected = np.asarray(horizons) - 1
    rgb_parts, posterior_parts, reward_parts, positive_parts, continuation_parts = ([[] for _ in draws] for _ in range(5))
    truth, context, validity, rewards, terminals, positions = ([] for _ in range(6))
    encoded, posterior_features, prior_features = [], [], []
    before = learner.updates
    for offset in range(0, len(locations), batch_size):
        actual = locations[offset:offset + batch_size]
        count = len(actual)
        data = evaluation_windows(streams, list(actual) + [actual[0]] * (batch_size - count), history=history, horizon=horizon)
        observed = data['rgb'][:history].astype(np.float32) / 255 - .5
        truth.append(data['rgb'][history + selected, :count])
        context.append(data['rgb'][history - 1, :count])
        validity.append(data['valid'][:, :count])
        rewards.append(data['reward'][history:, :count])
        terminals.append(data['terminal'][history:, :count])
        positions.append(data['position'][:, :count])
        for draw, seed in enumerate(draws):
            prediction = learner.predict(pixels(observed), data['action'][:history].ravel().tolist(),
                                         data['action'][history:].ravel().tolist(), seed=seed, include_features=draw == 0)
            if draw == 0:
                encoded.append(np.frombuffer(prediction['encoder_features'], '<f4').reshape(batch_size, -1)[:count])
                posterior_features.append(np.frombuffer(prediction['posterior_features'], '<f4').reshape(batch_size, -1)[:count])
                prior_features.append(np.frombuffer(prediction['prior_features'], '<f4').reshape(horizon, batch_size, -1)[selected, :count])
            rgb = np.frombuffer(prediction['prior_rgb'], '<f4').reshape(horizon, batch_size, 3, side, side)
            rgb_parts[draw].append(rgb[selected, :count])
            posterior_parts[draw].append(np.frombuffer(prediction['posterior_rgb'], '<f4').reshape(batch_size, 3, side, side)[:count])
            reward_parts[draw].append(np.asarray(prediction['reward'], np.float32)[:, :count])
            positive_parts[draw].append(np.asarray(prediction['reward_positive_probability'], np.float32)[:, :count])
            continuation_parts[draw].append(np.asarray(prediction['continuation'], np.float32)[:, :count])
    assert learner.updates == before
    arrays = dict(horizons=np.asarray(horizons), locations=np.asarray(locations), truth=np.concatenate(truth, axis=1),
                  context=np.concatenate(context), valid=np.concatenate(validity, axis=1),
                  target_reward=np.concatenate(rewards, axis=1), target_terminal=np.concatenate(terminals, axis=1),
                  position=np.concatenate(positions, axis=1), encoder_features=np.concatenate(encoded),
                  posterior_features=np.concatenate(posterior_features), prior_features=np.concatenate(prior_features, axis=1),
                  prediction=np.stack([np.concatenate(v, axis=1) for v in rgb_parts]),
                  posterior=np.stack([np.concatenate(v) for v in posterior_parts]),
                  reward=np.stack([np.concatenate(v, axis=1) for v in reward_parts]),
                  reward_positive_probability=np.stack([np.concatenate(v, axis=1) for v in positive_parts]),
                  continuation=np.stack([np.concatenate(v, axis=1) for v in continuation_parts]))
    for key in ('prediction', 'posterior', 'reward', 'reward_positive_probability', 'continuation'):
        assert np.isfinite(arrays[key]).all(), key
    np.savez_compressed(output / f'{name}.npz', **arrays)
    context = arrays['context'].astype(np.float32) / 255 - .5
    gamma = float(np.float32(1) - np.float32(1) / np.float32(learner.config['horizon']))
    result = dict(count=len(locations), side=side, posterior=[image_metrics(context, image, context) for image in arrays['posterior']], horizons={})
    for index, h in enumerate(horizons):
        valid = arrays['valid'][h - 1]
        target = arrays['truth'][index, valid].astype(np.float32) / 255 - .5
        actual_reward = arrays['target_reward'][h - 1, valid]
        probability = arrays['reward_positive_probability'][:, h - 1, valid].mean(axis=0)
        terminal_probability = 1 - np.clip(arrays['continuation'][:, h - 1, valid].mean(axis=0) / gamma, 0, 1)
        result['horizons'][str(h)] = dict(count=int(valid.sum()), excluded=int((~valid).sum()),
            images=[image_metrics(target, image[index, valid], context[valid]) for image in arrays['prediction']],
            persistence=image_metrics(target, context[valid], context[valid]),
            positive_reward=event_metrics(actual_reward > 0, probability),
            terminal=event_metrics(arrays['target_terminal'][h - 1, valid], terminal_probability))
    write(output / f'{name}.json', result)
    print(json.dumps(dict(event='evaluation', group=name, count=len(locations),
                          ball={h: [x['ball_within_1px'] for x in row['images']] for h, row in result['horizons'].items()})), flush=True)
    return result


def rerender(args):
    """Re-render a corpus that kept native frames at this config's frame side."""
    import kindle
    from breakout_world_model import require_device
    config = json.loads(args.config.read_text())
    lanes, side = config['batch_size'], frame_side(config)
    learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length, pixel_loss=args.pixel_loss)
    require_device(learner)
    source = json.loads((args.source / 'manifest.json').read_text())
    files = []
    for lane, item in enumerate(source['files']):
        with np.load(args.source / f'lane{lane}.npz', allow_pickle=False) as data:
            native = data['native_rgb']
            fields = {key: data[key] for key in FIELDS if key != 'rgb'}
        rgb = np.zeros((len(native), 3, side, side), np.uint8)
        for start in range(0, len(native), lanes):
            frames = [native[min(index, len(native) - 1)] for index in range(start, start + lanes)]
            resized = np.frombuffer(learner.preprocess(frames), '<f4').reshape(lanes, 3, side, side)
            count = min(lanes, len(native) - start)
            rgb[start:start + count] = np.rint((resized[:count] + .5) * 255).clip(0, 255).astype(np.uint8)
        path = args.output / f'lane{lane}.npz'
        np.savez(path, rgb=rgb, **fields)
        files.append(dict(item, path=str(path), sha256=sha(path), source_sha256=item['sha256']))
    write(args.output / 'manifest.json', dict(source, status='complete', side=side, rerendered_from=str(args.source), files=files))


def palette(args):
    """Most frequent frame colours of a corpus, as centered RGB for `rgb_history_palette`."""
    counts = {}
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    for lane in range(len(manifest['files'])):
        with np.load(args.dataset / f'lane{lane}.npz', allow_pickle=False) as data:
            frames = data['rgb'][::args.palette_stride]
        flat = frames.transpose(0, 2, 3, 1).reshape(-1, 3)
        colours, number = np.unique(flat, axis=0, return_counts=True)
        for colour, n in zip(map(tuple, colours.tolist()), number.tolist()):
            counts[colour] = counts.get(colour, 0) + n
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:args.colours]
    centered = [[round(channel / 255 - .5, 6) for channel in colour] for colour, _ in ranked]
    result = dict(dataset=str(args.dataset), colours=args.colours, stride=args.palette_stride, sampled_pixels=total,
                  coverage=sum(n for _, n in ranked) / total, rgb=[list(colour) for colour, _ in ranked], centered=centered)
    write(args.output / 'palette.json', result)
    if args.config is not None:
        config = json.loads(args.config.read_text())
        config['rgb_history_palette'] = centered
        write(args.output / 'config.json', config)
    print(json.dumps(dict(event='palette', colours=len(centered), coverage=result['coverage'])), flush=True)


def ridge(features, targets, lam):
    mean, deviation = features.mean(0), features.std(0) + 1e-6
    standardized = (features - mean) / deviation
    weights = np.linalg.solve(standardized.T @ standardized + lam * np.eye(features.shape[1]), standardized.T @ (targets - targets.mean(0)))
    return lambda x: ((x - mean) / deviation) @ weights + targets.mean(0)


def state_probe(fitting, development, horizons, history=16, lam=300.):
    """Linear readouts of RAM ball/paddle positions from saved features, fitting -> development."""
    report = {}
    sources = {'current': [('encoder_features', None), ('posterior_features', None)]}
    sources.update({str(h): [('prior_features', index)] for index, h in enumerate(horizons)})
    for label, items in sources.items():
        for key, index in items:
            for column, name in ((0, 'ball_x'), (1, 'ball_y'), (2, 'paddle_x')):
                row = history - 1 if index is None else history + int(label) - 1
                xf = fitting[key] if index is None else fitting[key][index]
                xd = development[key] if index is None else development[key][index]
                yf, yd = fitting['position'][row][:, column], development['position'][row][:, column]
                vf, vd = np.isfinite(yf), np.isfinite(yd)
                if index is not None:
                    vf &= fitting['valid'][int(label) - 1]
                    vd &= development['valid'][int(label) - 1]
                model = ridge(xf[vf], yf[vf][:, None], lam)
                error = float(np.sqrt(((model(xd[vd])[:, 0] - yd[vd]) ** 2).mean()))
                baseline = float(np.sqrt(((yf[vf].mean() - yd[vd]) ** 2).mean()))
                report[f'{label}:{key}:{name}'] = dict(rmse=error, baseline=baseline, fitting=int(vf.sum()), development=int(vd.sum()))
    return report


def window_streams(root, seed, count, history, horizon):
    """The dense origins of a corpus as one short stream each, loading one lane's frames at a time.

    Avoids holding a whole corpus in memory next to a running learner; the
    returned locations index the short streams, the originals are returned too.
    """
    manifest = json.loads((root / 'manifest.json').read_text())
    lanes = len(manifest['files'])
    lengths = []
    for lane in range(lanes):
        with np.load(root / f'lane{lane}.npz', allow_pickle=False) as data:
            lengths.append(len(data['action']))
    skeleton = [dict(action=np.zeros(length, np.int64)) for length in lengths]
    originals = sample_locations(skeleton, history + 1, count, np.random.default_rng(seed))
    streams = [None] * len(originals)
    for lane in range(lanes):
        wanted = [(index, start) for index, (origin_lane, start) in enumerate(originals) if origin_lane == lane]
        if not wanted:
            continue
        with np.load(root / f'lane{lane}.npz', allow_pickle=False) as data:
            fields = {key: data[key] for key in FIELDS}
        for index, start in wanted:
            stop = min(start + history + horizon, lengths[lane])
            streams[index] = {key: np.array(value[start:stop], copy=True) for key, value in fields.items()}
        del fields
    return streams, [(index, 0) for index in range(len(originals))], originals


def probe(args):
    """Restore a checkpoint, forecast the dense origins with features, and probe the state."""
    import kindle
    from breakout_world_corpus import DRAWS, HISTORY
    from breakout_world_model import require_device
    learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
    require_device(learner)
    groups, origins = {}, {}
    for split, root, seed in (('fitting', args.dataset, 182003), ('development', args.development, 182009)):
        streams, locations, origins[split] = window_streams(root, seed, args.dense, HISTORY, 32)
        evaluate_group(learner, streams, locations, args.output, f'dense-{split}', DRAWS[:1])
        groups[split] = np.load(args.output / f'dense-{split}.npz')
    report = state_probe(groups['fitting'], groups['development'], (1, 4, 8, 16, 32), lam=args.ridge)
    write(args.output / 'probe.json', dict(checkpoint=str(args.checkpoint), updates=learner.updates, ridge=args.ridge, origins=origins, readouts=report))
    for key, value in report.items():
        print(f"{key:40s} dev RMSE {value['rmse']:6.1f}  baseline {value['baseline']:6.1f}  (n {value['fitting']}/{value['development']})", flush=True)


def load_dataset(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    streams = []
    for lane, item in enumerate(manifest['files']):
        path = root / f'lane{lane}.npz'
        if sha(path) != item['sha256']:
            raise ValueError(f'corpus changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            streams.append({key: data[key] for key in FIELDS})
    return streams, manifest


def train(args):
    import kindle
    from breakout_world_corpus import DRAWS, HISTORY, choose_evaluation
    from breakout_world_model import require_device
    config = json.loads(args.config.read_text())
    fitting, fitting_manifest = load_dataset(args.dataset)
    development, development_manifest = load_dataset(args.development)
    assert not {row['seed'] for row in fitting_manifest['files']} & {row['seed'] for row in development_manifest['files']}
    learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length, pixel_loss=args.pixel_loss)
    require_device(learner)
    rng = np.random.default_rng(config['seed'])
    groups = dict(development=choose_evaluation(development, 112019))
    dense = dict(fitting=sample_locations(fitting, HISTORY + 1, args.dense, np.random.default_rng(182003)),
                 development=sample_locations(development, HISTORY + 1, args.dense, np.random.default_rng(182009)))
    write(args.output / 'declaration.json', dict(mode='train', config=config, steps=args.steps, eval_every=args.eval_every,
        prediction_length=args.prediction_length, pixel_loss=args.pixel_loss, dataset=str(args.dataset),
        development=str(args.development), fitting_sha256=sha(args.dataset / 'manifest.json'),
        development_sha256=sha(args.development / 'manifest.json'), source_sha256=sha(__file__),
        selections=groups, dense=dense, started_unix=time.time()))
    write(args.output / 'actual-config.json', learner.config)
    learner.save(str(args.output / 'initial'))
    native_seconds, started = 0., time.monotonic()

    def evaluate(step):
        output = args.output / 'evaluations' / str(step)
        output.mkdir(parents=True)
        summary = dict(step=step, training_seconds=native_seconds, groups={})
        for split, selections in groups.items():
            for kind, locations in selections.items():
                result = evaluate_group(learner, development, locations, output, f'{split}-{kind}', DRAWS)
                summary['groups'][f'{split}-{kind}'] = ball_summary(result)
        for split, locations in dense.items():
            streams = fitting if split == 'fitting' else development
            result = evaluate_group(learner, streams, locations, output, f'dense-{split}', draws=DRAWS[:1])
            summary['groups'][f'dense-{split}'] = ball_summary(result)
        with (args.output / 'curve.jsonl').open('a') as log:
            log.write(json.dumps(summary) + '\n')

    evaluate(0)
    with (args.output / 'learning.jsonl').open('x') as log:
        for step in range(1, args.steps + 1):
            locations = sample_locations(fitting, config['batch_length'], config['batch_size'], rng)
            batch = training_windows([fitting[lane] for lane, _ in locations], [start for _, start in locations],
                                     config['batch_length'], allow_resets=True, retain_initial_targets=True)
            observed = pixels(batch['rgb'].astype(np.float32) / 255 - .5)
            tick = time.monotonic()
            values = learner.learn(observed, *[batch[key].ravel().tolist() for key in ('action', 'reward', 'terminal')])
            native_seconds += time.monotonic() - tick
            assert np.isfinite(values).all(), values
            log.write(json.dumps(dict(step=step, locations=locations, losses=values, training_seconds=native_seconds)) + '\n')
            if step % 128 == 0:
                log.flush()
                print(json.dumps(dict(event='learning', step=step, losses=values, training_seconds=native_seconds,
                                      seconds=time.monotonic() - started)), flush=True)
                require_device(learner)
            if step % args.eval_every == 0 or step == args.steps:
                learner.save(str(args.output / f'checkpoint-{step}'))
                evaluate(step)
    write(args.output / 'result.json', dict(status='complete', updates=learner.updates, training_seconds=native_seconds,
        total_seconds=time.monotonic() - started, world_model_proven=False))


def ball_summary(result):
    if result.get('unsupported'):
        return None
    horizons = {horizon: [row['ball_within_1px'] for row in values['images']]
                for horizon, values in result['horizons'].items()}
    return dict(count=result['count'], observed=[row['ball_within_1px'] for row in result['posterior']], horizons=horizons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'train', 'rerender', 'palette', 'probe'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=65536, help='collection steps per lane')
    parser.add_argument('--seed', type=int, default=251001)
    parser.add_argument('--sticky', type=float, default=.25)
    parser.add_argument('--block-probability', type=float, default=.025)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--source', type=Path, help='corpus with native frames to re-render')
    parser.add_argument('--development', type=Path)
    parser.add_argument('--steps', type=int, default=16384)
    parser.add_argument('--eval-every', type=int, default=2048)
    parser.add_argument('--dense', type=int, default=128)
    parser.add_argument('--prediction-length', type=int, default=16)
    parser.add_argument('--pixel-loss', default='motion_event_rollout')
    parser.add_argument('--colours', type=int, default=32)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--ridge', type=float, default=300.)
    parser.add_argument('--palette-stride', type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'palette':
        if args.dataset is None:
            parser.error('palette requires --dataset')
        palette(args)
        return
    if args.mode == 'probe':
        if args.checkpoint is None or args.dataset is None or args.development is None:
            parser.error('probe requires --checkpoint, --dataset and --development')
        probe(args)
        return
    if args.config is None:
        parser.error('--config is required')
    if args.mode == 'collect':
        collect(args)
    elif args.mode == 'rerender':
        if args.source is None:
            parser.error('rerender requires --source')
        rerender(args)
    else:
        if args.dataset is None or args.development is None:
            parser.error('train requires --dataset and --development')
        train(args)


if __name__ == '__main__':
    main()
