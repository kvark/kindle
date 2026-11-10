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

from kindle._breakout_world import sample_locations, training_windows

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

    def __init__(self, capacity):
        self.rgb = np.zeros((capacity, 3, 64, 64), np.uint8)
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
    lanes = config['batch_size']
    learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length, pixel_loss=args.pixel_loss)
    require_device(learner)
    environments = [DreamerAtariPreprocessing(gym.make('ALE/Breakout-v5', frameskip=1,
        repeat_action_probability=args.sticky, full_action_space=True), noop_max=0, screen_size=None,
        max_episode_frames=108000) for _ in range(lanes)]
    teachers = [FeedbackPixelPolicy() for _ in range(lanes)]
    rngs = [np.random.default_rng(args.seed + lane * 1009 + 0x17a) for lane in range(lanes)]
    streams = [Lane(args.ticks + 1 + args.ticks // 16) for _ in range(lanes)]
    episodes = np.zeros(lanes, np.int64)
    blocks, command = np.zeros(lanes, int), np.zeros(lanes, int)
    frames = [environment.reset(seed=args.seed + lane * 1009)[0] for lane, environment in enumerate(environments)]
    started = time.monotonic()

    def append(indices, actions, rewards, terminals):
        rgb = np.frombuffer(learner.preprocess(frames), '<f4').reshape(lanes, 3, 64, 64)
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
    write(args.output / 'manifest.json', dict(status='complete', name=args.output.name, sticky=args.sticky, actions=lanes * args.ticks,
        emulator_frames=int(sum(environment.executed_action_frames for environment in environments)),
        block_probability=args.block_probability, action_counts=action_counts.tolist(), files=files,
        seconds=time.monotonic() - started))


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
    from breakout_world_corpus import DRAWS, HISTORY, choose_evaluation, evaluate_group
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
                result = evaluate_group(learner, development, locations, output, f'{split}-{kind}')
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
    parser.add_argument('mode', choices=('collect', 'train'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=65536, help='collection steps per lane')
    parser.add_argument('--seed', type=int, default=251001)
    parser.add_argument('--sticky', type=float, default=.25)
    parser.add_argument('--block-probability', type=float, default=.025)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--development', type=Path)
    parser.add_argument('--steps', type=int, default=16384)
    parser.add_argument('--eval-every', type=int, default=2048)
    parser.add_argument('--dense', type=int, default=128)
    parser.add_argument('--prediction-length', type=int, default=16)
    parser.add_argument('--pixel-loss', default='motion_event_rollout')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'collect':
        collect(args)
    else:
        if args.dataset is None or args.development is None:
            parser.error('train requires --dataset and --development')
        train(args)


if __name__ == '__main__':
    main()
