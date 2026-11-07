"""Fresh fixed-plan emulator branches for frozen world-model diagnosis.

Collection/clone fixtures construct only CPU Atari adapters. Model evaluation
is separate and must use the guarded, exact matching native runtime. Emulator
state, labels and category names are never model inputs.
"""
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HORIZONS = (1, 4, 8, 16, 32)
HISTORY = 32
COUNTERS = ('_episode_frames', 'executed_action_frames', 'reset_noop_frames',
            'emulator_resets', '_elapsed_steps', '_has_reset')
CATEGORIES = ('serve', 'freeflight', 'contact', 'brick', 'wall', 'life_loss', 'ending')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def wrappers(environment):
    while True:
        yield environment
        if 'env' not in environment.__dict__: break
        environment = environment.env


def environment():
    import ale_py
    import gymnasium as gym
    from atari import DreamerAtariPreprocessing
    gym.register_envs(ale_py)
    result = DreamerAtariPreprocessing(gym.make('ALE/Breakout-v5', frameskip=1,
        repeat_action_probability=.25, full_action_space=True), noop_max=0,
        screen_size=None, max_episode_frames=108000)
    assert len(result.action_meanings) == 18
    assert [result.action_meanings[i] for i in [0, 1, 3, 4, 12]] == ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'LEFTFIRE']
    return result


@dataclass
class Snapshot:
    full: bytes
    game: bytes
    frames: list
    counters: list
    ram: np.ndarray

    @classmethod
    def capture(cls, env):
        counters = [dict(type=f'{type(w).__module__}.{type(w).__qualname__}',
                         fields={key:copy.deepcopy(w.__dict__[key]) for key in COUNTERS if key in w.__dict__})
                    for w in wrappers(env)]
        return cls(env.unwrapped.clone_state(include_rng=True).serialize(),
                   env.unwrapped.clone_state(include_rng=False).serialize(),
                   [x.copy() for x in env._frames], counters, env.unwrapped.ale.getRAM().copy())

    def restore(self, env, *, rng):
        import ale_py
        env.unwrapped.restore_state(ale_py.ALEState(self.full if rng else self.game))
        chain = list(wrappers(env))
        assert len(chain) == len(self.counters)
        for wrapper, saved in zip(chain, self.counters):
            assert f'{type(wrapper).__module__}.{type(wrapper).__qualname__}' == saved['type']
            for key, value in saved['fields'].items(): wrapper.__dict__[key] = copy.deepcopy(value)
        env._frames = [x.copy() for x in self.frames]
        assert np.array_equal(env.unwrapped.ale.getRAM(), self.ram)
        assert env.unwrapped.clone_state(include_rng=False).equals(ale_py.ALEState(self.game))

    def save(self, root):
        root.mkdir()
        (root/'ale-with-rng.bin').write_bytes(self.full)
        (root/'ale-without-rng.bin').write_bytes(self.game)
        np.savez_compressed(root/'wrapper.npz', frames=np.stack(self.frames), ram=self.ram)
        write(root/'counters.json', self.counters)

    @classmethod
    def load(cls, root):
        with np.load(root/'wrapper.npz', allow_pickle=False) as data:
            frames, ram = data['frames'], data['ram']
        return cls((root/'ale-with-rng.bin').read_bytes(), (root/'ale-without-rng.bin').read_bytes(),
                   list(frames), json.loads((root/'counters.json').read_text()), ram)


def arrival(env, frame, action, reward=0., terminal=False):
    from kindle._representation_probe import positions
    return dict(native_rgb=frame.copy(), action=action, reward=reward, terminal=terminal,
                position=positions('Breakout', env.unwrapped.ale.getRAM()), lives=int(env.unwrapped.ale.lives()))


def rollout(env, commands):
    count = len(commands)
    rows = dict(native_rgb=np.zeros((count, 210, 160, 3), np.uint8),
                reward=np.zeros(count), terminal=np.zeros(count, bool), valid=np.zeros(count, bool),
                position=np.full((count, 3), np.nan), lives=np.zeros(count, np.int32))
    before = env.executed_action_frames
    for index, command in enumerate(commands):
        frame, reward, terminal, truncated, _ = env.step(int(command))
        assert not truncated, 'unexpected branch truncation'
        record = arrival(env, frame, int(command), reward, terminal)
        rows['valid'][index] = True
        for key in ['native_rgb', 'reward', 'terminal', 'position', 'lives']: rows[key][index] = record[key]
        if terminal: break
    return rows, env.executed_action_frames-before


def restore_branch(env, desired, origin, seed, transcript):
    """Reconstruct ALE's omitted sticky-action state, then install desired RNG.

    ALE v0.12.1 serializes machine/RNG state but not m_player_a_action.
    Replaying the original submitted commands reconstructs that private state.
    No branch outcome selects the commands or the target physical state.
    """
    env.reset(seed=seed)
    actions = frames = resets = 0
    for command in transcript:
        if command == -1:
            env.reset(); resets += 1
        else:
            before = env.executed_action_frames
            _, _, _, truncated, _ = env.step(int(command))
            assert not truncated
            frames += env.executed_action_frames-before; actions += 1
    import ale_py
    assert env.unwrapped.clone_state(include_rng=True).equals(ale_py.ALEState(origin.full)), 'source replay differs'
    desired.restore(env, rng=True)
    assert env.unwrapped.clone_state(include_rng=True).equals(ale_py.ALEState(desired.full))
    return dict(actions=actions, emulator_frames=frames, resets=resets+1)


def sticky_fixture(output, original):
    """Separate repair evidence; preserve the earlier partial clone fixture."""
    origin = Snapshot.load(original/'origin')
    transcript = [0]*31+[1]+[3]*5
    commands = np.zeros(32, np.int32)
    env = environment()
    failures, replay_actions, replay_frames = [], 0, 0
    try:
        env.reset(seed=901001)
        origin.restore(env, rng=True)
        for trial in range(16):
            origin.restore(env, rng=False)
            desired = Snapshot.capture(env)
            raw, repaired = [], []
            for previous in [4, 3]:
                for _ in range(8): env.step(previous)
                desired.restore(env, rng=True)
                raw.append(rollout(env, commands)[0])
                counts = restore_branch(env, desired, origin, 901001, transcript)
                replay_actions += counts['actions']; replay_frames += counts['emulator_frames']
                repaired.append(rollout(env, commands)[0])
            equal = lambda a,b: all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
            failures.append(not equal(*raw))
            assert equal(*repaired), 'source replay did not repair prior-action dependence'
            np.savez_compressed(output/f'trial-{trial}.npz', raw=np.stack([r['native_rgb'] for r in raw]),
                repaired=np.stack([r['native_rgb'] for r in repaired]))
        write(output/'result.json', dict(status='complete', cpu_only=True, gpu_constructed=False,
            trials=16, raw_restore_dependent_trials=sum(failures), repaired_exact_pairs=16,
            raw_defect_exposed=any(failures),
            replay_actions=replay_actions, replay_emulator_frames=replay_frames,
            original_fixture_preserved=str(original)))
    finally: env.close()


def clone_fixture(output):
    env = environment()
    try:
        env.reset(seed=901001)
        for command in [0]*31+[1]+[3]*5: env.step(command)
        snap = Snapshot.capture(env)
        commands = [4]*8+[3]*8+[0]*16
        exact = []
        for _ in range(2):
            snap.restore(env, rng=True)
            row, frames = rollout(env, commands)
            exact.append((row, frames))
        assert exact[0][1] == exact[1][1]
        assert all(np.array_equal(exact[0][0][k], exact[1][0][k], equal_nan=True) for k in exact[0][0])
        snap.restore(env, rng=True)
        hashes, all_rows, frames = [], [], 0
        for _ in range(16):
            snap.restore(env, rng=False)
            row, elapsed = rollout(env, commands)
            frames += elapsed
            all_rows.append(row)
            hashes.append(digest(row['native_rgb'].tobytes()+row['reward'].tobytes()+row['terminal'].tobytes()))
        assert len(set(hashes)) > 1, 'RNG-excluding restores did not expose stochastic variation'
        snap.save(output/'origin')
        np.savez_compressed(output/'oracle.npz', **{k:np.stack([row[k] for row in all_rows]) for k in all_rows[0]})
        write(output/'result.json', dict(status='complete', cpu_only=True, gpu_constructed=False,
            exact_rng_restore=True, stochastic_repeats=16, unique_stochastic_traces=len(set(hashes)),
            oracle_emulator_frames=frames, exact_repeat_frames=exact[0][1]+exact[1][1],
            wrapper_counters_restored=True, physical_states_equal=True))
    finally: env.close()


def plans(category):
    if category == 'serve': return [(f'action-{i}', np.full(32, i, np.int32)) for i in range(18)]
    return [('noop', np.zeros(32, np.int32)), ('left', np.full(32, 4, np.int32)),
            ('right', np.full(32, 3, np.int32)),
            ('reverse', np.r_[np.full(4, 3, np.int32), np.full(28, 4, np.int32)])]


def eligible(category, history):
    if len(history) < HISTORY: return False
    current, previous = history[-1], history[-2]
    if current['terminal']: return False
    point, old = current['position'], previous['position']
    if category == 'serve': return not np.isfinite(point[:2]).all() and current['lives'] > 0
    if not np.isfinite(point[:2]).all() or not np.isfinite(old[:2]).all(): return False
    if current['lives'] != previous['lives']: return False
    dx, dy = point[:2]-old[:2]
    x, y = point[:2]
    if category == 'freeflight': return 120 <= y <= 155 and dy < 0 and 25 < x < 135
    if category == 'contact': return 160 <= y <= 172 and dy > 0
    if category == 'brick': return 97 <= y <= 112 and dy < 0
    if category == 'wall': return 100 <= y <= 175 and ((x <= 18 and dx < 0) or (x >= 140 and dx > 0))
    if category == 'life_loss': return 150 <= y <= 160 and dy > 0 and current['lives'] >= 2
    if category == 'ending': return 150 <= y <= 160 and dy > 0 and current['lives'] == 1
    raise ValueError(category)


def source_action(category, history, teacher):
    proposal = teacher.propose(history[-1]['native_rgb'])
    if len(history) < HISTORY: return 0
    if category in ['life_loss', 'ending']: return 12
    return int(proposal)


def collect(output, repeats, source_ticks):
    from kindle._breakout_guidance import FeedbackPixelPolicy
    import ale_py
    summaries = []
    started = time.monotonic()
    for group, category in enumerate(CATEGORIES):
        root = output/category
        root.mkdir()
        envs = [environment() for _ in range(8)]
        teachers = [FeedbackPixelPolicy() for _ in range(8)]
        histories, snaps = [[] for _ in range(8)], [None]*8
        transcripts = [[] for _ in range(8)]
        seeds = [251001+10000*group+1009*lane for lane in range(8)]
        source_actions = source_frames = source_resets = 0
        try:
            for lane, env in enumerate(envs):
                frame, _ = env.reset(seed=seeds[lane]); source_resets += 1
                histories[lane].append(arrival(env, frame, -1))
            for tick in range(source_ticks):
                for lane, env in enumerate(envs):
                    if snaps[lane] is not None: continue
                    history = histories[lane]
                    if eligible(category, history):
                        snaps[lane] = Snapshot.capture(env)
                        continue
                    command = source_action(category, history, teachers[lane])
                    before = env.executed_action_frames
                    frame, reward, terminal, truncated, _ = env.step(int(command))
                    source_actions += 1; source_frames += env.executed_action_frames-before
                    transcripts[lane].append(int(command))
                    assert not truncated
                    teachers[lane].feedback(int(command))
                    history.append(arrival(env, frame, int(command), float(reward), bool(terminal)))
                    if len(history) > HISTORY: del history[0]
                    if terminal:
                        frame, _ = env.reset(); source_resets += 1
                        transcripts[lane].append(-1)
                        teachers[lane] = FeedbackPixelPolicy()
                        histories[lane] = [arrival(env, frame, -1)]
                if all(snap is not None for snap in snaps): break
            for lane, snap in enumerate(snaps):
                (snap if snap is not None else Snapshot.capture(envs[lane])).save(root/f'lane{lane}')
                h = histories[lane]
                np.savez_compressed(root/f'prefix-lane{lane}.npz', **{key:np.stack([row[key] for row in h]) for key in h[0]})
                write(root/f'transcript-lane{lane}.json', dict(seed=seeds[lane], commands=transcripts[lane]))
            if not all(snap is not None for snap in snaps):
                write(root/'source-failure.json', dict(category=category, ticks=source_ticks,
                    missing_lanes=[i for i,s in enumerate(snaps) if s is None], source_actions=source_actions, source_emulator_frames=source_frames))
                raise RuntimeError(f'incomplete source roots for {category}')
            source = {key:np.stack([np.stack([row[key] for row in h]) for h in histories], axis=1) for key in histories[0][0]}
            np.savez_compressed(root/'prefix.npz', **source)
            oracle_actions = oracle_frames = replay_actions = replay_frames = replay_resets = 0
            for name, commands in plans(category):
                rows, frame_count = [], 0
                for repeat in range(repeats):
                    batch = []
                    for lane, env in enumerate(envs):
                        snaps[lane].restore(env, rng=repeat == 0)
                        desired = Snapshot.capture(env)
                        counts = restore_branch(env, desired, snaps[lane], seeds[lane], transcripts[lane])
                        replay_actions += counts['actions']; replay_frames += counts['emulator_frames']
                        replay_resets += counts['resets']
                        row, elapsed = rollout(env, commands)
                        batch.append(row); frame_count += elapsed
                    rows.append({key:np.stack([row[key] for row in batch], axis=1) for key in batch[0]})
                selected = np.array(HORIZONS)-1
                arrays = {key:np.stack([row[key] for row in rows]) for key in rows[0] if key != 'native_rgb'}
                arrays['native_rgb'] = np.stack([row['native_rgb'][selected] for row in rows])
                arrays['first_native_sequence'] = rows[0]['native_rgb']
                arrays['future_action'] = np.broadcast_to(commands[:, None], (32, 8)).copy()
                np.savez_compressed(root/f'{name}.npz', **arrays)
                oracle_actions += int(arrays['valid'].sum()); oracle_frames += frame_count
            summary = dict(category=category, source_seeds=seeds, origins=8, plans=len(plans(category)),
                unique_native_contexts=len({digest(source['native_rgb'][:, lane].tobytes()) for lane in range(8)}),
                oracle_repeats=repeats, source_actions=source_actions, source_emulator_frames=source_frames,
                source_resets=source_resets, oracle_actions=oracle_actions, oracle_emulator_frames=oracle_frames,
                replay_actions=replay_actions, replay_emulator_frames=replay_frames, replay_resets=replay_resets)
            summaries.append(summary)
            write(root/'result.json', dict(status='complete', **summary))
            print(json.dumps(summary), flush=True)
        finally:
            for env in envs: env.close()
    files = [p for p in output.rglob('*') if p.is_file()]
    write(output/'result.json', dict(status='complete', cpu_only=True, ale_version=ale_py.__version__,
        wall_seconds=time.monotonic()-started,
        categories=summaries, horizon=32, histories=32, model_inputs='RGB and submitted actions only',
        split='fresh development challenge; not untouched test or natural-frequency calibration',
        files={str(p.relative_to(output)):digest(p.read_bytes()) for p in files}))


def energy_score(points, weights, truth):
    """Score a reported finite-mixture forecast against oracle samples."""
    points, weights, truth = (np.asarray(x, float) for x in (points, weights, truth))
    if points.ndim != 2 or truth.ndim != 2 or points.shape[1] != truth.shape[1]:
        raise ValueError('equal-dimensional point samples required')
    if len(weights) != len(points) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('normalized nonnegative mixture weights required')
    cross = np.linalg.norm(points[:, None]-truth[None], axis=-1)
    pairs = np.linalg.norm(points[:, None]-points[None], axis=-1)
    return float(weights@cross.mean(axis=1)-.5*np.sum(pairs*weights[:, None]*weights[None]))


def event_forecasts(reward, positive, continuation, gamma):
    """Conditional head factorization, with physical survival discount removed."""
    reward, positive, continuation = (np.asarray(x, float) for x in (reward, positive, continuation))
    alive = np.clip(continuation/gamma, 0, 1)
    positive = np.clip(positive, 0, 1)
    survival = np.cumprod(alive, axis=1)
    previous_survival = np.concatenate([np.ones_like(survival[:, :1]), survival[:, :-1]], axis=1)
    cumulative_reward = np.cumsum(previous_survival*reward, axis=1)
    no_reward_alive = np.ones_like(reward[:, 0])
    reward_seen = np.zeros_like(no_reward_alive)
    any_reward = []
    for step in range(reward.shape[1]):
        reward_seen += no_reward_alive*positive[:, step]
        no_reward_alive *= (1-positive[:, step])*alive[:, step]
        any_reward.append(reward_seen.copy())
    return survival, np.stack(any_reward, axis=1), cumulative_reward


def observable_features(images):
    from kindle._breakout_world import ball_center, paddle_center, brick_strength
    images = np.asarray(images)
    leading = images.shape[:-3]
    flat = images.reshape(-1, 3, 64, 64)
    balls, paddles = np.zeros((len(flat), 4)), np.zeros((len(flat), 3))
    for i, frame in enumerate(flat):
        ball, paddle = ball_center(frame), paddle_center(frame)
        if ball is not None: balls[i, 1:] = [1., ball[0]/64, ball[1]/64]
        if paddle is not None: paddles[i, 1:] = [1., paddle/64]
    bricks = brick_strength(flat) >= .5
    return dict(ball=balls.reshape(*leading, 4), paddle=paddles.reshape(*leading, 3),
                bricks=bricks.reshape(*leading, 108))


def native_bricks(images):
    from kindle._breakout_world import BRICK_COLORS
    leading = images.shape[:-3]
    flat = images.reshape(-1, 210, 160, 3)
    cells = flat[:, 57:93, 8:152].reshape(len(flat), 6, 6, 18, 8, 3).transpose(0, 1, 3, 2, 4, 5)
    colors = np.rint(BRICK_COLORS*255).astype(np.uint8)[None, :, None, None, None]
    matches = (cells == colors).all(axis=-1).sum(axis=(-1, -2))
    return (matches > 24).reshape(*leading, 108), (matches != 24).reshape(*leading, 108)


def preprocess(learner, native):
    """The fixed native camera resize; never downscale on CPU."""
    leading = native.shape[:-4]
    if native.shape[-4:] != (8, 210, 160, 3): raise ValueError('eight native RGB frames per batch required')
    resized = []
    for batch in native.reshape(-1, 8, 210, 160, 3):
        values = np.frombuffer(learner.preprocess(list(batch)), '<f4').reshape(8, 3, 64, 64)
        quantized = np.rint((values+.5)*255).clip(0, 255).astype(np.uint8)
        assert np.max(np.abs(quantized.astype(np.float32)/255-.5-values)) < 1e-6
        resized.append(quantized)
    return np.stack(resized).reshape(*leading, 8, 3, 64, 64)


def predict(learner, prefix, actions, future, draws):
    from breakout_world_model import pixels
    results = []
    for seed in draws:
        result = learner.predict(pixels(prefix), actions.ravel().tolist(), future.ravel().tolist(), seed=seed)
        results.append(dict(prediction=np.frombuffer(result['prior_rgb'], '<f4').reshape(32, 8, 3, 64, 64),
            posterior=np.frombuffer(result['posterior_rgb'], '<f4').reshape(8, 3, 64, 64),
            reward=np.asarray(result['reward']), positive=np.asarray(result['reward_positive_probability']),
            continuation=np.asarray(result['continuation'])))
    arrays = {key:np.stack([row[key] for row in results]) for key in results[0]}
    assert all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite forecast'
    return arrays


def score(predictions, oracle, context, gamma):
    from kindle._breakout_world import ball_center, image_metrics
    selected = np.array(HORIZONS)-1
    feature = observable_features(predictions['prediction'][:, selected])
    truth = observable_features(oracle['rgb'].astype(np.float32)/255-.5)
    dead = np.maximum.accumulate(oracle['terminal'], axis=1)[:, selected]
    survival, any_reward, cumulative_reward = event_forecasts(predictions['reward'], predictions['positive'], predictions['continuation'], gamma)
    brick_truth, brick_valid = native_bricks(oracle['native_rgb'])
    result = dict(posterior=[image_metrics(context[-1], image, context[-1]) for image in predictions['posterior']], horizons={})
    previous_centers = [ball_center(image) for image in context[-2]]
    current_centers = [ball_center(image) for image in context[-1]]
    context_features = observable_features(context[-1])
    for index, horizon in enumerate(HORIZONS):
        row = dict(energy={}, per_origin={}, controls={}, events={})
        truth_end = dead[:, index]
        truth_reward = (oracle['reward'][:, :horizon] > 0).any(axis=1)
        truth_total = oracle['reward'][:, :horizon].sum(axis=1)
        end_probability = 1-survival[:, horizon-1].mean(axis=0)
        reward_probability = any_reward[:, horizon-1].mean(axis=0)
        predicted_total = cumulative_reward[:, horizon-1].mean(axis=0)
        row['events'] = dict(end_brier=float(np.square(end_probability[None]-truth_end).mean()),
            reward_brier=float(np.square(reward_probability[None]-truth_reward).mean()),
            cumulative_reward_mae=float(np.abs(predicted_total[None]-truth_total).mean()),
            oracle_end_probability=truth_end.mean(axis=0).tolist(), model_end_probability=end_probability.tolist(),
            oracle_reward_probability=truth_reward.mean(axis=0).tolist(), model_reward_probability=reward_probability.tolist())
        for name in ['ball', 'paddle']:
            y = truth[name][:, index].copy()
            y[truth_end] = 0
            y[..., 0] = truth_end
            x = feature[name][:, index]
            atom = np.zeros(x.shape[-1]); atom[0] = 1
            scores, persistence, velocity, oracle_floor = [], [], [], []
            for lane in range(8):
                alive_weight = survival[:, horizon-1, lane]/len(x)
                weights = np.r_[alive_weight, max(0., 1-alive_weight.sum())]
                points = np.concatenate([x[:, lane], atom[None]], axis=0)
                scores.append(energy_score(points, weights, y[:, lane]))
                base = context_features[name][lane]
                persistence.append(energy_score(base[None], [1.], y[:, lane]))
                if name == 'ball':
                    position = None
                    a, b = previous_centers[lane], current_centers[lane]
                    if a is not None and b is not None: position = b+(b-a)*horizon
                    cv = np.zeros(4)
                    if position is not None and 4 <= position[0] < 60 and 28 <= position[1] < 56:
                        cv[1:] = [1., position[0]/64, position[1]/64]
                    velocity.append(energy_score(cv[None], [1.], y[:, lane]))
                distances = np.linalg.norm(y[:, lane, None]-y[None, :, lane], axis=-1)
                oracle_floor.append(float(distances.sum()/(2*len(y)*(len(y)-1))))
            row['energy'][name] = float(np.mean(scores))
            row['per_origin'][name] = scores
            row['controls'][name] = dict(persistence=float(np.mean(persistence)),
                oracle_entropy_reference=float(np.mean(oracle_floor)))
            if velocity: row['controls'][name]['constant_velocity'] = float(np.mean(velocity))
        alive = ~truth_end
        estimated_bricks = feature['bricks'][:, index].mean(axis=0)
        valid_cells = brick_valid[:, index] & alive[..., None]
        squared = np.square(estimated_bricks[None]-brick_truth[:, index])
        row['brick_brier_alive_only'] = float(squared[valid_cells].mean()) if valid_cells.any() else None
        row['brick_cells_scored'] = int(valid_cells.sum())
        row['oracle_alive_endpoints'] = int(alive.sum())
        result['horizons'][str(horizon)] = row
    return result, dict(survival=survival, any_reward=any_reward, cumulative_reward=cumulative_reward,
                        ball_features=feature['ball'], paddle_features=feature['paddle'])


def evaluate(args):
    import kindle
    from safetensors.numpy import load_file
    from breakout_world_model import require_device
    started = time.monotonic()
    manifest = json.loads((args.dataset/'result.json').read_text())
    assert manifest['status'] == 'complete'
    for path, expected in manifest['files'].items(): assert digest((args.dataset/path).read_bytes()) == expected
    learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
    require_device(learner)
    before = learner.updates
    learner.save(str(args.output/'before'))
    gamma = float(np.float32(1)-np.float32(1)/np.float32(learner.config['horizon']))
    draws = [983001+1009*i for i in range(args.draws)]
    summaries = {}
    for category in CATEGORIES:
        with np.load(args.dataset/category/'prefix.npz', allow_pickle=False) as data:
            prefix, action = preprocess(learner, data['native_rgb']), data['action']
        context = prefix.astype(np.float32)/255-.5
        for plan, _ in plans(category):
            with np.load(args.dataset/category/f'{plan}.npz', allow_pickle=False) as data:
                oracle = {key:data[key] for key in ['native_rgb', 'reward', 'terminal', 'valid', 'future_action']}
            oracle['rgb'] = preprocess(learner, oracle['native_rgb'])
            np.savez_compressed(args.output/f'{category}-{plan}-truth.npz', rgb=oracle['rgb'], reward=oracle['reward'],
                                terminal=oracle['terminal'], valid=oracle['valid'], future_action=oracle['future_action'])
            for history in args.histories:
                incoming = action[-history:].copy(); incoming[0] = -1
                prediction = predict(learner, context[-history:], incoming, oracle['future_action'], draws)
                report, arrays = score(prediction, oracle, context, gamma)
                label = f'{category}-{plan}-h{history}'
                write(args.output/f'{label}.json', report)
                np.savez_compressed(args.output/f'{label}.npz', **arrays, prediction=prediction['prediction'][:, np.array(HORIZONS)-1],
                    first_prediction=prediction['prediction'][0], posterior=prediction['posterior'],
                    reward=prediction['reward'], positive=prediction['positive'], continuation=prediction['continuation'])
                summaries[label] = report
                assert learner.updates == before
                print(json.dumps(dict(group=label, ball_energy={h:v['energy']['ball'] for h,v in report['horizons'].items()})), flush=True)
    learner.save(str(args.output/'after'))
    a, b = (load_file(args.output/phase/'world.safetensors') for phase in ['before', 'after'])
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    write(args.output/'result.json', dict(status='complete', updates_before=before, updates_after=learner.updates,
        frozen_tensors=len(a), groups=len(summaries), draws=draws, histories=args.histories,
        wall_seconds=time.monotonic()-started,
        no_learning=True, model_proven=False, split='fresh development challenge; not untouched test'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fixture', 'sticky-fixture', 'collect', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--source-ticks', type=int, default=1024)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--histories', type=int, nargs='+', default=[16, 32])
    args = parser.parse_args()
    if args.repeats < 2 or args.source_ticks < HISTORY: parser.error('invalid bounded collection budgets')
    if args.mode == 'evaluate' and (args.checkpoint is None or args.dataset is None or args.draws < 2 or not set(args.histories) <= {16, 32}):
        parser.error('evaluate requires checkpoint/dataset, at least2 draws and history16/32')
    if args.mode == 'sticky-fixture' and args.dataset is None: parser.error('sticky fixture requires original fixture dataset')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'declaration.json', dict(mode=args.mode, source_sha256=digest(Path(__file__).read_bytes()),
        repeats=args.repeats, source_ticks=args.source_ticks, started_unix=time.time()))
    if args.mode == 'fixture': clone_fixture(args.output)
    elif args.mode == 'sticky-fixture': sticky_fixture(args.output, args.dataset)
    elif args.mode == 'collect': collect(args.output, args.repeats, args.source_ticks)
    else: evaluate(args)


if __name__ == '__main__': main()
