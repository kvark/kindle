"""Frozen native forecasts on saved, paired Breakout rescue sequences."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


HORIZON = 9


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(row):
    return row['stream'], row['episode'], row['step']


def load_inputs(log, annotations, branches, origins):
    events = [json.loads(line) for line in Path(log).open()]
    header, end = events[0], events[-1]
    if (header['event'] != 'run_start' or end['event'] != 'run_end'
            or header['num_envs'] != 8 or header['config']['action_count'] != 18
            or header['environment'] != 'ALE/Breakout-v5'
            or not header['full_action_space'] or header['noop_max'] != 0
            or header['action_repeat'] != 4 or header['sticky_actions'] != .25
            or end['learner_updates'] != 0):
        raise ValueError('complete unchanged frozen Breakout source required')
    rows = [json.loads(line) for line in Path(annotations).open()]
    cursor, episodes, steps = 0, [0]*8, [0]*8
    lanes = [[] for _ in range(8)]
    for event in events:
        if event['event'] != 'transition':
            continue
        for lane in range(8):
            row = rows[cursor]
            expected = dict(stream=lane, episode=episodes[lane], step=steps[lane],
                action=event['actions'][lane], reward=event['rewards'][lane],
                terminated=event['terminated'][lane], truncated=event['truncated'][lane],
                frames=event['executed_action_frames'][lane])
            if any(row[k] != v for k, v in expected.items()):
                raise ValueError('annotation differs from original vector log')
            lanes[lane].append(row)
            cursor += 1
            steps[lane] += 1
            if row['terminated'] or row['truncated']:
                episodes[lane] += 1
                steps[lane] = 0
    if cursor != len(rows) or cursor != end['run_step']:
        raise ValueError('source row count differs')
    lookup = {identity(row): row for row in rows}
    chosen = {identity(row): row for row in read(origins)}
    if len(lookup) != len(rows) or len(chosen) != 120:
        raise ValueError('unique complete source and fixed origins required')
    pairs = {}
    for branch in map(json.loads, Path(branches).open()):
        origin = branch['origin']
        key = identity(origin)
        if key in pairs or chosen.get(key) != origin or origin['recorded_actions'] != HORIZON:
            raise ValueError('saved origin differs')
        recorded = branch['recorded']['trajectory']
        teacher = branch['branches']['existing']['trajectory'][:HORIZON]
        if len(recorded) != HORIZON or len(teacher) != HORIZON:
            raise ValueError('complete nine-action plans required')
        for offset, row in enumerate(recorded):
            source = lookup[key[0], key[1], key[2]+offset]
            for name in ('action', 'reward', 'frames', 'terminated', 'truncated',
                         'input_pixel_sha256', 'output_pixel_sha256'):
                if row[name] != source[name]:
                    raise ValueError('recorded branch differs from source')
            if row['lives'] != source['lives_after']:
                raise ValueError('recorded life count differs')
        initial_lives = branch['recorded']['initial_lives']
        for plan in (recorded, teacher):
            if (plan[0]['input_pixel_sha256'] != origin['input_pixel_sha256']
                    or any(r['terminated'] or r['truncated'] for r in plan[:-1])
                    or plan[-1]['truncated']
                    or any(type(r['action']) is not int or not 0 <= r['action'] < 18 for r in plan)):
                raise ValueError('invalid branch horizon or action')
            if any(a['output_pixel_sha256'] != b['input_pixel_sha256'] for a, b in zip(plan, plan[1:])):
                raise ValueError('branch pixel chain differs')
        pairs[key] = dict(origin=origin, initial_lives=initial_lives,
            recorded=recorded, teacher=teacher,
            teacher_avoids_loss=teacher[-1]['lives'] == initial_lives,
            recorded_game_over=recorded[-1]['terminated'],
            teacher_game_over=teacher[-1]['terminated'])
    if set(pairs) != set(chosen):
        raise ValueError('missing fixed origin')
    return header, end, lanes, pairs


def forecast_scores(prediction):
    rewards, continuations, values = (prediction[name] for name in ('rewards', 'continuations', 'values'))
    if (any(len(x) != HORIZON for x in (rewards, continuations, values))
            or not all(math.isfinite(x) for xs in (rewards, continuations, values) for x in xs)
            or not all(0 <= c <= 1 for c in continuations)):
        raise ValueError('finite nine-step forecast with bounded continuations required')
    live, reward = 1.0, 0.0
    for r, c in zip(rewards, continuations):
        reward += live*r
        live *= c
    return dict(reward=reward, survival_mass=live,
                reward_plus_endpoint_value=reward+live*values[-1])


def preference(teacher, recorded, tolerance=1e-6):
    if not math.isfinite(teacher) or not math.isfinite(recorded):
        raise ValueError('finite scores required')
    return .5 if abs(teacher-recorded) <= tolerance else float(teacher > recorded)


def compare_pair(pair, predictions, gamma):
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError('finite model discount in (0,1) required')
    scores = {name: forecast_scores(predictions[name]) for name in ('recorded', 'teacher')}
    truth = {}
    for name in scores:
        rows = pair[name]
        if (len(rows) != HORIZON or any(r['terminated'] or r['truncated'] for r in rows[:-1])
                or rows[-1]['truncated'] or not all(math.isfinite(r['reward']) for r in rows)):
            raise ValueError('complete branch without an earlier end required')
        truth[name] = dict(reward=sum(gamma**i*r['reward'] for i, r in enumerate(rows)),
                          survival_mass=gamma**HORIZON*(not rows[-1]['terminated']))
    return dict(origin=pair['origin'], initial_lives=pair['initial_lives'],
        teacher_avoids_loss=pair['teacher_avoids_loss'],
        recorded_game_over=pair['recorded_game_over'], teacher_game_over=pair['teacher_game_over'],
        predictions=scores, actual=truth,
        continuation_teacher_preference=preference(scores['teacher']['survival_mass'], scores['recorded']['survival_mass']),
        value_teacher_preference=preference(scores['teacher']['reward_plus_endpoint_value'], scores['recorded']['reward_plus_endpoint_value']),
        reward_teacher_preference=preference(scores['teacher']['reward'], scores['recorded']['reward']))


def capture(log, annotations, branches, origins, checkpoint, output):
    header, end, lanes, pairs = load_inputs(log, annotations, branches, origins)
    metadata = read(checkpoint/'metadata.json')
    if header['config'] != metadata['config']:
        raise ValueError('source and checkpoint configurations differ')
    output.mkdir(exist_ok=False)
    import ale_py
    import gymnasium as gym
    import kindle
    from atari import DreamerAtariPreprocessing
    from safetensors import safe_open
    import numpy as np
    gym.register_envs(ale_py)
    started = time.monotonic()
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    if agent.config != metadata['config'] or agent.gpu_device != header['gpu_device']:
        raise ValueError('restored config or device differs')
    if sha(kindle._native.__file__) != header['native_extension_sha256']:
        raise ValueError('native extension differs')
    initial_actions, initial_updates = agent.environment_step, agent.learner_step
    result = dict(status='running', source_log_sha256=sha(log), actions=0, frames=0,
        updates=0, origins=[], streams=[], native_sha256=sha(kindle._native.__file__),
        source_sha256=sha(__file__), gpu_device=agent.gpu_device)
    try:
        for lane, rows in enumerate(lanes):
            env = DreamerAtariPreprocessing(gym.make(header['environment'], frameskip=1,
                repeat_action_probability=header['sticky_actions'], full_action_space=True),
                noop_max=0, max_episode_frames=header['max_episode_frames'], screen_size=None)
            try:
                if list(env.action_meanings) != header['action_meanings']:
                    raise ValueError('action mapping differs')
                frame, _ = env.reset(seed=header['environment_seeds'][lane])
                agent.begin_episode(frame)
                digest = hashlib.sha256()
                for index, row in enumerate(rows):
                    if hashlib.sha256(frame.tobytes()).hexdigest() != row['input_pixel_sha256']:
                        raise ValueError('input pixels differ')
                    key = identity(row)
                    if key in pairs:
                        pair = pairs[key]
                        before = (agent.latent_feature, agent.posterior_action_probabilities(),
                                  agent.environment_step, agent.learner_step)
                        predictions = {}
                        for name in ('recorded', 'teacher'):
                            plan = [r['action'] for r in pair[name]]
                            rewards, continuations, values = agent.prior_behavior_rollout(plan)
                            prediction = dict(rewards=rewards, continuations=continuations, values=values)
                            forecast_scores(prediction)
                            predictions[name] = prediction
                        repeated = agent.prior_behavior_rollout([r['action'] for r in pair['recorded']])
                        first = predictions['recorded']
                        if repeated != (first['rewards'], first['continuations'], first['values']):
                            raise ValueError('repeated diagnostic forecast differs')
                        after = (agent.latent_feature, agent.posterior_action_probabilities(),
                                 agent.environment_step, agent.learner_step)
                        if before != after:
                            raise ValueError('diagnostic changed live state')
                        result['origins'].append(dict(origin=pair['origin'], predictions=predictions,
                            probabilities=before[1]))
                    action = row['action']
                    if agent.act(action_mask=[a == action for a in range(18)]) != action:
                        raise ValueError('forced source action differs')
                    frame, reward, terminal, cutoff, _ = env.step(action)
                    if (reward, terminal, cutoff, env.executed_action_frames) != (
                            row['reward'], row['terminated'], row['truncated'], row['frames']):
                        raise ValueError('source transition differs')
                    if hashlib.sha256(frame.tobytes()).hexdigest() != row['output_pixel_sha256']:
                        raise ValueError('output pixels differ')
                    digest.update(frame.tobytes())
                    tail = index+1 == len(rows) and not terminal and not cutoff
                    agent.observe(frame, extrinsic_reward=float(reward), terminated=terminal,
                                  truncated=cutoff or tail)
                    result['actions'] += 1
                    if terminal or cutoff:
                        frame, _ = env.reset()
                        if index+1 < len(rows):
                            agent.begin_episode(frame)
                    if index % 256 == 0:
                        memory = agent.gpu_memory_budget
                        headroom = memory['budget_bytes']-memory['usage_bytes']
                        if headroom < 2 << 30:
                            raise RuntimeError('less than 2GiB GPU budget headroom')
                        result['minimum_headroom_bytes'] = min(headroom, result.get('minimum_headroom_bytes', headroom))
                result['frames'] += env.executed_action_frames
                result['streams'].append(dict(stream=lane, actions=len(rows), frames=env.executed_action_frames,
                    resets=env.emulator_resets, output_pixel_sha256=digest.hexdigest()))
                if (env.executed_action_frames != end['executed_action_frames'][lane]
                        or env.emulator_resets != end['emulator_resets'][lane]):
                    raise ValueError('source frame/reset totals differ')
                (output/'progress.json').write_text(json.dumps(dict(actions=result['actions'], origins=len(result['origins'])))+'\n')
            finally:
                env.close()
        agent.save_checkpoint(str(output/'frozen-after'))
        counts = {}
        for name in ('world', 'behavior', 'slow_value'):
            with safe_open(checkpoint/f'{name}.safetensors', framework='np') as left, \
                    safe_open(output/'frozen-after'/f'{name}.safetensors', framework='np') as right:
                if set(left.keys()) != set(right.keys()):
                    raise ValueError('frozen tensor keys differ')
                for key in left.keys():
                    a, b = left.get_tensor(key), right.get_tensor(key)
                    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes() or not np.isfinite(b).all():
                        raise ValueError(f'frozen tensor differs: {name}/{key}')
                counts[name] = len(left.keys())
        if (agent.learner_step != initial_updates or agent.environment_step-initial_actions != result['actions']
                or result['actions'] != end['run_step'] or len(result['origins']) != len(pairs)):
            raise ValueError('capture counters differ')
        result.update(status='complete', seconds=time.monotonic()-started, unchanged_tensors=counts)
        write(output/'result.json', result)
    except BaseException as error:
        result.update(status='failed', seconds=time.monotonic()-started, reason=repr(error))
        write(output/'result.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('log', 'annotations', 'branches', 'origins', 'checkpoint', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    capture(**vars(args))
