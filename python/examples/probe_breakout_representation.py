"""Frozen CNN/RSSM readouts on recorded Breakout states; no policy learning.

Run capture under gpu_host_guard.py. Fit mode uses CPU ridge regression and
never imports the native extension. Teacher detections are diagnostic labels.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from kindle._representation_probe import regression_metrics, ridge_probe


PROTOCOL = 'kindle-breakout-representation-v1'
TARGETS = ('ball_x', 'ball_y', 'paddle_x', 'dx', 'dy', 'response_target_x')
SPLITS = ('train', 'validation', 'test')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def row_split(lane, episode, step):
    if episode >= 3 or step % 8:
        return None
    return 'train' if lane < 4 else 'validation' if lane < 6 else 'test'


def validate_corpus(corpus):
    if (corpus['status'] != 'complete' or tuple(corpus['targets']) != TARGETS
            or len(corpus['streams']) != 8 or corpus['source_header']['num_envs'] != 8
            or corpus['source_header']['environment'] != 'ALE/Breakout-v5'
            or corpus['source_header']['mode'] != 'evaluate_greedy'):
        raise ValueError('requires the complete eight-lane Breakout corpus')
    counts = dict.fromkeys(SPLITS, 0)
    for lane, rows in enumerate(corpus['streams']):
        episode, step, frames, resets = 0, 0, 0, 1
        if not rows:
            raise ValueError('empty stream')
        for index, row in enumerate(rows):
            if (row['episode'], row['step']) != (episode, step):
                raise ValueError('episode/step boundary alignment differs')
            split = row_split(lane, episode, step)
            if row['split'] != split:
                raise ValueError('lane split or fixed stride differs')
            if split is not None:
                counts[split] += 1
            if not 0 <= row['action'] < corpus['source_header']['config']['action_count']:
                raise ValueError('invalid recorded action')
            if not 0 < row['frames'] - frames <= 4 or row['truncated']:
                raise ValueError('frame accounting or natural boundary differs')
            if (len(row['targets']) != len(TARGETS)
                    or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                           for v in row['targets'])):
                raise ValueError('invalid target vector')
            if step == 0 and any(row['targets'][i] is not None for i in (3, 4)):
                raise ValueError('motion target crosses episode reset')
            if len(bytes.fromhex(row['input_pixel_sha256'])) != 32:
                raise ValueError('invalid pixel hash')
            frames = row['frames']
            if row['reset_after']:
                if not row['terminated']:
                    raise ValueError('reset without natural boundary')
                episode, step, resets = episode + 1, 0, resets + 1
            else:
                if row['terminated'] and index + 1 < len(rows):
                    raise ValueError('terminal arrival lacks recorded reset')
                step += 1
        if (frames != corpus['source_end']['executed_action_frames'][lane]
                or resets != corpus['source_end']['emulator_resets'][lane]):
            raise ValueError('stream accounting differs')
    if counts != corpus['counts'] or sum(map(len, corpus['streams'])) != corpus['source_end']['run_step']:
        raise ValueError('corpus totals differ')
    return counts


def read_features(corpus, directory):
    validate_corpus(corpus)
    result = json.loads((directory/'result.json').read_text())
    if (result['status'] != 'complete' or result['smoke'] or result['protocol'] != PROTOCOL
            or len(result['files']) != 8 or result['actor_updates'] != 0
            or result['diagnostic_actions'] != corpus['source_end']['run_step']):
        raise ValueError('requires a complete full frozen capture')
    parts = []
    for lane, file in enumerate(result['files']):
        path = directory/file['file']
        if file['stream'] != lane or sha(path) != file['sha256']:
            raise ValueError('capture file identity differs')
        rows = corpus['streams'][lane]
        digest = hashlib.sha256()
        for row in rows:
            digest.update(bytes.fromhex(row['input_pixel_sha256']))
        if file['input_hash_sequence_sha256'] != digest.hexdigest():
            raise ValueError('captured input trajectory differs')
        indices = np.array([i for i, row in enumerate(rows) if row['split'] is not None], dtype=np.int32)
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {'indices', 'cnn', 'posterior'} or not np.array_equal(values['indices'], indices):
                raise ValueError('feature/target row alignment differs')
            part = {key: values[key].copy() for key in ('cnn', 'posterior')}
        if any(x.ndim != 2 or len(x) != len(indices) or not x.shape[1] or not np.isfinite(x).all()
               for x in part.values()):
            raise ValueError('invalid captured features')
        part.update(lane=np.full(len(indices), lane), index=indices,
                    split=np.array([rows[i]['split'] for i in indices]),
                    phase=np.array([rows[i]['phase'] for i in indices]),
                    targets=np.asarray([rows[i]['targets'] for i in indices], dtype=np.float64))
        parts.append(part)
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def evaluate(prediction, targets, lanes, phases):
    masks = dict(all=np.ones(len(targets), dtype=bool), near_descending=phases == 'near_descending')
    masks.update({f'lane{lane}': lanes == lane for lane in np.unique(lanes)})
    return {name: dict(zip(TARGETS, regression_metrics(prediction[mask], targets[mask])))
            for name, mask in masks.items()}


def fit_readouts(data):
    masks = {split: data['split'] == split for split in SPLITS}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('missing readout split')
    train, validation, test = [masks[key] for key in SPLITS]
    if (set(data['lane'][train]) - {0, 1, 2, 3} or set(data['lane'][validation]) - {4, 5}
            or set(data['lane'][test]) - {6, 7}):
        raise ValueError('readout lanes overlap split roles')
    ys = [data['targets'][mask] for mask in (train, validation, test)]
    if any(np.any(np.isfinite(y).sum(axis=0) < 2) for y in ys[:2]):
        raise ValueError('insufficient training/validation target support')
    baseline = np.broadcast_to(np.nanmean(ys[0], axis=0), ys[2].shape)
    result = dict(counts={key: int(mask.sum()) for key, mask in masks.items()},
                  training_mean=np.nanmean(ys[0], axis=0).tolist(),
                  constant=evaluate(baseline, ys[2], data['lane'][test], data['phase'][test]), variants={})
    predictions = {'constant': baseline.copy()}
    for feature in ('cnn', 'posterior'):
        xs = [data[feature][mask] for mask in (train, validation, test)]
        prediction, selection = ridge_probe(xs[0], ys[0], xs[1], ys[1], xs[2])
        predictions[feature] = prediction
        result['variants'][feature] = dict(width=xs[0].shape[1], selection=selection,
            metrics=evaluate(prediction, ys[2], data['lane'][test], data['phase'][test]))
    return result, predictions


def require_cpu_only():
    if 'kindle._native' in sys.modules:
        raise RuntimeError('CPU readout imported native extension')
    for descriptor in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if target.startswith('/dev/dri/'):
            raise RuntimeError('CPU readout opened a GPU device')


def fit(corpus_path, final, initial, output):
    require_cpu_only()
    started = time.monotonic()
    corpus = json.loads(corpus_path.read_text())
    for directory in (final, initial):
        if json.loads((directory/'result.json').read_text())['corpus_sha256'] != sha(corpus_path):
            raise ValueError('capture corpus identity differs')
    arrays = {name: read_features(corpus, path) for name, path in (('final', final), ('initial', initial))}
    for field in ('lane', 'index', 'split', 'phase', 'targets'):
        left, right = [data[field] for data in arrays.values()]
        equal = np.array_equal(left, right, equal_nan=True) if field == 'targets' else np.array_equal(left, right)
        if not equal:
            raise ValueError('initial/final target trajectories differ')
    output.mkdir(exist_ok=False)
    result = dict(protocol=PROTOCOL, status='running', corpus_sha256=sha(corpus_path),
                  source_sha256=sha(__file__), fits={}, actor_updates=0, native_imported=False,
                  limits=['Linear accessibility of teacher estimates, not all encoded information or causal policy diagnosis.',
                          'Teacher ball crop y90..185; unseen targets remain masked.',
                          'Selected development learner and reused states; readout test lanes are not learner confirmation.'])
    for name, data in arrays.items():
        result['fits'][name], predictions = fit_readouts(data)
        with (output/(name+'-predictions.npz')).open('xb') as stream:
            np.savez_compressed(stream, **predictions)
    require_cpu_only()
    result.update(status='complete', seconds=time.monotonic()-started,
                  capture_result_hashes={name: sha(path/'result.json') for name, path in (('final', final), ('initial', initial))},
                  prediction_hashes={name: sha(output/(name+'-predictions.npz')) for name in arrays})
    write(output/'result.json', result)
    print(json.dumps(dict(status='complete', seconds=result['seconds'], actor_updates=0)), flush=True)


def advance(agent, environment, row, *, last, action_count):
    """Consume one recorded transition, preserving real and collection boundaries."""
    action = row['action']
    if agent.act(action_mask=[i == action for i in range(action_count)]) != action:
        raise RuntimeError('forced source action changed')
    frame, reward, terminal, cutoff, _ = environment.step(action)
    if (reward, terminal, cutoff, environment.executed_action_frames) != (row['reward'], row['terminated'], row['truncated'], row['frames']):
        raise RuntimeError('recorded transition differs')
    cut = last and not terminal and not cutoff
    agent.observe(frame, extrinsic_reward=float(reward), terminated=terminal, truncated=cutoff or cut)
    if row['reset_after']:
        frame, _ = environment.reset()
        if not last:
            agent.begin_episode(frame)
    return frame, cut


def validate_capture_config(source_config, checkpoint_config, *, compare_centroid100=False):
    source, checkpoint = dict(source_config), dict(checkpoint_config)
    if compare_centroid100:
        expected = dict(scale=100.0, channel=0, background_scale=1.5,
                        region=[.05, .42857143, .95, .88095236])
        if (source.pop('color_centroid_supervision', None) is not None
                or checkpoint.pop('color_centroid_supervision', None) != expected):
            raise ValueError('requires an unsupervised source and fixed centroid100 checkpoint')
    if checkpoint != source or checkpoint_config.get('observation_kind') not in ('rgb32', 'rgb48', 'rgb64'):
        raise ValueError('checkpoint configuration differs from source policy')


def capture(corpus_path, checkpoint, manifest, output, smoke, *, compare_centroid100=False):
    import ale_py
    import gymnasium as gym
    import kindle
    from atari import DreamerAtariPreprocessing, checkpoint_identity
    from audit_atari import read_run
    from check_atari_adapter import rom_identity
    from fit_atari_probes import checked_memory
    from probe_fixed_latents import assert_frozen_tensors
    from probe_policy_forecasts import validate_checkpoint_config
    from replay_atari import verify_replay_identity

    corpus = json.loads(corpus_path.read_text())
    validate_corpus(corpus)
    source = read_run(Path(corpus['source_log']))
    if (source['sha256'] != corpus['source_log_sha256'] or source['start'] != corpus['source_header']
            or source['end'] != corpus['source_end'] or source['accounting']['updates']):
        raise ValueError('recorded source changed')
    header = source['start']
    checkpoint_config = json.loads((checkpoint/'metadata.json').read_text())['config']
    validate_capture_config(header['config'], checkpoint_config, compare_centroid100=compare_centroid100)
    gym.register_envs(ale_py)
    verify_replay_identity(header, json.loads(manifest.read_text()), rom_identity(header['environment']))
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    validate_checkpoint_config(checkpoint_config, agent.config)
    initial_updates, initial_actions = agent.learner_step, agent.environment_step
    result = dict(protocol=PROTOCOL, status='running', smoke=smoke, corpus_sha256=sha(corpus_path),
                  checkpoint=checkpoint_identity(checkpoint), source_sha256=sha(__file__),
                  native_sha256=sha(kindle._native.__file__), gpu_device=agent.gpu_device,
                  files=[], diagnostic_actions=0, actor_updates=0, collection_cuts=0, readback_checks=0,
                  belief='scalar re-observation; not original vector categorical draws',
                  compare_centroid100=compare_centroid100)

    def save():
        (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    def memory_check():
        memory = checked_memory(agent)
        headroom = memory['budget_bytes']-memory['usage_bytes']
        result['minimum_sampled_headroom_bytes'] = min(headroom, result.get('minimum_sampled_headroom_bytes', headroom))

    memory_check()
    save()
    try:
        for lane, records in enumerate(corpus['streams'][:1] if smoke else corpus['streams']):
            records = records[:128] if smoke else records
            environment = DreamerAtariPreprocessing(gym.make(header['environment'], frameskip=1,
                repeat_action_probability=header['sticky_actions'], full_action_space=header['full_action_space']),
                noop_max=0, max_episode_frames=header['max_episode_frames'], screen_size=None)
            frame, _ = environment.reset(seed=header['environment_seeds'][lane])
            agent.begin_episode(frame)
            values = dict(indices=[], cnn=[], posterior=[])
            pixel_hash = hashlib.sha256()
            try:
                for index, row in enumerate(records):
                    digest = hashlib.sha256(frame.tobytes(order='C')).hexdigest()
                    if digest != row['input_pixel_sha256']:
                        raise RuntimeError(f'input pixel mismatch at lane={lane}, index={index}')
                    pixel_hash.update(bytes.fromhex(digest))
                    if row['split'] is not None:
                        before = (agent.latent_feature, agent.posterior_action_probabilities(),
                                  agent.environment_step, agent.learner_step) if index == 0 else None
                        cnn, posterior = np.asarray(agent.encoded_observation, np.float32), np.asarray(agent.latent_feature, np.float32)
                        if not all(value.ndim == 1 and len(value) and np.isfinite(value).all() for value in (cnn, posterior)):
                            raise RuntimeError('invalid native feature readback')
                        if before is not None:
                            if (before != (agent.latent_feature, agent.posterior_action_probabilities(),
                                           agent.environment_step, agent.learner_step)
                                    or not np.array_equal(cnn, np.asarray(agent.encoded_observation, np.float32))):
                                raise RuntimeError('readback changed state or action probabilities')
                            result['readback_checks'] += 1
                        values['indices'].append(index)
                        values['cnn'].append(cnn)
                        values['posterior'].append(posterior)
                    frame, cut = advance(agent, environment, row, last=index+1 == len(records),
                                         action_count=header['config']['action_count'])
                    result['diagnostic_actions'] += 1
                    result['collection_cuts'] += int(cut)
                    if index % 256 == 0:
                        memory_check()
                if not smoke:
                    if (environment.executed_action_frames != corpus['source_end']['executed_action_frames'][lane]
                            or environment.emulator_resets != corpus['source_end']['emulator_resets'][lane]):
                        raise RuntimeError('final environment accounting differs')
                frames, resets = environment.executed_action_frames, environment.emulator_resets
            finally:
                environment.close()
            arrays = {key: np.asarray(value, dtype=np.int32 if key == 'indices' else np.float32) for key, value in values.items()}
            path = output/f'stream-{lane}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            result['files'].append(dict(file=path.name, sha256=sha(path), stream=lane, actions=len(records),
                frames=frames, resets=resets, examples=len(arrays['indices']), widths={key:arrays[key].shape[1] for key in ('cnn','posterior')},
                input_hash_sequence_sha256=pixel_hash.hexdigest()))
            save()
            print(json.dumps(dict(stream=lane, actions=result['diagnostic_actions'])), flush=True)
        after = output/'frozen-after'
        agent.save_checkpoint(str(after))
        result['unchanged_tensors'] = assert_frozen_tensors(checkpoint, after)
        expected = 128 if smoke else corpus['source_end']['run_step']
        if agent.learner_step != initial_updates or agent.environment_step-initial_actions != expected or result['diagnostic_actions'] != expected:
            raise RuntimeError('frozen agent accounting differs')
        memory_check()
        result.update(status='complete', seconds=time.monotonic()-started)
        save()
        print(json.dumps(dict(status='complete', actions=expected, seconds=result['seconds'])), flush=True)
    except BaseException as error:
        result.update(status='failed', reason=str(error), seconds=time.monotonic()-started)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    native = modes.add_parser('capture')
    for field in ('corpus', 'checkpoint', 'manifest', 'output'):
        native.add_argument(field, type=Path)
    native.add_argument('--smoke', action='store_true')
    native.add_argument('--compare-centroid100', action='store_true',
                        help='compare the fixed centroid100 checkpoint on the original control corpus')
    cpu = modes.add_parser('fit')
    for field in ('corpus', 'final', 'initial', 'output'):
        cpu.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args.corpus, args.checkpoint, args.manifest, args.output, args.smoke,
                compare_centroid100=args.compare_centroid100)
    else:
        fit(args.corpus, args.final, args.initial, args.output)
