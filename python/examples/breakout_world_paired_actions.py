"""Collect fitting-only matched action branches with a frozen native camera.

Run under the host guard with a separately pinned parent manifest. Emulator
references annotate the dataset; only RGB and commands are model inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from breakout_world_branches import CATEGORIES, Snapshot, arrival, environment, restore_branch
from breakout_world_model import require_device, sha, write

FIELDS = ('rgb', 'native_rgb', 'action', 'reward', 'terminal', 'position',
          'lives', 'frames', 'episode', 'native_sha256')


def plans(category):
    if category == 'serve':
        return [(name, np.full(32, action, np.int32)) for name, action in
                [('noop', 0), ('fire', 1), ('leftfire', 12), ('rightfire', 11)]]
    return [('noop', np.zeros(32, np.int32)), ('left', np.full(32, 4, np.int32)),
            ('right', np.full(32, 3, np.int32)),
            ('reverse', np.r_[np.full(4, 3, np.int32), np.full(28, 4, np.int32)])]


def resize(camera, frames):
    values = np.frombuffer(camera.preprocess(frames), '<f4').reshape(8, 3, 64, 64)
    assert np.isfinite(values).all()
    rgb = np.rint((values+.5)*255).clip(0, 255).astype(np.uint8)
    assert np.max(np.abs(rgb.astype(np.float32)/255-.5-values)) < 1e-6
    return rgb


def combine(prefix, future):
    length = len(future['action'])
    assert 0 < length <= 32
    assert not np.asarray(future['terminal'])[:-1].any()
    assert not prefix['terminal'].any() and (prefix['action'][1:] >= 0).all()
    assert (np.asarray(future['action']) >= 0).all()
    result = {key: np.concatenate([prefix[key], np.asarray(future[key], dtype=prefix[key].dtype)])
              for key in FIELDS}
    assert all(len(value) == 32+length for value in result.values())
    assert np.all(np.diff(result['frames']) > 0)
    assert np.unique(result['episode']).size == 1
    return result


def reconstruct(env, parent, transcript):
    with np.load(parent['source'], allow_pickle=False) as data:
        source = {key: data[key] for key in
                  ['action', 'reward', 'terminal', 'position', 'lives', 'frames', 'native_sha256']}
    assert transcript['seed'] == parent['source_seed']
    assert transcript['commands'] == source['action'][1:parent['source_index']+1].tolist()
    frame = env.reset(seed=transcript['seed'])[0]
    actions = frames = 0
    resets = 1
    prefix_hashes = []
    for index in range(parent['source_index']+1):
        command = int(source['action'][index])
        reward, terminal = 0., False
        if index:
            if command == -1:
                frame = env.reset()[0]; resets += 1
            else:
                before = env.executed_action_frames
                frame, reward, terminal, truncated, _ = env.step(command)
                assert not truncated
                actions += 1; frames += env.executed_action_frames-before
        else:
            assert command == -1
        record = arrival(env, frame, command, reward, terminal)
        frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        assert frame_hash == source['native_sha256'][index], (parent['id'], index)
        assert reward == source['reward'][index] and terminal == source['terminal'][index]
        assert env.executed_action_frames == source['frames'][index]
        assert record['lives'] == source['lives'][index]
        np.testing.assert_array_equal(record['position'], source['position'][index])
        if index >= parent['source_index']-31: prefix_hashes.append(frame_hash)
    assert hashlib.sha256(''.join(prefix_hashes).encode('ascii')).hexdigest() == parent['native_history_fingerprint']
    assert actions == parent['source_actions'] and resets == parent['source_resets']+1
    return Snapshot.capture(env), dict(actions=actions, emulator_frames=frames, resets=resets)


def collect(args):
    import ale_py
    import kindle
    from safetensors.numpy import load_file
    started = time.monotonic()
    declaration = json.loads(args.declaration.read_text())
    for path, expected in declaration['inputs'].items(): assert sha(path) == expected, path
    parents = json.loads(args.parents.read_text())
    assert parents['status'] == 'prepared' and len(parents['parents']) == 56
    camera = kindle.WorldModelLearner.restore(str(args.checkpoint))
    require_device(camera)
    assert camera.updates == 0 and camera.pixel_loss == 'motion_cross_entropy'
    assert camera.prediction_length == 0
    camera.save(str(args.output/'before'))
    counts = {name: dict(actions=0, emulator_frames=0, resets=0) for name in ['source', 'replay', 'new']}
    files, prefix_checks = [], 0
    dataset = args.output/'dataset'; dataset.mkdir()
    for category in CATEGORIES:
        members = [p for p in parents['parents'] if p['category'] == category]
        assert len(members) == 8 and len({p['source_stream'] for p in members}) == 8
        envs = [environment() for _ in range(8)]
        try:
            prefixes, transcripts, snapshots = [], [], []
            for env, parent in zip(envs, members):
                transcript = json.loads(Path(parent['transcript']).read_text())
                snapshot, used = reconstruct(env, parent, transcript)
                snapshot.save(args.output/parent['id'])
                for key, value in used.items(): counts['source'][key] += value
                with np.load(parent['prefix'], allow_pickle=False) as data:
                    prefixes.append({key: data[key] for key in FIELDS})
                transcripts.append(transcript); snapshots.append(snapshot)
            for tick in range(32):
                actual = resize(camera, [prefix['native_rgb'][tick] for prefix in prefixes])
                for lane, prefix in enumerate(prefixes):
                    np.testing.assert_array_equal(actual[lane], prefix['rgb'][tick])
                    prefix_checks += 1
            for plan_name, commands in plans(category):
                for draw in range(2):
                    for env, snapshot, transcript in zip(envs, snapshots, transcripts):
                        snapshot.restore(env, rng=draw == 0)
                        desired = Snapshot.capture(env)
                        used = restore_branch(env, desired, snapshot, transcript['seed'], transcript['commands'])
                        for key, value in used.items(): counts['replay'][key] += value
                    frames = [prefix['native_rgb'][-1].copy() for prefix in prefixes]
                    streams = [{key: [] for key in FIELDS} for _ in envs]
                    active = np.ones(8, bool)
                    for command in commands:
                        records = {}
                        for lane in np.flatnonzero(active):
                            env = envs[lane]; before = env.executed_action_frames
                            frames[lane], reward, terminal, truncated, _ = env.step(int(command))
                            assert not truncated, 'unexpected collection truncation'
                            counts['new']['actions'] += 1
                            counts['new']['emulator_frames'] += env.executed_action_frames-before
                            record = arrival(env, frames[lane], int(command), reward, terminal)
                            record.update(frames=env.executed_action_frames,
                                episode=members[lane]['source_episode'],
                                native_sha256=hashlib.sha256(frames[lane].tobytes()).hexdigest())
                            records[lane] = record
                            if terminal: active[lane] = False
                        rgb = resize(camera, frames)
                        for lane, record in records.items():
                            record['rgb'] = rgb[lane].copy()
                            for key in FIELDS: streams[lane][key].append(record[key])
                        if not active.any(): break
                    for lane, (parent, prefix, stream) in enumerate(zip(members, prefixes, streams)):
                        path = dataset/f'{parent["id"]}-{plan_name}-draw{draw}.npz'
                        arrays = combine(prefix, stream)
                        np.testing.assert_array_equal(arrays['action'][32:], commands[:len(stream['action'])])
                        np.savez_compressed(path, **arrays)
                        files.append(dict(path=str(path), sha256=sha(path), parent=parent['id'],
                            category=category, plan=plan_name, draw=draw, shared_prefix=32,
                            future_arrivals=len(stream['action']), game_over=bool(arrays['terminal'][-1]),
                            reward=float(arrays['reward'][32:].sum())))
                    assert counts['source']['actions']+counts['replay']['actions'] <= 1_100_000
                    assert counts['source']['emulator_frames']+counts['replay']['emulator_frames'] <= 4_400_000
                    assert counts['new']['actions'] <= 14336 and counts['new']['emulator_frames'] <= 57344
                    assert camera.updates == 0
                    print(json.dumps(dict(category=category, plan=plan_name, draw=draw,
                        files=len(files), counts=counts, seconds=time.monotonic()-started)), flush=True)
        finally:
            for env in envs: env.close()
    assert len(files) == 448 and prefix_checks == 1792
    assert counts['source']['actions'] == parents['source_reconstruction_actions']
    assert counts['replay']['actions'] == 8*counts['source']['actions']
    assert sum(row['future_arrivals'] for row in files) == counts['new']['actions']
    camera.save(str(args.output/'after'))
    before, after = (load_file(args.output/phase/'world.safetensors') for phase in ['before', 'after'])
    assert before.keys() == after.keys() and len(before) == 204
    assert all(np.array_equal(before[key], after[key]) for key in before)
    write(dataset/'manifest.json', dict(files=files, counts=counts, split='fitting paired branches',
        ale_version=ale_py.__version__, labels_analysis_only=True, shared_prefix_arrivals=32))
    outputs = {str(path): sha(path) for path in args.output.rglob('*') if path.is_file()}
    write(args.output/'result.json', dict(status='complete', no_learning=True,
        predictions=0, updates=camera.updates, frozen_tensors=len(before), prefix_checks=prefix_checks,
        parents=56, files=len(files), counts=counts, outputs=outputs,
        wall_seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parents', 'checkpoint', 'declaration', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    collect(args)


if __name__ == '__main__': main()
