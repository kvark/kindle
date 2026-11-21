"""CPU branches of retained Breakout states; no model inference or learning."""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import pickle
import select
import signal
import sys
import time

import numpy as np

HORIZON = 64


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixel_hash(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def select_origins(corpus, probabilities, phase_rows):
    lookup = {(r['stream'], r['episode'], r['step']): r for r in phase_rows}
    if len(lookup) != len(phase_rows) or len(lookup) != sum(map(len, corpus['streams'])):
        raise ValueError('complete unique phase rows required')
    count = len(probabilities['teacher'])
    for name in ('control', 'candidate'):
        p = probabilities[name]
        if (p.shape != (count, 18) or not np.isfinite(p).all() or (p <= 0).any()
                or not np.allclose(p.sum(1), 1, atol=1e-10, rtol=0)):
            raise ValueError('finite normalized full-action probabilities required')
    result, seen = [], set()
    for j, (lane, index, split, teacher) in enumerate(zip(
            probabilities['lane'], probabilities['index'], probabilities['split'],
            probabilities['teacher'], strict=True)):
        lane, index, teacher = int(lane), int(index), int(teacher)
        if (lane, index) in seen:
            raise ValueError('duplicate probability row')
        seen.add((lane, index))
        row = corpus['streams'][lane][index]
        phase = lookup[lane, row['episode'], row['step']]
        if (split != row['split'] or teacher != phase['label']
                or row['phase'] != phase['phase'] or row['action'] != phase['action']
                or row['input_pixel_sha256'] != phase['input_pixel_sha256']):
            raise ValueError('probability/corpus/teacher identity differs')
        if split != 'test' or row['phase'] != 'near_descending':
            continue
        result.append(dict(lane=lane, index=index, episode=row['episode'], step=row['step'],
            input_pixel_sha256=row['input_pixel_sha256'], teacher=teacher,
            choices={name:int(probabilities[name][j].argmax()) for name in ('control','candidate')},
            confidence={name:float(probabilities[name][j].max()) for name in ('control','candidate')}))
    return result


def comparison(origins, name, *, disagreements_only=False, confidence_min=0.0, lane=None):
    selected = [r for r in origins if (not disagreements_only or r['choices'][name] != r['teacher'])
                and r['confidence'][name] >= confidence_min and (lane is None or r['lane'] == lane)]
    life_delta, reward_delta = [], []
    for row in selected:
        policy = row['outcomes'][str(row['choices'][name])]
        teacher = row['outcomes'][str(row['teacher'])]
        life_delta.append(int(policy['life_lost']) - int(teacher['life_lost']))
        reward_delta.append(policy['reward'] - teacher['reward'])
    return dict(count=len(selected),
        policy_saves_life=sum(d < 0 for d in life_delta),
        same_life_outcome=sum(d == 0 for d in life_delta),
        teacher_saves_life=sum(d > 0 for d in life_delta),
        policy_more_reward=sum(d > 0 for d in reward_delta),
        same_reward=sum(d == 0 for d in reward_delta),
        teacher_more_reward=sum(d < 0 for d in reward_delta),
        mean_policy_minus_teacher_reward=float(np.mean(reward_delta)) if selected else None)


def require_cpu():
    if 'kindle._native' in sys.modules:
        raise RuntimeError('native agent must not be imported')
    for path in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        if target.startswith('/dev/dri/'):
            raise RuntimeError('CPU branches require no GPU context')


def continuation(environment, teacher, frame, first_action, recorded=None):
    """Apply one action, then teacher actions, stopping at the first lost life."""
    if teacher.pending is None:
        raise ValueError('origin proposal must be pending')
    initial_lives = int(environment.unwrapped.ale.lives())
    initial_frames = environment.executed_action_frames
    trajectory = []
    for step in range(HORIZON if recorded is None else min(HORIZON, len(recorded))):
        if step:
            teacher.propose(frame)
        action = recorded[step]['action'] if recorded is not None else first_action if step == 0 else teacher.pending
        before_hash = pixel_hash(frame)
        label = int(teacher.pending)
        frame, reward, terminal, cutoff, _ = environment.step(int(action))
        teacher.feedback(int(action))
        lives = int(environment.unwrapped.ale.lives())
        trajectory.append(dict(action=int(action), teacher=label, reward=float(reward),
            lives=lives, input_pixel_sha256=before_hash, output_pixel_sha256=pixel_hash(frame),
            frames=environment.executed_action_frames, terminated=terminal, truncated=cutoff))
        if terminal or cutoff or (recorded is None and lives < initial_lives):
            break
    if not trajectory:
        raise ValueError('empty continuation')
    return dict(initial_lives=initial_lives, final_lives=trajectory[-1]['lives'],
        life_lost=trajectory[-1]['lives'] < initial_lives,
        reward=sum(row['reward'] for row in trajectory), actions=len(trajectory),
        frames=environment.executed_action_frames-initial_frames,
        terminated=trajectory[-1]['terminated'], truncated=trajectory[-1]['truncated'],
        trajectory=trajectory)


def fork_continuation(environment, teacher, frame, first_action, recorded=None):
    """Copy the process so ALE sticky-action memory and teacher history survive."""
    require_cpu()
    before = environment.unwrapped.ale.cloneSystemState().serialize()
    teacher_before = pickle.dumps(teacher)
    reader, writer = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(reader)
        try:
            result = continuation(environment, teacher, frame, first_action, recorded)
            code = 0
        except BaseException as error:
            result, code = dict(error=repr(error)), 1
        with os.fdopen(writer, 'w') as stream:
            json.dump(result, stream, allow_nan=False)
        os._exit(code)
    os.close(writer)
    with os.fdopen(reader) as stream:
        ready, _, _ = select.select([stream], [], [], 5)
        if not ready:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
            raise TimeoutError('CPU branch exceeded five seconds')
        payload = stream.read()
    _, status = os.waitpid(child, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f'CPU branch failed: {payload}')
    if before != environment.unwrapped.ale.cloneSystemState().serialize() or teacher_before != pickle.dumps(teacher):
        raise RuntimeError('branching changed source state')
    return json.loads(payload)


def collect(declaration, output):
    from atari import DreamerAtariPreprocessing
    import ale_py
    import gymnasium as gym
    from kindle._breakout_guidance import FeedbackPixelPolicy
    import kindle._breakout_guidance as teacher_module
    from check_atari_adapter import rom_identity
    from replay_atari import verify_replay_identity
    started = time.monotonic()
    require_cpu()
    for path, digest in declaration['pins'].items():
        assert sha(path) == digest, path
    corpus = json.loads(Path(declaration['corpus']).read_text())
    phase_rows = [json.loads(line) for line in Path(declaration['phase_rows']).open()]
    labels = {(r['stream'],r['episode'],r['step']):r['label'] for r in phase_rows}
    header = corpus['source_header']
    gym.register_envs(ale_py)
    verify_replay_identity(header, json.loads(Path(declaration['manifest']).read_text()), rom_identity(header['environment']))
    assert sha(teacher_module.__file__) == declaration['teacher_sha256']
    selected = {(r['lane'],r['index']):r for r in declaration['origins']}
    assert len(selected) == len(declaration['origins']) == 136
    results, streams, branch_frames, validation_frames, reverse_checks = [], [], 0, 0, 0
    with (output/'origins.jsonl').open('x') as journal:
        for lane in sorted({lane for lane, _ in selected}):
            rows = corpus['streams'][lane]
            environment = DreamerAtariPreprocessing(gym.make(header['environment'], frameskip=1,
                repeat_action_probability=header['sticky_actions'], full_action_space=True),
                noop_max=0, max_episode_frames=header['max_episode_frames'], screen_size=None)
            frame, _ = environment.reset(seed=header['environment_seeds'][lane])
            teacher = FeedbackPixelPolicy()
            checked_order = False
            try:
                for index, row in enumerate(rows):
                    if time.monotonic()-started > declaration['analysis_cap_seconds']:
                        raise TimeoutError('CPU analysis cap reached; retain completed origins')
                    assert pixel_hash(frame) == row['input_pixel_sha256']
                    label = teacher.propose(frame)
                    assert label == labels[lane,row['episode'],row['step']]
                    if (lane,index) in selected:
                        origin = selected[lane,index]
                        assert label == origin['teacher']
                        recorded = fork_continuation(environment, teacher, frame, None, rows[index:index+HORIZON])
                        expected = rows[index:index+recorded['actions']]
                        for actual, reference in zip(recorded['trajectory'], expected, strict=True):
                            assert all(actual[k] == reference[k] for k in
                                ('action','reward','frames','terminated','truncated','input_pixel_sha256'))
                        validation_frames += recorded['frames']
                        choices = sorted({label, *origin['choices'].values()})
                        outcomes = {str(action):fork_continuation(environment, teacher, frame, action) for action in choices}
                        assert all(not r['truncated'] for r in outcomes.values())
                        branch_frames += sum(r['frames'] for r in outcomes.values())
                        if not checked_order:
                            reverse = {str(action):fork_continuation(environment, teacher, frame, action) for action in reversed(choices)}
                            assert reverse == outcomes
                            validation_frames += sum(r['frames'] for r in reverse.values())
                            reverse_checks += 1
                            checked_order = True
                        result = dict(**origin, outcomes=outcomes, recorded_validation=recorded)
                        journal.write(json.dumps(result, separators=(',',':'), allow_nan=False)+'\n')
                        journal.flush()
                        results.append(result)
                    frame, reward, terminal, cutoff, _ = environment.step(row['action'])
                    teacher.feedback(row['action'])
                    assert (reward,terminal,cutoff,environment.executed_action_frames) == (
                        row['reward'],row['terminated'],row['truncated'],row['frames'])
                    if row['reset_after']:
                        frame, _ = environment.reset()
                        teacher = FeedbackPixelPolicy()
                assert environment.executed_action_frames == corpus['source_end']['executed_action_frames'][lane]
                assert environment.emulator_resets == corpus['source_end']['emulator_resets'][lane]
                streams.append(dict(lane=lane, actions=len(rows), frames=environment.executed_action_frames,
                                    emulator_resets=environment.emulator_resets))
            finally:
                environment.close()
    assert len(results) == 136 and reverse_checks == 2
    for path,digest in declaration['pins'].items():
        assert sha(path) == digest, path
    require_cpu()
    metrics = {name:dict(all=comparison(results,name),
        disagreements=comparison(results,name,disagreements_only=True),
        confident_disagreements=comparison(results,name,disagreements_only=True,confidence_min=.75),
        very_confident_disagreements=comparison(results,name,disagreements_only=True,confidence_min=.9),
        per_lane={str(lane):comparison(results,name,disagreements_only=True,lane=lane) for lane in [6,7]})
        for name in ('control','candidate')}
    report = dict(status='complete', checked_unix=time.time(), seconds=time.monotonic()-started,
        pins_unchanged=len(declaration['pins']), origins=len(results), streams=streams,
        branches=sum(len(r['outcomes']) for r in results), branch_frames=branch_frames,
        validation_frames=validation_frames, reverse_order_checks=reverse_checks, metrics=metrics,
        origins_sha256=sha(output/'origins.jsonl'), native_imported=False, learner_updates=0,
        gpu_work=False, limits=declaration['limits'])
    write(output/'review.json', report)
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'], metrics=metrics)), flush=True)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kindle._native':
            raise RuntimeError('native learner imports are forbidden')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('declaration', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sys.meta_path.insert(0, NoNative())
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    collect(json.loads(args.declaration.read_text()), args.output)


if __name__ == '__main__':
    main()
