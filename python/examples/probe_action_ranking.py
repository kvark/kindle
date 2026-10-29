"""Frozen short action-block rankings with matched emulator branch outcomes.

CPU oracle generation and native prediction capture are separate processes.
Only current observations and proposed actions enter the native forecasts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import time

import numpy as np

from atari import DreamerAtariPreprocessing, checkpoint_identity, sha256_file
from audit_atari import read_run
from probe_policy_forecasts import stream_records, validate_checkpoint_config
from summarize_atari_comparison import cohort


HORIZON = 15
HOLD = 4


def ranking_metrics(truth, prediction, *, tolerance=1e-6):
    """Score ties fairly; report agreement only for unequal actual outcomes."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.ndim != 1 or len(truth) < 2 or prediction.shape != truth.shape or not np.isfinite([truth, prediction]).all():
        raise ValueError('matching finite candidate scores required')
    selected = prediction >= prediction.max()-tolerance
    best = truth >= truth.max()-tolerance
    left, right = np.triu_indices(len(truth), 1)
    difference = truth[left]-truth[right]
    useful = np.abs(difference) > tolerance
    predicted = prediction[left]-prediction[right]
    agreement = np.where(np.abs(predicted) <= tolerance, .5, (predicted*difference > 0).astype(float))
    chosen = float(truth[selected].mean())
    uniform = float(truth.mean())
    return dict(informative=bool(useful.any()), unequal_pairs=int(useful.sum()),
        pairwise_agreement=float(agreement[useful].mean()) if useful.any() else None,
        chosen_actual=chosen, uniform_actual=uniform, best_actual=float(truth.max()),
        regret=float(truth.max()-chosen), gain_over_uniform=chosen-uniform,
        best_action_fraction=float(best[selected].mean()), predicted_ties=int(selected.sum()))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_records(log, replay):
    run = read_run(log, allow_capped_evaluation=True)
    selected = cohort(run['episodes'], natural_only=True)
    outcomes = json.loads(Path(replay).read_text())
    if not selected['complete'] or run['accounting']['updates'] or outcomes['source_log_sha256'] != run['sha256']:
        raise ValueError('completed frozen source and matching replay required')
    with Path(log).open() as stream:
        records = stream_records(map(json.loads, stream), run['start']['num_envs'], selected['selected'], outcomes['episodes'])
    return run, records


def select_origins(records):
    """Four evenly spaced valid origins per phase and test episode, never by reward."""
    selected = []
    for stream, rows in enumerate(records):
        episodes = sorted({row['episode'] for row in rows if row['split'] == 1})
        for episode in episodes:
            for phase in (0, 1):
                eligible = [i for i in range(0, len(rows)-HORIZON+1, 16)
                    if rows[i]['episode'] == rows[i+HORIZON-1]['episode'] == episode
                    and rows[i]['split'] == 1 and rows[i]['phase'] == phase]
                if len(eligible) < 4:
                    raise ValueError('need four origins per selected episode/phase')
                for rank in np.linspace(0, len(eligible)-1, 4, dtype=int):
                    index = eligible[rank]
                    selected.append(dict(stream=stream, index=index, episode=episode, phase=phase))
    return selected


def plans_at(rows, index, action_count):
    recorded = [row['action'] for row in rows[index:index+HORIZON]]
    if len(recorded) != HORIZON:
        raise ValueError('incomplete recorded continuation')
    return [recorded] + [[action]*HOLD+recorded[HOLD:] for action in range(action_count)]


def branch_outcomes(environment, plans, *, reverse=False):
    """Fork CPU state per plan, including sticky-action memory absent from ALEState."""
    ale = environment.unwrapped.ale
    serialized = ale.cloneSystemState().serialize()
    actions = ale.getLegalActionSet()
    if len(actions) != len(environment.action_meanings):
        raise ValueError('full legal action mapping required')
    if not plans or any(len(plan) != HORIZON or any(type(a) is not int or not 0 <= a < len(actions) for a in plan) for plan in plans):
        raise ValueError('complete valid action plans required')
    results = [None]*len(plans)
    order = reversed(range(len(plans))) if reverse else range(len(plans))
    if any(os.readlink(fd).startswith('/dev/dri/') for fd in Path('/proc/self/fd').iterdir() if fd.exists()):
        raise RuntimeError('CPU oracle must run before any GPU context is constructed')
    for index in order:
        reader, writer = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(reader)
            try:
                rewards, terminals, cutoffs, frames = [], [], [], 0
                ended, terminal, cutoff = False, False, False
                for action in plans[index]:
                    reward = 0.0
                    if not ended:
                        for _ in range(4):
                            reward += float(ale.act(actions[action]))
                            frames += 1
                            terminal = bool(ale.game_over(with_truncation=False))
                            cutoff = bool(ale.game_truncated()) or environment._episode_frames+frames >= environment._max_episode_frames
                            if terminal or cutoff:
                                ended = True
                                break
                    rewards.append(reward)
                    terminals.append(terminal)
                    cutoffs.append(cutoff)
                result = dict(rewards=rewards, terminals=terminals, cutoffs=cutoffs, frames=frames)
                os.write(writer, json.dumps(result).encode())
                os.close(writer)
                os._exit(0)
            except BaseException as error:
                os.write(writer, json.dumps(dict(error=str(error))).encode())
                os.close(writer)
                os._exit(1)
        os.close(writer)
        with os.fdopen(reader) as pipe:
            ready, _, _ = select.select([pipe], [], [], 5)
            if not ready:
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
                raise TimeoutError('CPU branch exceeded five seconds')
            payload = pipe.read()
        _, status = os.waitpid(child, 0)
        if os.waitstatus_to_exitcode(status) != 0:
            raise RuntimeError(f'CPU branch failed: {payload}')
        results[index] = json.loads(payload)
    if ale.cloneSystemState().serialize() != serialized:
        raise RuntimeError('branching changed the source emulator or RNG')
    return results


def environment_for(header, stream):
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    environment = DreamerAtariPreprocessing(gym.make(header['environment'], frameskip=1,
        repeat_action_probability=header['sticky_actions'], full_action_space=True),
        noop_max=0, max_episode_frames=header['max_episode_frames'], screen_size=None)
    frame, _ = environment.reset(seed=header['environment_seeds'][stream])
    return environment, frame


def step_recorded(environment, row):
    frame, reward, terminal, cutoff, _ = environment.step(row['action'])
    if (reward, terminal, cutoff, environment.executed_action_frames) != (
            row['reward'], row['terminated'], row['truncated'], row['frames']):
        raise RuntimeError(f'recorded transition changed: actual={(reward, terminal, cutoff, environment.executed_action_frames)}, expected={(row["reward"], row["terminated"], row["truncated"], row["frames"])}')
    return frame


def oracle(log, replay, selection, output):
    run, records = source_records(log, replay)
    chosen = json.loads(selection.read_text())
    if chosen['source_log_sha256'] != run['sha256'] or chosen['origins'] != select_origins(records):
        raise ValueError('origin selection differs')
    started = time.monotonic()
    result = dict(status='running', source_log_sha256=run['sha256'], selection_sha256=sha256_file(selection),
        origins=[], streams=[], actions=0, branch_frames=0, reverse_order_checks=0)
    selected = {(r['stream'], r['index']):r for r in chosen['origins']}
    for stream, rows in enumerate(records):
        environment, _ = environment_for(run['start'], stream)
        digest = hashlib.sha256()
        try:
            for index, row in enumerate(rows):
                if (stream, index) in selected:
                    plans = plans_at(rows, index, run['start']['config']['action_count'])
                    outcomes = branch_outcomes(environment, plans)
                    actual = outcomes[0]
                    expected = rows[index:index+HORIZON]
                    assert actual['rewards'] == [r['reward'] for r in expected]
                    assert actual['terminals'] == [r['terminated'] for r in expected]
                    assert actual['cutoffs'] == [r['truncated'] for r in expected]
                    before_frames = rows[index-1]['frames'] if index else 0
                    assert actual['frames'] == expected[-1]['frames']-before_frames
                    if sum(r['stream']==stream for r in result['origins']) == 0:
                        assert branch_outcomes(environment, plans, reverse=True) == outcomes
                        result['reverse_order_checks'] += 1
                        result['branch_frames'] += sum(r['frames'] for r in outcomes)
                    result['origins'].append(dict(**selected[stream,index], plans=plans, outcomes=outcomes))
                    result['branch_frames'] += sum(r['frames'] for r in outcomes)
                frame = step_recorded(environment, row)
                digest.update(frame.tobytes())
                result['actions'] += 1
                if row['reset_after']:
                    environment.reset()
        finally:
            environment.close()
        result['streams'].append(dict(stream=stream, actions=len(rows), pooled_frame_sha256=digest.hexdigest()))
    result.update(status='complete', seconds=time.monotonic()-started, source_sha256=sha256_file(__file__))
    write(output, result)


def capture(log, replay, selection, checkpoint, output, *, smoke=False):
    import kindle
    from fit_atari_probes import checked_memory
    from probe_fixed_latents import assert_frozen_tensors
    run, records = source_records(log, replay)
    chosen = json.loads(selection.read_text())
    if chosen['source_log_sha256'] != run['sha256'] or chosen['origins'] != select_origins(records):
        raise ValueError('origin selection differs')
    output.mkdir(exist_ok=False)
    config = json.loads((checkpoint/'metadata.json').read_text())['config']
    validate_checkpoint_config(run['start']['config'], config)
    started = time.monotonic()
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    validate_checkpoint_config(config, agent.config)
    checked_memory(agent)
    initial_actions, initial_updates = agent.environment_step, agent.learner_step
    result = dict(status='running', smoke=smoke, source_log_sha256=run['sha256'],
        selection_sha256=sha256_file(selection), checkpoint=checkpoint_identity(checkpoint),
        native_sha256=sha256_file(kindle._native.__file__), source_sha256=sha256_file(__file__),
        gpu_device=agent.gpu_device, actions=0, updates=0, origins=[], streams=[])
    selected = {(r['stream'], r['index']):r for r in chosen['origins']}
    if smoke:
        selected = {(0,0):dict(stream=0,index=0,episode=0,phase=0)}
    try:
        for stream, rows in enumerate(records[:1] if smoke else records):
            rows = rows[:128] if smoke else rows
            environment, frame = environment_for(run['start'], stream)
            agent.begin_episode(frame)
            digest = hashlib.sha256()
            try:
                for index, row in enumerate(rows):
                    if (stream,index) in selected:
                        before = (agent.latent_feature, agent.posterior_action_probabilities(), agent.environment_step, agent.learner_step)
                        plans = plans_at(rows, index, config['action_count'])
                        predictions = []
                        for plan in plans:
                            reward, continuation, value = agent.prior_behavior_rollout(plan)
                            if not np.isfinite([reward,continuation,value]).all() or not all(0 <= c <= 1 for c in continuation):
                                raise RuntimeError('invalid branch prediction')
                            predictions.append(dict(rewards=reward,continuations=continuation,values=value))
                        assert before == (agent.latent_feature, agent.posterior_action_probabilities(), agent.environment_step, agent.learner_step)
                        repeated = agent.prior_behavior_rollout(plans[0])
                        assert repeated == (predictions[0]['rewards'],predictions[0]['continuations'],predictions[0]['values'])
                        result['origins'].append(dict(**selected[stream,index],plans=plans,predictions=predictions,probabilities=before[1]))
                    action = row['action']
                    assert agent.act(action_mask=[a==action for a in range(config['action_count'])]) == action
                    frame = step_recorded(environment,row)
                    digest.update(frame.tobytes())
                    cut = index+1==len(rows) and not row['terminated'] and not row['truncated']
                    agent.observe(frame,extrinsic_reward=float(row['reward']),terminated=row['terminated'],truncated=row['truncated'] or cut)
                    result['actions'] += 1
                    if row['reset_after']:
                        frame,_=environment.reset()
                        if index+1 < len(rows):
                            agent.begin_episode(frame)
                    if index%256==0:
                        memory=checked_memory(agent)
                        headroom=memory['budget_bytes']-memory['usage_bytes']
                        result['minimum_sampled_headroom_bytes']=min(headroom,result.get('minimum_sampled_headroom_bytes',headroom))
            finally:
                environment.close()
            result['streams'].append(dict(stream=stream,actions=len(rows),pooled_frame_sha256=digest.hexdigest()))
            (output/'progress.json').write_text(json.dumps(dict(actions=result['actions'],origins=len(result['origins'])))+'\n')
        agent.save_checkpoint(str(output/'frozen-after'))
        result['unchanged_tensors']=assert_frozen_tensors(checkpoint,output/'frozen-after')
        assert agent.learner_step==initial_updates and agent.environment_step-initial_actions==result['actions']
        assert result['actions']==(128 if smoke else run['end']['run_step'])
        result.update(status='complete',seconds=time.monotonic()-started)
        write(output/'result.json',result)
    except BaseException as error:
        result.update(status='failed',reason=str(error),seconds=time.monotonic()-started)
        write(output/'result.json',result)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('select','oracle','capture'))
    parser.add_argument('log',type=Path)
    parser.add_argument('replay',type=Path)
    parser.add_argument('selection',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.mode=='select':
        run,records=source_records(args.log,args.replay)
        write(args.selection,dict(source_log_sha256=run['sha256'],horizon=HORIZON,hold=HOLD,origins=select_origins(records)))
    elif args.mode=='oracle':
        oracle(args.log,args.replay,args.selection,args.output)
    else:
        capture(args.log,args.replay,args.selection,args.checkpoint,args.output,smoke=args.smoke)
