"""Bounded behavior-coverage data and matched-init world-only learning.

All model processing remains native. Controller phases select fitting windows;
phase, lives and RAM measurements never enter model inputs or objectives.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import breakout_world_corpus as corpus
from breakout_world_model import require_device, sha, write

INITIAL_WAITS = (32, 48, 64, 96, 16, 32, 48, 64)
WAITS = (16, 32, 48, 64)
FIRE = {1, *range(10, 18)}


def waiting_command(lane, tick):
    if lane in (1, 5): return 4
    if lane in (2, 6): return 3
    if lane == 7: return 3 if tick//8 % 2 == 0 else 4
    return 0


def source_action(lane, tick, proposal, waiting, lives, rng, remaining, command):
    if waiting:
        return waiting_command(lane, tick), remaining, command
    if lane >= 4:
        return (1 if proposal == 1 else (4 if (lane+lives) % 2 else 3)), 0, command
    if remaining == 0 and rng.random() < .10:
        remaining, command = int(rng.choice([4, 8, 16, 32])), int(rng.integers(18))
    return (command if remaining else proposal), max(0, remaining-1), command


def collect(args):
    import kindle
    import ale_py
    from safetensors.numpy import load_file
    from breakout_world_branches import environment
    from kindle._breakout_guidance import FeedbackPixelPolicy
    from kindle._representation_probe import positions
    learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
    require_device(learner)
    assert learner.updates == 0 and learner.pixel_loss == 'motion_cross_entropy'
    learner.save(str(args.output/'before'))
    envs = [environment() for _ in range(8)]
    teachers = [FeedbackPixelPolicy() for _ in envs]
    seeds = [451001+1009*lane for lane in range(8)]
    rngs = [np.random.default_rng(seed ^ 0x636f766572616765) for seed in seeds]
    waiting = np.array(INITIAL_WAITS)
    remaining, command, episodes = (np.zeros(8, int) for _ in range(3))
    rows = [dict(rgb=[], native_rgb=[], action=[], reward=[], terminal=[], position=[], lives=[],
                 frames=[], episode=[], native_sha256=[], source_wait=[]) for _ in envs]
    started = time.monotonic()
    try:
        frames = [env.reset(seed=seed)[0] for env,seed in zip(envs,seeds)]
        lives = np.array([env.unwrapped.ale.lives() for env in envs])

        def append(indices, actions, rewards, terminals, wait_flags):
            values = np.frombuffer(learner.preprocess(frames), '<f4').reshape(8, 3, 64, 64)
            rgb = np.rint((values+.5)*255).clip(0, 255).astype(np.uint8)
            assert np.max(np.abs(rgb.astype(np.float32)/255-.5-values)) < 1e-6
            for lane in indices:
                env, stream = envs[lane], rows[lane]
                record = dict(rgb=rgb[lane].copy(), native_rgb=frames[lane].copy(), action=actions[lane],
                    reward=rewards[lane], terminal=terminals[lane], position=positions('Breakout',env.unwrapped.ale.getRAM()),
                    lives=int(env.unwrapped.ale.lives()), frames=env.executed_action_frames, episode=int(episodes[lane]),
                    native_sha256=sha_bytes(frames[lane].tobytes()), source_wait=bool(wait_flags[lane]))
                for key,value in record.items(): stream[key].append(value)

        append(range(8), [-1]*8, [0.]*8, [False]*8, [True]*8)
        reset = []
        for tick in range(args.ticks):
            if reset:
                for lane in reset:
                    frames[lane] = envs[lane].reset()[0]
                    teachers[lane] = FeedbackPixelPolicy()
                    lives[lane] = envs[lane].unwrapped.ale.lives()
                    episodes[lane] += 1
                    waiting[lane] = WAITS[(lane+episodes[lane]) % len(WAITS)]
                    remaining[lane] = 0
                append(reset, [-1]*8, [0.]*8, [False]*8, [True]*8)
            wait_flags = waiting > 0
            actions = []
            for lane,teacher in enumerate(teachers):
                proposal = teacher.propose(frames[lane])
                actual, remaining[lane], command[lane] = source_action(lane, tick, proposal, wait_flags[lane],
                    lives[lane], rngs[lane], remaining[lane], command[lane])
                actual = int(actual)
                teacher.feedback(actual)
                actions.append(actual)
                if waiting[lane]: waiting[lane] -= 1
            transitions = [env.step(action) for env,action in zip(envs,actions)]
            assert not any(row[3] for row in transitions), 'unexpected collection truncation'
            frames = [row[0] for row in transitions]
            rewards, terminals = [float(row[1]) for row in transitions], [bool(row[2]) for row in transitions]
            append(range(8), actions, rewards, terminals, wait_flags)
            for lane,env in enumerate(envs):
                current_lives = env.unwrapped.ale.lives()
                if current_lives < lives[lane] and not terminals[lane]:
                    waiting[lane] = WAITS[(lane+episodes[lane]+current_lives) % len(WAITS)]
                    remaining[lane] = 0
                lives[lane] = current_lives
            reset = [lane for lane,done in enumerate(terminals) if done]
            if (tick+1) % 128 == 0:
                print(json.dumps(dict(ticks=tick+1, actions=8*(tick+1), endings=sum(sum(s['terminal']) for s in rows))),flush=True)
                assert learner.updates == 0
        dataset = args.output/'dataset'; dataset.mkdir()
        files, counts = [], np.zeros(18, int)
        for lane,stream in enumerate(rows):
            data = {key:np.asarray(value) for key,value in stream.items()}
            path = dataset/f'lane{lane}.npz'; np.savez_compressed(path, **data)
            counts += np.bincount(data['action'][data['action'] >= 0], minlength=18)
            files.append(dict(path=str(path), sha256=sha(path), seed=seeds[lane], arrivals=len(data['action']),
                game_overs=int(data['terminal'].sum()), reward=float(data['reward'].sum())))
        write(dataset/'manifest.json',dict(files=files,action_counts=counts.tolist(),split='fitting supplement',
            ale_version=ale_py.__version__, source_actions=8*args.ticks,
            source_emulator_frames=sum(env.executed_action_frames for env in envs), labels_analysis_only=True))
        learner.save(str(args.output/'after'))
        a,b=(load_file(args.output/phase/'world.safetensors') for phase in ['before','after'])
        assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
        write(args.output/'result.json',dict(status='complete',updates=learner.updates,frozen_tensors=len(a),
            actions=8*args.ticks,emulator_frames=sum(env.executed_action_frames for env in envs),
            game_overs=sum(row['game_overs'] for row in files),wall_seconds=time.monotonic()-started,no_learning=True))
    finally:
        for env in envs: env.close()


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def sampling_pools(streams, length=32):
    if len(streams) != 16: raise ValueError('eight retained and eight supplement fitting streams required')
    waits, endings = [], []
    for lane,stream in enumerate(streams):
        action = stream['action']
        for terminal in np.flatnonzero(stream['terminal']):
            for offset in range(16,32):
                start = int(terminal)-offset
                if start >= 0 and start+length <= len(action) and not (action[start+1:terminal+1] == -1).any():
                    endings.append((lane,start))
        if lane < 8: continue
        for start in range(len(action)-length+1):
            part = action[start+1:start+length]
            if (stream['source_wait'][start:start+length].all() and not stream['terminal'][start:start+length].any()
                    and (part >= 0).all() and not np.isin(part,list(FIRE)).any()):
                waits.append((lane,start))
    if not waits or not endings: raise ValueError('waiting and terminal fitting pools must both be nonempty')
    return np.array(waits,np.int32),np.array(endings,np.int32)


def coverage_plan(streams, config, steps, prediction_length=0):
    if config['batch_size'] != 8 or config['batch_length'] != 32 or prediction_length not in (0,16):
        raise ValueError('coverage sampler requires B8/T32/H0 or H16')
    waits,endings = sampling_pools(streams)
    rng = np.random.default_rng(config['seed'] ^ 0x636f766572616765)
    plan = []
    for _ in range(steps):
        old = corpus.sample_locations(streams[:8],32,4,rng)
        new = [(lane+8,start) for lane,start in corpus.sample_locations(streams[8:],32,2,rng)]
        plan.append(old+new+[tuple(waits[rng.integers(len(waits))]),tuple(endings[rng.integers(len(endings))])])
    plan = np.asarray(plan,np.int32)
    return plan,corpus.plan_counts(streams,config,plan,prediction_length)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['collect','train','restore-check'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--supplement',type=Path)
    parser.add_argument('--development',type=Path)
    parser.add_argument('--initial-reference',type=Path)
    parser.add_argument('--sampling-plan',type=Path)
    parser.add_argument('--pixel-loss',choices=['motion_cross_entropy',corpus.LATENT_OBJECTIVE,corpus.RECURRENT_OBJECTIVE,*corpus.PREDICTIVE_OBJECTIVES],default='motion_cross_entropy')
    parser.add_argument('--prediction-length',type=int,choices=(0,16),default=0)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--ticks',type=int,default=2048)
    parser.add_argument('--steps',type=int,default=8192)
    parser.add_argument('--eval-every',type=int,default=1024)
    parser.add_argument('--dense-fitting',type=int,default=256)
    parser.add_argument('--dense-development',type=int,default=128)
    args=parser.parse_args()
    if (args.pixel_loss in (corpus.LATENT_OBJECTIVE,corpus.RECURRENT_OBJECTIVE)) != (args.prediction_length==16):parser.error('state suffix objectives require H16; other coverage objectives require H0')
    if min(args.ticks,args.steps,args.eval_every,args.dense_fitting,args.dense_development)<=0:parser.error('positive budgets required')
    if args.mode=='collect' and args.checkpoint is None:parser.error('collection requires actual-initial checkpoint')
    if args.mode=='collect' and args.pixel_loss!='motion_cross_entropy':parser.error('resize-only collection uses the retained actual-initial objective')
    if args.mode!='collect' and args.development is None:parser.error('development required')
    if args.mode=='train' and any(x is None for x in [args.dataset,args.supplement,args.initial_reference,args.sampling_plan]):parser.error('training requires both fitting corpora, actual-initial reference and declared sampling plan')
    if args.mode=='restore-check' and any(x is None for x in [args.checkpoint,args.reference]):parser.error('restore requires checkpoint and new training reference')
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text())
    write(args.output/'declaration.json',dict(mode=args.mode,config=config,source_sha256=sha(__file__),
        started_unix=time.time(),prediction_length=args.prediction_length,pixel_loss=args.pixel_loss,steps=args.steps,
        loss_names=corpus.report_names(args.pixel_loss,args.prediction_length),
        data_recipe='4 retained uniform + 2 supplement uniform + 1 supplement waiting + 1 terminal window'))
    if args.mode=='collect':collect(args);return
    development,_=corpus.load_dataset(args.development)
    if args.mode=='restore-check':corpus.restore_check(args,config,development);return
    fitting,old_manifest=corpus.load_dataset(args.dataset)
    new,new_manifest=corpus.load_dataset(args.supplement)
    for lane,stream in enumerate(new):
        with np.load(args.supplement/f'lane{lane}.npz') as data:stream['source_wait']=data['source_wait']
    assert not {r['seed'] for r in old_manifest['files']}.intersection(r['seed'] for r in new_manifest['files'])
    dev_manifest=json.loads((args.development/'manifest.json').read_text())
    assert not {r['seed'] for r in old_manifest['files']+new_manifest['files']}.intersection(r['seed'] for r in dev_manifest['files'])
    plan,counts=coverage_plan(fitting+new,config,args.steps,args.prediction_length)
    with np.load(args.sampling_plan,allow_pickle=False) as expected:
        assert np.array_equal(plan,expected['locations']), 'source plan differs from declaration'
    corpus.train(args,config,fitting+new,development,sampler=lambda *_:(plan,counts),initial_reference=args.initial_reference)


if __name__=='__main__':main()
