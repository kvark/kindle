"""Standalone native Breakout world learner. Run GPU modes under gpu_host_guard.py.

The fitting contact clips are a capacity test. They are not held-out physics
qualification, gameplay evaluation, or evidence of later-wall coverage.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kindle._breakout_world import first_contact_window, image_metrics, training_windows


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def pixels(values):
    return np.asarray(values, dtype='<f4').tobytes()


def require_device(learner):
    from atari_vector import require_gpu_budget, require_gpu_device
    require_gpu_device(learner.gpu_device, 'AMD Radeon RX 7900 XT (RADV NAVI31)')
    require_gpu_budget(learner.gpu_memory_budget, 2 << 30)
    if hasattr(learner, 'parameter_names'):
        assert all(name.startswith('world.') for name in learner.parameter_names)


def fixture(config, output, prediction_length=0, pixel_loss="squared_error"):
    import kindle
    from safetensors.numpy import load_file
    learner = kindle.WorldModelLearner(config, prediction_length=prediction_length, pixel_loss=pixel_loss)
    require_device(learner)
    batch, length = config['batch_size'], config['batch_length']
    frames = [np.full((210, 160, 3), [64, 128, 192], np.uint8) for _ in range(batch)]
    actual = np.frombuffer(learner.preprocess(frames), '<f4').reshape(batch, 3, 64, 64)
    expected = np.broadcast_to(np.array([64, 128, 192], np.float32)[None, :, None, None]/255-.5, actual.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    observations = np.stack([np.roll(actual, i, axis=-1) for i in range(length)])
    actions = np.zeros((length, batch), np.int32)
    actions[0] = -1
    reward = np.zeros((length, batch), np.float32)
    reward[-1, 0] = 1
    terminal = np.zeros((length, batch), bool)
    terminal[-1, 0] = True
    learner.save(str(output/'initial'))
    initial = load_file(output/'initial/world.safetensors')
    reports = [learner.learn(pixels(observations), actions.ravel().tolist(), reward.ravel().tolist(), terminal.ravel().tolist())
               for _ in range(8)]
    if prediction_length:
        for values in reports:
            assert len(values) == 11 and values[10] == 1
            expected = (config['loss_scales']['reconstruction']*(values[1]+values[9])
                        +config['loss_scales']['dynamics']*values[2]
                        +config['loss_scales']['representation']*values[3]
                        +config['loss_scales']['reward']*values[4]
                        +config['loss_scales']['continuation']*values[5])
            np.testing.assert_allclose(values[0], expected, rtol=1e-5, atol=1e-4)
    prediction = learner.predict(pixels(observations[:2]), actions[:2].ravel().tolist(), [0]*(batch*4), seed=73001, include_features=True)
    assert np.frombuffer(prediction['prior_rgb'], '<f4').size == 4*batch*3*64*64
    assert len(prediction['prior_features']) == 4*len(prediction['posterior_features'])
    for name in ('encoder_features', 'posterior_features', 'prior_features'):
        assert np.isfinite(np.frombuffer(prediction[name], '<f4')).all()
    probability = np.asarray(prediction['reward_positive_probability'])
    assert probability.shape == (4, batch) and np.all((probability >= 0) & (probability <= 1))
    learner.save(str(output/'trained'))
    final = load_file(output/'trained/world.safetensors')
    changes = {}
    for prefix in ('world.representation.encoder.', 'world.representation.posterior.', 'world.dynamics.',
                   'world.decoder.', 'world.reward.', 'world.continuation.'):
        names = [name for name in initial if name.startswith(prefix)]
        assert names and all(np.isfinite(final[name]).all() for name in names)
        changes[prefix] = max(float(np.abs(final[name]-initial[name]).max()) for name in names)
        assert changes[prefix] > 0
    frozen = learner.predict(pixels(observations[:2]), actions[:2].ravel().tolist(), [0]*(batch*4), seed=73001, include_features=True)
    assert prediction == frozen and learner.updates == 8
    learner.save(str(output/'frozen'))
    after = load_file(output/'frozen/world.safetensors')
    assert initial.keys() == final.keys() == after.keys()
    assert all(np.array_equal(final[name], after[name]) for name in final)
    restored = kindle.WorldModelLearner.restore(str(output/'trained'))
    repeat = restored.predict(pixels(observations[:2]), actions[:2].ravel().tolist(), [0]*(batch*4), seed=73001, include_features=True)
    assert repeat == prediction and restored.updates == 8 and restored.prediction_length == prediction_length and restored.pixel_loss == pixel_loss
    write(output/'result.json', dict(status='complete', updates=8, no_behavior=True, changes=changes,
                                   reports=reports, prediction_restore_equal=True, frozen_tensors=len(final), prediction_length=prediction_length, pixel_loss=learner.pixel_loss))


def mask_fixture(config, output, prediction_length, pixel_loss="squared_error"):
    import kindle
    assert config['batch_length'] == 4 and prediction_length == 2
    (output/'unmasked').mkdir()
    fixture(config, output/'unmasked', prediction_length, pixel_loss)
    batch = config['batch_size']
    observations = np.full((4, batch, 3, 64, 64), -.25, np.float32)
    actions = np.zeros((4, batch), np.int32)
    actions[0] = actions[3] = -1
    terminals = np.zeros((4, batch), bool)
    terminals[2] = True
    rewards = np.zeros((4, batch), np.float32)
    reports = []
    for replacement in (0., .4):
        learner = kindle.WorldModelLearner(config, prediction_length=prediction_length, pixel_loss=pixel_loss)
        require_device(learner)
        changed = observations.copy()
        changed[3] = replacement
        reports.append(learner.learn(pixels(changed), actions.ravel().tolist(), rewards.ravel().tolist(), terminals.ravel().tolist()))
    np.testing.assert_allclose(reports[0][9], reports[1][9], rtol=1e-6, atol=1e-5)
    assert abs(reports[0][1]-reports[1][1]) > 1e-3
    assert reports[0][10] == reports[1][10] == .5
    learner = kindle.WorldModelLearner(config, prediction_length=prediction_length, pixel_loss=pixel_loss)
    actions[2], actions[3] = -1, 0
    terminals[2], terminals[1] = False, True
    all_masked = learner.learn(pixels(observations), actions.ravel().tolist(), rewards.ravel().tolist(), terminals.ravel().tolist())
    assert all_masked[9] == all_masked[10] == 0
    actions[1], actions[2:] = -1, 0
    terminals[:] = False
    terminals[0], rewards[0] = True, 7
    clipped_terminal = learner.learn(pixels(observations), actions.ravel().tolist(), rewards.ravel().tolist(), terminals.ravel().tolist())
    assert clipped_terminal[10] == 1
    write(output/'result.json', dict(status='complete', prediction_length=prediction_length, synthetic_updates=12,
          masked_target_reports=reports, all_masked_report=all_masked, clipped_terminal_report=clipped_terminal,
          target_changes_do_not_affect_prior_loss=True, pixel_loss=learner.pixel_loss))


def collect(config, output, *, action_pool=(0, 1, 3, 4), groups=None):
    import ale_py
    import gymnasium as gym
    import kindle
    from atari import DreamerAtariPreprocessing
    from kindle._breakout_guidance import FeedbackPixelPolicy
    from kindle._representation_probe import positions
    gym.register_envs(ale_py)
    learner = kindle.WorldModelLearner(config)
    require_device(learner)
    assert config['batch_size'] == 8
    if groups is None:
        groups = [('standard-fit', .25, 51001, 1024), ('debug-fit', 0., 71001, 512),
                  ('standard-development', .25, 91001, 512)]
    if not action_pool or any(type(value) is not int or not 0 <= value < 18 for value in action_pool):
        raise ValueError('nonempty full18 action pool required')
    summaries = []
    for name, sticky, base_seed, ticks in groups:
        root = output/name
        root.mkdir()
        environments = [DreamerAtariPreprocessing(gym.make('ALE/Breakout-v5', frameskip=1,
            repeat_action_probability=sticky, full_action_space=True), noop_max=0, screen_size=None,
            max_episode_frames=108000) for _ in range(8)]
        teachers = [FeedbackPixelPolicy() for _ in range(8)]
        rngs = [np.random.default_rng(base_seed+lane*1009+0x17a) for lane in range(8)]
        streams = [dict(rgb=[], native_rgb=[], action=[], reward=[], terminal=[], position=[], lives=[], frames=[],
                        episode=[], native_sha256=[]) for _ in range(8)]
        episodes = np.zeros(8, np.int32)
        frames = [environment.reset(seed=base_seed+lane*1009)[0] for lane, environment in enumerate(environments)]
        total_frames, blocks, command = 0, np.zeros(8, int), np.zeros(8, int)

        def append(indices, actions, rewards, terminals):
            rgb = np.frombuffer(learner.preprocess(frames), '<f4').reshape(8, 3, 64, 64)
            rgb8 = np.rint((rgb+.5)*255).clip(0, 255).astype(np.uint8)
            assert np.max(np.abs(rgb8.astype(np.float32)/255-.5-rgb)) < 1e-6
            for lane in indices:
                stream, environment = streams[lane], environments[lane]
                for key, value in dict(rgb=rgb8[lane].copy(), native_rgb=frames[lane].copy(), action=actions[lane], reward=rewards[lane],
                        terminal=terminals[lane], position=positions('Breakout', environment.unwrapped.ale.getRAM()),
                        lives=int(environment.unwrapped.ale.lives()), frames=environment.executed_action_frames,
                        episode=int(episodes[lane]), native_sha256=hashlib.sha256(frames[lane].tobytes()).hexdigest()).items():
                    stream[key].append(value)

        append(range(8), [-1]*8, [0.]*8, [False]*8)
        reset = []
        for tick in range(ticks):
            if reset:
                for lane in reset:
                    frames[lane] = environments[lane].reset()[0]
                    teachers[lane] = FeedbackPixelPolicy()
                    episodes[lane] += 1
                    blocks[lane] = 0
                append(reset, [-1]*8, [0.]*8, [False]*8)
            actions = []
            for lane, (teacher, rng) in enumerate(zip(teachers, rngs)):
                proposed = teacher.propose(frames[lane])
                if blocks[lane] == 0 and rng.random() < .025:
                    blocks[lane], command[lane] = int(rng.integers(4, 13)), int(rng.choice(action_pool))
                action = int(command[lane]) if blocks[lane] else int(proposed)
                if blocks[lane]: blocks[lane] -= 1
                teacher.feedback(action)
                actions.append(action)
            transitions = [environment.step(action) for environment, action in zip(environments, actions)]
            frames = [row[0] for row in transitions]
            rewards = [float(row[1]) for row in transitions]
            terminals = [bool(row[2]) for row in transitions]
            assert not any(row[3] for row in transitions), 'unexpected collection time limit'
            append(range(8), actions, rewards, terminals)
            reset = [lane for lane, terminal in enumerate(terminals) if terminal]
            if (tick+1) % 128 == 0:
                require_device(learner)
                print(json.dumps(dict(event='collection', group=name, ticks=tick+1, actions=8*(tick+1))), flush=True)
        files = []
        action_counts = np.zeros(18, int)
        for lane, stream in enumerate(streams):
            data = {key: np.asarray(value) for key, value in stream.items()}
            action_counts += np.bincount(data['action'][data['action'] >= 0], minlength=18)
            path = root/f'lane{lane}.npz'
            np.savez_compressed(path, **data)
            files.append(dict(path=str(path), sha256=sha(path), seed=base_seed+lane*1009,
                              arrivals=len(data['action']), reward=float(data['reward'].sum()),
                              game_overs=int(data['terminal'].sum()), episodes=int(episodes[lane])+1))
            total_frames += environments[lane].executed_action_frames
        for environment in environments: environment.close()
        summary = dict(name=name, sticky=sticky, actions=8*ticks, emulator_frames=total_frames, files=files,
                       selection='fixed controller with random command blocks; labels evaluation-only',
                       action_pool=list(action_pool), action_counts=action_counts.tolist())
        write(root/'manifest.json', summary)
        summaries.append(summary)
    assert learner.updates == 0
    write(output/'result.json', dict(status='complete', groups=summaries, updates=0))


def baseline(output, dataset, checkpoint):
    """Inspect existing RGB weights on the fitting clips, without learning."""
    import kindle
    from safetensors.numpy import load_file
    streams = [dict(np.load(dataset/f'lane{i}.npz', allow_pickle=False)) for i in range(8)]
    starts = [first_contact_window(stream) for stream in streams]
    batch = training_windows(streams, starts, 32)
    agent = kindle.Agent.restore_rgb(str(checkpoint))
    require_device(agent)
    before_updates = agent.learner_step
    agent.save_checkpoint(str(output/'before'))
    predictions, posteriors = [], []
    for stream, start in zip(streams, starts):
        native = stream['native_rgb'][start:start+32]
        actions = stream['action'][start:start+32]
        agent.begin_episode(native[0])
        for index in range(1, 32):
            if index == 16:
                posteriors.append(np.asarray(agent.observation_prediction(), np.float32).reshape(3, 64, 64))
                rows = []
                for horizon in (1, 4, 8, 16):
                    _, images = agent.forecast_states(actions[16:16+horizon].tolist())
                    rows.append(np.asarray(images[-1], np.float32).reshape(3, 64, 64))
                predictions.append(rows)
            mask = [action == int(actions[index]) for action in range(18)]
            assert agent.act(action_mask=mask) == int(actions[index])
            agent.observe(native[index], extrinsic_reward=float(stream['reward'][start+index]),
                          terminated=bool(stream['terminal'][start+index]), truncated=index == 31)
    agent.save_checkpoint(str(output/'after'))
    frozen_tensors = 0
    for filename in ('world.safetensors', 'behavior.safetensors', 'slow_value.safetensors'):
        before, after = (load_file(output/phase/filename) for phase in ('before', 'after'))
        assert before.keys() == after.keys()
        assert all(np.array_equal(before[name], after[name]) for name in before)
        frozen_tensors += len(before)
    assert agent.learner_step == before_updates
    truth = batch['rgb'].astype(np.float32)/255-.5
    prediction = np.asarray(predictions).transpose(1, 0, 2, 3, 4)
    result = dict(status='complete', scope='earlier RGB weights on eight fitting clips; new runtime, scalar re-observation',
        checkpoint=str(checkpoint), world_sha256=sha(checkpoint/'world.safetensors'), starts=starts,
        updates=0, frozen_tensors=frozen_tensors, posterior=image_metrics(truth[15], np.asarray(posteriors), truth[15]),
        horizons={str(h): image_metrics(truth[15+h], prediction[i], truth[15]) for i, h in enumerate((1, 4, 8, 16))},
        persistence={str(h): image_metrics(truth[15+h], truth[15], truth[15]) for h in (1, 4, 8, 16)})
    np.savez_compressed(output/'forecast.npz', truth=truth, prediction=prediction, posterior=np.asarray(posteriors))
    write(output/'result.json', result)


def evaluate(learner, batch, output, step):
    history = 16
    rgb = batch['rgb'].astype(np.float32)/255-.5
    target, context = rgb[history:], rgb[history-1]
    draws, predictions = [], []
    for seed in (83001, 83003, 83009, 83023):
        prediction = learner.predict(pixels(rgb[:history]), batch['action'][:history].ravel().tolist(),
                                     batch['action'][history:].ravel().tolist(), seed=seed)
        predicted = np.frombuffer(prediction['prior_rgb'], '<f4').reshape(target.shape)
        posterior = np.frombuffer(prediction['posterior_rgb'], '<f4').reshape(context.shape)
        row = dict(seed=seed, posterior=image_metrics(context, posterior, context), horizons={})
        for horizon in (1, 4, 8, 16):
            row['horizons'][str(horizon)] = image_metrics(target[horizon-1], predicted[horizon-1], context)
        draws.append(row)
        predictions.append(predicted)
    result = dict(step=step, scope='eight reused fitting clips; four fixed draws, not independent episodes', draws=draws,
        persistence={str(h): image_metrics(target[h-1], context, context) for h in (1, 4, 8, 16)})
    write(output/f'eval-{step}.json', result)
    np.savez_compressed(output/f'forecast-{step}.npz', truth=target, context=context, prediction=np.stack(predictions))
    print(json.dumps(dict(event='evaluation', step=step, first_draw=draws[0])), flush=True)
    return result


def overfit(config, output, dataset, steps, prediction_length=0, pixel_loss="squared_error"):
    import kindle
    from safetensors.numpy import load_file
    streams = [dict(np.load(dataset/f'lane{i}.npz', allow_pickle=False)) for i in range(8)]
    starts = [first_contact_window(stream) for stream in streams]
    batch = training_windows(streams, starts, config['batch_length'])
    write(output/'selection.json', dict(starts=starts, source_manifest=sha(dataset/'manifest.json'),
        training_only=True, labels_used_only_to_select_contact_candidates=True))
    learner = kindle.WorldModelLearner(config, prediction_length=prediction_length, pixel_loss=pixel_loss)
    require_device(learner)
    learner.save(str(output/'initial'))
    evaluate(learner, batch, output, 0)
    rgb = pixels(batch['rgb'].astype(np.float32)/255-.5)
    arguments = [batch[key].ravel().tolist() for key in ('action', 'reward', 'terminal')]
    started = time.monotonic()
    training_seconds = 0.
    with (output/'learning.jsonl').open('x') as log:
        for step in range(1, steps+1):
            tick = time.monotonic()
            losses = learner.learn(rgb, *arguments)
            training_seconds += time.monotonic()-tick
            log.write(json.dumps(dict(step=step, losses=losses, training_seconds=training_seconds))+'\n')
            if step % 128 == 0 or step == steps:
                log.flush()
                require_device(learner)
                evaluate(learner, batch, output, step)
                if step % 512 == 0 or step == steps:
                    learner.save(str(output/f'checkpoint-{step}'))
    initial = load_file(output/'initial/world.safetensors')
    final = load_file(output/f'checkpoint-{steps}/world.safetensors')
    names = [name for name in final if not name.startswith(('adam_', 'optimizer.'))]
    assert all(name.startswith('world.') for name in names)
    assert all(np.isfinite(value).all() for value in final.values())
    changed = sum(not np.array_equal(initial[name], final[name]) for name in names)
    assert learner.updates == steps and changed > 0
    write(output/'result.json', dict(status='complete', scope='capacity test on fitting contact clips', updates=steps,
        source_arrivals=int(np.prod(batch['action'].shape)), elapsed_seconds=time.monotonic()-started,
        training_seconds=training_seconds, changed_parameters=changed, no_behavior=True,
        held_out_competence_proven=False, prediction_length=prediction_length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fixture', 'mask-fixture', 'collect', 'baseline', 'overfit'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--prediction-length', type=int, default=0)
    parser.add_argument("--pixel-loss", choices=("squared_error", "cross_entropy", "motion_cross_entropy"), default="squared_error")
    args = parser.parse_args()
    if args.mode in ("collect", "baseline") and args.pixel_loss != "squared_error":
        parser.error("pixel-loss applies to training/fixtures only")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    write(args.output/'declaration.json', dict(mode=args.mode, config=config, steps=args.steps,
        source_sha256=sha(__file__), dataset=str(args.dataset), prediction_length=args.prediction_length, pixel_loss=args.pixel_loss, started_unix=time.time()))
    if args.mode == 'fixture': fixture(config, args.output, args.prediction_length, args.pixel_loss)
    elif args.mode == 'mask-fixture': mask_fixture(config, args.output, args.prediction_length, args.pixel_loss)
    elif args.mode == 'collect': collect(config, args.output)
    elif args.mode == 'baseline':
        if args.dataset is None or args.checkpoint is None: parser.error('baseline requires dataset and checkpoint')
        baseline(args.output, args.dataset, args.checkpoint)
    else:
        if args.dataset is None or args.steps <= 0: parser.error('overfit requires dataset and positive steps')
        overfit(config, args.output, args.dataset, args.steps, args.prediction_length, args.pixel_loss)


if __name__ == '__main__':
    main()
