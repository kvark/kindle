"""Fixed-corpus native world training; run under gpu_host_guard.py.

Evaluation conditions on recorded future controls. It is retrospective prediction,
not a prospective planner or an Atari policy assessment. Labels are analysis-only.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from breakout_world_model import pixels, require_device, sha, write
from kindle._breakout_world import (
    evaluation_windows, event_metrics, first_contact_window, image_metrics,
    sample_locations, training_windows,
)


HISTORY = 16
HORIZONS = (1, 4, 8, 16, 32)
DRAWS = (83001, 83003, 83009, 83023)
FIELDS = ('rgb', 'action', 'reward', 'terminal', 'position', 'lives', 'frames', 'episode')
LOSS_NAMES = ('total', 'reconstruction', 'dynamics_kl', 'representation_kl', 'reward',
              'continuation', 'replay_value', 'raw_kl', 'future_prediction',
              'pixel_rollout', 'pixel_rollout_valid_fraction')
PREDICTIVE_OBJECTIVE = 'predictive_motion_energy'
SCORE_FUNCTION_OBJECTIVE = 'predictive_motion_energy_score_function'
LATENT_OBJECTIVE = 'motion_latent_overshooting'
RECURRENT_OBJECTIVE = 'motion_recurrent_energy'
FIXED_TEACHER_OBJECTIVE = 'motion_fixed_teacher'
PREDICTIVE_OBJECTIVES = (PREDICTIVE_OBJECTIVE, SCORE_FUNCTION_OBJECTIVE)


def report_names(pixel_loss, prediction_length):
    if pixel_loss == FIXED_TEACHER_OBJECTIVE:
        if prediction_length != 16: raise ValueError('fixed teacher requires H16')
        return LOSS_NAMES[:9]+('fixed_teacher_latent_kl','fixed_teacher_h_energy',
            'fixed_teacher_target_distance','fixed_teacher_sample_distance','fixed_teacher_valid_fraction')
    if pixel_loss == RECURRENT_OBJECTIVE:
        if not prediction_length: raise ValueError('recurrent energy requires a positive suffix length')
        return LOSS_NAMES[:9]+('recurrent_energy','recurrent_target_distance','recurrent_sample_distance','recurrent_valid_fraction')
    if pixel_loss == LATENT_OBJECTIVE:
        if not prediction_length: raise ValueError('latent overshooting requires a positive suffix length')
        return LOSS_NAMES[:9]+('latent_overshooting_kl', 'latent_rollout_valid_fraction')
    if pixel_loss in PREDICTIVE_OBJECTIVES:
        if prediction_length: raise ValueError('predictive motion uses one-step targets, not suffix rollouts')
        return LOSS_NAMES[:9]+('predictive_joint_energy', 'predictive_target_distance', 'predictive_sample_distance', 'predictive_valid_fraction')
    return LOSS_NAMES if prediction_length else LOSS_NAMES[:9]


def reported_total(config, values, pixel_loss, prediction_length):
    scales = config['loss_scales']
    if pixel_loss == FIXED_TEACHER_OBJECTIVE:
        return (scales['reconstruction']*(values[1]+values[10])
            +scales['dynamics']*(values[2]+values[9])+scales['representation']*values[3]
            +scales['reward']*values[4]+scales['continuation']*values[5])
    latent = pixel_loss == LATENT_OBJECTIVE
    recurrent = pixel_loss == RECURRENT_OBJECTIVE
    result = (scales['reconstruction']*(values[1]+(values[9] if prediction_length and not latent and not recurrent else 0))
        +scales['dynamics']*values[2]+scales['representation']*values[3]
        +scales['reward']*values[4]+scales['continuation']*values[5])
    if pixel_loss in PREDICTIVE_OBJECTIVES or recurrent:
        result += scales['reconstruction']*values[9]
    if latent:
        result += scales['dynamics']*values[9]
    return result


def load_dataset(root):
    manifest = json.loads((root/'manifest.json').read_text())
    streams = []
    for lane, item in enumerate(manifest['files']):
        path = root/f'lane{lane}.npz'
        if sha(path) != item['sha256']:
            raise ValueError(f'corpus changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            streams.append({key: data[key] for key in FIELDS})
    if len(streams) != 8:
        raise ValueError('eight source streams required')
    return streams, manifest


def choose_evaluation(streams, seed):
    natural = sample_locations(streams, HISTORY+1, 32, np.random.default_rng(seed))
    contacts, endings = [], []
    for lane, stream in enumerate(streams):
        try:
            contacts.append((lane, first_contact_window(stream)))
        except ValueError:
            pass
        endings.extend((lane, int(index)-HISTORY) for index in np.flatnonzero(stream['terminal']) if index >= HISTORY)
    return dict(natural=natural, contacts=contacts, endings=endings)


def rollout_validity(actions, prediction_length):
    if not 0 <= prediction_length < len(actions):
        raise ValueError('prediction length must leave an observed prefix')
    if prediction_length == 0:
        return np.zeros((0, actions.shape[1]), bool)
    return np.cumprod(actions[-prediction_length:] != -1, axis=0).astype(bool)


def sampler_plan(streams, config, steps, prediction_length=16):
    rng = np.random.default_rng(config['seed'] ^ 0x636f72707573)
    locations = np.asarray([sample_locations(streams, config['batch_length'], config['batch_size'], rng)
                            for _ in range(steps)], np.int32)
    return locations, plan_counts(streams, config, locations, prediction_length)


def plan_counts(streams, config, locations, prediction_length):
    seen = [np.zeros(len(stream['action']), bool) for stream in streams]
    total_terminal = prior_terminal = positive = context_resets = source_resets = 0
    for row in locations:
        actions = np.stack([streams[lane]['action'][start:start+32] for lane, start in row], axis=1).copy()
        terminals = np.stack([streams[lane]['terminal'][start:start+32] for lane, start in row], axis=1)
        rewards = np.stack([streams[lane]['reward'][start:start+32] for lane, start in row], axis=1)
        source_resets += int((actions == -1).sum())
        actions[0] = -1
        keep = rollout_validity(actions, prediction_length)
        total_terminal += int(terminals.sum())
        prior_terminal += int((terminals[len(actions)-prediction_length:] & keep).sum())
        positive += int((rewards > 0).sum())
        context_resets += int((actions == -1).sum())
        for lane, start in row: seen[lane][start:start+32] = True
    counts = dict(terminal_rows=total_terminal, prior_terminal_rows=prior_terminal, positive_reward_rows=positive,
                  context_reset_rows=context_resets, source_reset_rows=source_resets,
                  unique_source_arrivals=sum(int(row.sum()) for row in seen))
    return counts


def evaluate_group(learner, streams, locations, output, name, draws=DRAWS):
    if not locations:
        result = dict(count=0, unsupported=True)
        write(output/f'{name}.json', result)
        return result
    batch_size = learner.config['batch_size']
    rgb_parts, posterior_parts, reward_parts, positive_parts, continuation_parts = (
        [[] for _ in draws] for _ in range(5))
    truth, context, previous, validity, rewards, terminals, positions, lives = ([] for _ in range(8))
    encoded, posterior_features, prior_features = [], [], []
    before = learner.updates
    selected = np.asarray(HORIZONS)-1
    for offset in range(0, len(locations), batch_size):
        actual = locations[offset:offset+batch_size]
        count = len(actual)
        padded = list(actual)+[actual[0]]*(batch_size-count)
        data = evaluation_windows(streams, padded, history=HISTORY, horizon=max(HORIZONS))
        observed = data['rgb'][:HISTORY].astype(np.float32)/255-.5
        truth.append(data['rgb'][HISTORY+selected, :count])
        context.append(data['rgb'][HISTORY-1, :count])
        previous.append(data['rgb'][HISTORY-2, :count])
        validity.append(data['valid'][:, :count])
        rewards.append(data['reward'][HISTORY:, :count])
        terminals.append(data['terminal'][HISTORY:, :count])
        positions.append(data['position'][:, :count])
        lives.append(data['lives'][:, :count])
        for draw, seed in enumerate(draws):
            prediction = learner.predict(pixels(observed), data['action'][:HISTORY].ravel().tolist(),
                data['action'][HISTORY:].ravel().tolist(), seed=seed, include_features=draw == 0)
            rgb = np.frombuffer(prediction['prior_rgb'], '<f4').reshape(max(HORIZONS), batch_size, 3, 64, 64)
            rgb_parts[draw].append(rgb[selected, :count])
            posterior_parts[draw].append(np.frombuffer(prediction['posterior_rgb'], '<f4').reshape(batch_size, 3, 64, 64)[:count])
            reward_parts[draw].append(np.asarray(prediction['reward'], np.float32)[:, :count])
            positive_parts[draw].append(np.asarray(prediction['reward_positive_probability'], np.float32)[:, :count])
            continuation_parts[draw].append(np.asarray(prediction['continuation'], np.float32)[:, :count])
            if draw == 0:
                encoded.append(np.frombuffer(prediction['encoder_features'], '<f4').reshape(batch_size, -1)[:count])
                posterior_features.append(np.frombuffer(prediction['posterior_features'], '<f4').reshape(batch_size, -1)[:count])
                prior_features.append(np.frombuffer(prediction['prior_features'], '<f4').reshape(max(HORIZONS), batch_size, -1)[selected, :count])
    assert learner.updates == before
    arrays = dict(horizons=np.asarray(HORIZONS), locations=np.asarray(locations),
        truth=np.concatenate(truth, axis=1), context=np.concatenate(context), previous_context=np.concatenate(previous),
        valid=np.concatenate(validity, axis=1), target_reward=np.concatenate(rewards, axis=1),
        target_terminal=np.concatenate(terminals, axis=1), position=np.concatenate(positions, axis=1),
        lives=np.concatenate(lives, axis=1), prediction=np.stack([np.concatenate(v, axis=1) for v in rgb_parts]),
        posterior=np.stack([np.concatenate(v) for v in posterior_parts]),
        reward=np.stack([np.concatenate(v, axis=1) for v in reward_parts]),
        reward_positive_probability=np.stack([np.concatenate(v, axis=1) for v in positive_parts]),
        continuation=np.stack([np.concatenate(v, axis=1) for v in continuation_parts]),
        encoder_features=np.concatenate(encoded), posterior_features=np.concatenate(posterior_features),
        prior_features=np.concatenate(prior_features, axis=1))
    for key in ('prediction', 'posterior', 'reward', 'reward_positive_probability', 'continuation',
                'encoder_features', 'posterior_features', 'prior_features'):
        assert np.isfinite(arrays[key]).all(), key
    np.savez_compressed(output/f'{name}.npz', **arrays)
    context = arrays['context'].astype(np.float32)/255-.5
    result = dict(count=len(locations), source_streams=sorted({int(lane) for lane, _ in locations}),
        scope='retrospective recorded-control forecasts; four draws are not independent episodes',
        posterior=[image_metrics(context, image, context) for image in arrays['posterior']], horizons={})
    gamma = float(np.float32(1)-np.float32(1)/np.float32(learner.config['horizon']))
    for index, horizon in enumerate(HORIZONS):
        valid = arrays['valid'][horizon-1]
        target = arrays['truth'][index, valid].astype(np.float32)/255-.5
        actual_reward = arrays['target_reward'][horizon-1, valid]
        predicted_reward = arrays['reward'][:, horizon-1, valid].mean(axis=0)
        probability = arrays['reward_positive_probability'][:, horizon-1, valid].mean(axis=0)
        terminal_probability = 1-np.clip(arrays['continuation'][:, horizon-1, valid].mean(axis=0)/gamma, 0, 1)
        result['horizons'][str(horizon)] = dict(count=int(valid.sum()), excluded=int((~valid).sum()),
            images=[image_metrics(target, image[index, valid], context[valid]) for image in arrays['prediction']],
            persistence=image_metrics(target, context[valid], context[valid]),
            reward_mae=float(np.abs(predicted_reward-actual_reward).mean()) if valid.any() else None,
            rewarded_mae=float(np.abs(predicted_reward[actual_reward > 0]-actual_reward[actual_reward > 0]).mean()) if (actual_reward > 0).any() else None,
            positive_reward=event_metrics(actual_reward > 0, probability),
            terminal=event_metrics(arrays['target_terminal'][horizon-1, valid], terminal_probability))
    write(output/f'{name}.json', result)
    print(json.dumps(dict(event='evaluation', group=name, count=len(locations),
        ball={h:[x['ball_within_1px'] for x in row['images']] for h, row in result['horizons'].items()})), flush=True)
    return result


def train(args, config, fitting, development, *, sampler=sampler_plan, initial_reference=None,
          evaluation_fitting=None, learner=None, initial_checkpoint=None):
    import kindle
    from safetensors.numpy import load_file
    plan, expected_counts = sampler(fitting, config, args.steps, args.prediction_length)
    evaluation_fitting = fitting if evaluation_fitting is None else evaluation_fitting
    selections = dict(fitting=choose_evaluation(evaluation_fitting, 112003), development=choose_evaluation(development, 112019))
    if args.reference is not None:
        assert config == json.loads((args.reference/'declaration.json').read_text())['config']
        assert np.array_equal(plan, np.load(args.reference/'sampler-plan.npz', allow_pickle=False)['locations'])
        assert selections == {split:{kind:[tuple(x) for x in locations] for kind,locations in values.items()}
                              for split,values in json.loads((args.reference/'selections.json').read_text()).items()}
    np.savez_compressed(args.output/'sampler-plan.npz', locations=plan)
    write(args.output/'sampler-plan.json', expected_counts)
    if learner is None:
        learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length, pixel_loss=args.pixel_loss)
    require_device(learner)
    start_updates = learner.updates
    write(args.output/'actual-config.json', learner.config)
    write(args.output/'selections.json', selections)
    learner.save(str(args.output/'initial'))
    initial_reference = args.reference if args.reference is not None else initial_reference
    if initial_checkpoint is not None or initial_reference is not None:
        reference_file = (initial_checkpoint/'world.safetensors' if initial_checkpoint is not None
                          else initial_reference/'initial/world.safetensors')
        initial = load_file(args.output/'initial/world.safetensors')
        control = load_file(reference_file)
        assert initial.keys() == control.keys() and all(np.array_equal(initial[k], control[k]) for k in initial)
    if getattr(args, 'dense_initial', False):
        initial_output = args.output/'dense-initial'
        initial_output.mkdir()
        for label, streams, count, seed in [('fitting', evaluation_fitting, args.dense_fitting, 182003),
                                           ('development', development, args.dense_development, 182009)]:
            locations = sample_locations(streams, HISTORY+1, count, np.random.default_rng(seed))
            evaluate_group(learner, streams, locations, initial_output, label, draws=DRAWS[:1])
        learner.save(str(args.output/'initial-dense-frozen'))
        before = load_file(args.output/'initial/world.safetensors')
        after = load_file(args.output/'initial-dense-frozen/world.safetensors')
        assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
        assert learner.updates == start_updates
    seen = [np.zeros(len(stream['action']), bool) for stream in fitting]
    total_terminal, prior_terminal, positive_rows, context_reset_rows, source_reset_rows = 0, 0, 0, 0, 0
    native_seconds = 0.
    predictive_valid_rows = 0
    recurrent_valid_rows = 0
    started = time.monotonic()

    def evaluate(step):
        output = args.output/'evaluations'/str(step)
        output.mkdir(parents=True)
        for split, streams in [('fitting', evaluation_fitting), ('development', development)]:
            for kind, locations in selections[split].items():
                evaluate_group(learner, streams, locations, output, f'{split}-{kind}')

    evaluate(start_updates)
    with (args.output/'learning.jsonl').open('x') as log:
        for step in range(1, args.steps+1):
            locations = plan[step-1].tolist()
            streams = [fitting[lane] for lane, _ in locations]
            batch = training_windows(streams, [start for _, start in locations], config['batch_length'],
                                     allow_resets=True, retain_initial_targets=True)
            observed = pixels(batch['rgb'].astype(np.float32)/255-.5)
            tick = time.monotonic()
            values = learner.learn(observed, *[batch[key].ravel().tolist() for key in ('action', 'reward', 'terminal')])
            native_seconds += time.monotonic()-tick
            keep = rollout_validity(batch['action'], args.prediction_length)
            assert len(values) == len(report_names(args.pixel_loss, args.prediction_length))
            if args.prediction_length and args.pixel_loss not in (RECURRENT_OBJECTIVE,FIXED_TEACHER_OBJECTIVE): assert values[10] == float(keep.mean())
            if args.pixel_loss == FIXED_TEACHER_OBJECTIVE:
                assert values[13] == float(keep.mean())
                np.testing.assert_allclose(values[10],values[11]-.5*values[12],rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(values[0],reported_total(config,values,args.pixel_loss,args.prediction_length),rtol=1e-5)
            if args.pixel_loss == LATENT_OBJECTIVE or args.pixel_loss == 'motion_cross_entropy':
                np.testing.assert_allclose(values[0], reported_total(config, values, args.pixel_loss, args.prediction_length), rtol=1e-5)
            if args.pixel_loss in PREDICTIVE_OBJECTIVES or args.pixel_loss == RECURRENT_OBJECTIVE:
                valid = keep if args.pixel_loss == RECURRENT_OBJECTIVE else batch['action'] >= 0
                assert values[12] == float(valid.mean())
                if args.pixel_loss == RECURRENT_OBJECTIVE: recurrent_valid_rows += int(valid.sum())
                else: predictive_valid_rows += int(valid.sum())
                np.testing.assert_allclose(values[9], values[10]-.5*values[11], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(values[0], reported_total(config, values, args.pixel_loss, args.prediction_length), rtol=1e-5)
            terminals = int(batch['terminal'].sum())
            kept_terminals = int((batch['terminal'][len(batch['action'])-args.prediction_length:] & keep).sum())
            total_terminal += terminals
            prior_terminal += kept_terminals
            positive_rows += int((batch['reward'] > 0).sum())
            context_reset_rows += int((batch['action'] == -1).sum())
            source_reset_rows += sum(int((stream['action'][start:start+config['batch_length']] == -1).sum())
                                     for stream, (_, start) in zip(streams, locations))
            for lane, start in locations: seen[lane][start:start+config['batch_length']] = True
            log.write(json.dumps(dict(step=step, global_update=learner.updates, locations=locations, losses=values,
                terminal_rows=terminals, prior_terminal_rows=kept_terminals, training_seconds=native_seconds))+'\n')
            if step % 128 == 0:
                log.flush()
                print(json.dumps(dict(event='learning', step=step, losses=values, training_seconds=native_seconds,
                    unique_source_arrivals=sum(int(row.sum()) for row in seen))), flush=True)
                require_device(learner)
            if step in (2048, 4096) and step != args.steps:
                learner.save(str(args.output/f'checkpoint-{learner.updates}'))
            if step == args.steps: learner.save(str(args.output/'final'))
            if step % args.eval_every == 0 or step == args.steps: evaluate(learner.updates)
    final_extra = args.output/'dense-final'
    final_extra.mkdir()
    for label, streams, count, seed in [('fitting', evaluation_fitting, args.dense_fitting, 182003),
                                        ('development', development, args.dense_development, 182009)]:
        locations = sample_locations(streams, HISTORY+1, count, np.random.default_rng(seed))
        evaluate_group(learner, streams, locations, final_extra, label, draws=DRAWS[:1])
    learner.save(str(args.output/'final-frozen'))
    before, after = (load_file(args.output/phase/'world.safetensors') for phase in ('final', 'final-frozen'))
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
    assert all(np.isfinite(v).all() for v in after.values()) and learner.updates == start_updates+args.steps
    observed_counts = dict(terminal_rows=total_terminal, prior_terminal_rows=prior_terminal,
        positive_reward_rows=positive_rows, context_reset_rows=context_reset_rows, source_reset_rows=source_reset_rows,
        unique_source_arrivals=sum(int(row.sum()) for row in seen))
    assert observed_counts == expected_counts
    result = dict(status='complete', updates=learner.updates, start_updates=start_updates, new_updates=args.steps,
        prediction_length=args.prediction_length,
        training_seconds=native_seconds, total_seconds=time.monotonic()-started, no_behavior=True, pixel_loss=learner.pixel_loss,
        unique_source_arrivals=sum(int(row.sum()) for row in seen), source_arrivals=sum(len(row) for row in seen),
        replayed_terminal_rows=total_terminal, prior_terminal_rows=prior_terminal,
        positive_reward_rows=positive_rows, context_reset_rows=context_reset_rows,
        source_reset_rows=source_reset_rows, frozen_tensors=len(before),
        world_model_proven=False, untouched_test_scored=False)
    if args.pixel_loss in PREDICTIVE_OBJECTIVES: result['predictive_valid_rows'] = predictive_valid_rows
    if args.pixel_loss == RECURRENT_OBJECTIVE: result['recurrent_valid_rows'] = recurrent_valid_rows
    write(args.output/'result.json', result)


def restore_check(args, config, development, *, learner=None):
    import kindle
    if learner is None:
        learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
    require_device(learner)
    assert learner.pixel_loss == getattr(args, 'pixel_loss', 'squared_error')
    selection = json.loads((args.reference/'selections.json').read_text())['development']['natural'][:config['batch_size']]
    evaluate_group(learner, development, selection, args.output, 'restored')
    expected = np.load(args.reference/'evaluations'/str(learner.updates)/'development-natural.npz', allow_pickle=False)
    actual = np.load(args.output/'restored.npz', allow_pickle=False)
    assert np.array_equal(expected["horizons"], actual["horizons"])
    axes = dict(locations=0, truth=1, context=0, previous_context=0, valid=1, target_reward=1, target_terminal=1,
        position=1, lives=1, prediction=2, posterior=1, reward=2, reward_positive_probability=2,
        continuation=2, encoder_features=0, posterior_features=0, prior_features=1)
    for key, axis in axes.items():
        reference = np.take(expected[key], np.arange(len(selection)), axis=axis)
        assert np.array_equal(reference, actual[key], equal_nan=True), key
    write(args.output/'result.json', dict(status='complete', updates=learner.updates,
        exact_prediction_restore=True, checked_arrays=len(axes)+1, source_origins=len(selection), pixel_loss=learner.pixel_loss))


def initial_check(args, config, fitting, development, *, locations=None):
    """Verify matched initial tensors/traces and one full-shape finite update."""
    import kindle
    from safetensors.numpy import load_file
    if args.reference is None:
        raise ValueError('initial-check requires a reference')
    learner = kindle.WorldModelLearner(config, prediction_length=args.prediction_length, pixel_loss=args.pixel_loss)
    require_device(learner)
    learner.save(str(args.output/'initial'))
    actual = load_file(args.output/'initial/world.safetensors')
    expected = load_file(args.reference/'initial/world.safetensors')
    assert actual.keys() == expected.keys() and all(np.array_equal(actual[k], expected[k]) for k in actual)
    selection = json.loads((args.reference/'selections.json').read_text())['development']['natural'][:config['batch_size']]
    evaluate_group(learner, development, selection, args.output, 'candidate')
    expected_trace = np.load(args.reference/'evaluations/0/development-natural.npz', allow_pickle=False)
    actual_trace = np.load(args.output/'candidate.npz', allow_pickle=False)
    assert np.array_equal(expected_trace["horizons"], actual_trace["horizons"])
    axes = dict(locations=0, truth=1, context=0, previous_context=0, valid=1, target_reward=1, target_terminal=1,
        position=1, lives=1, prediction=2, posterior=1, reward=2, reward_positive_probability=2,
        continuation=2, encoder_features=0, posterior_features=0, prior_features=1)
    for key, axis in axes.items():
        assert np.array_equal(np.take(expected_trace[key], np.arange(len(selection)), axis=axis), actual_trace[key], equal_nan=True), key
    legacy = kindle.WorldModelLearner.restore(str(args.reference/'initial'))
    require_device(legacy)
    metadata = json.loads((args.reference/'initial/metadata.json').read_text())
    assert legacy.pixel_loss == metadata.get('pixel_loss', 'squared_error')
    assert legacy.prediction_length == metadata.get('prediction_length', 0)
    evaluate_group(legacy, development, selection, args.output, 'legacy')
    with np.load(args.output/'legacy.npz', allow_pickle=False) as trace:
        assert all(np.array_equal(trace[k], actual_trace[k], equal_nan=True) for k in (*axes, "horizons"))
    locations = [(i,0) for i in range(config['batch_size'])] if locations is None else locations
    batch = training_windows([fitting[lane] for lane,_ in locations], [start for _,start in locations], config['batch_length'],
                             allow_resets=True, retain_initial_targets=True)
    values = learner.learn(pixels(batch['rgb'].astype(np.float32)/255-.5),
        *[batch[key].ravel().tolist() for key in ('action', 'reward', 'terminal')])
    assert len(values) == len(report_names(args.pixel_loss, args.prediction_length)) and np.isfinite(values).all() and learner.updates == 1
    if args.pixel_loss in PREDICTIVE_OBJECTIVES or args.pixel_loss == RECURRENT_OBJECTIVE:
        valid = rollout_validity(batch['action'],args.prediction_length) if args.pixel_loss == RECURRENT_OBJECTIVE else batch['action'] >= 0
        assert values[12] == float(valid.mean())
        np.testing.assert_allclose(values[9], values[10]-.5*values[11], rtol=1e-5, atol=1e-5)
    expected_loss = reported_total(config, values, args.pixel_loss, args.prediction_length)
    np.testing.assert_allclose(values[0], expected_loss, rtol=1e-5)
    learner.save(str(args.output/'trained'))
    evaluate_group(learner, development, selection, args.output, 'trained')
    learner.save(str(args.output/'trained-frozen'))
    before, after = (load_file(args.output/phase/'world.safetensors') for phase in ('trained', 'trained-frozen'))
    assert all(np.array_equal(before[k], after[k]) for k in before)
    del legacy
    restored = kindle.WorldModelLearner.restore(str(args.output/'trained'))
    require_device(restored)
    assert restored.pixel_loss == learner.pixel_loss and restored.prediction_length == learner.prediction_length
    assert restored.updates == learner.updates == 1
    evaluate_group(restored, development, selection, args.output, 'trained-restored')
    with np.load(args.output/'trained.npz', allow_pickle=False) as first, np.load(args.output/'trained-restored.npz', allow_pickle=False) as second:
        assert all(np.array_equal(first[k], second[k], equal_nan=True) for k in first.files)
    write(args.output/'result.json', dict(status='complete', updates=1, pixel_loss=learner.pixel_loss,
        exact_initial_tensors=len(actual), exact_initial_arrays=len(axes)+1, exact_legacy_restore=True,
        first_full_shape_report=values, prediction_length=learner.prediction_length, loss_count=len(values),
        exact_trained_restore_arrays=len(axes)+1, frozen_tensors=len(before), world_model_proven=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'train', 'restore-check', 'initial-check'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--development', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--prediction-length', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=1024)
    parser.add_argument('--dense-fitting', type=int, default=256)
    parser.add_argument('--dense-development', type=int, default=128)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument("--pixel-loss", choices=("squared_error", "cross_entropy", "motion_cross_entropy", LATENT_OBJECTIVE, RECURRENT_OBJECTIVE, *PREDICTIVE_OBJECTIVES), default="squared_error")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.prediction_length not in (0, 16) or min(args.dense_fitting, args.dense_development) < 1:
        parser.error('positive training/evaluation budgets and prediction-length0/16 required')
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if config['batch_size'] != 8 or config['batch_length'] != 32:
        parser.error('this protocol requires B8/T32')
    if args.mode == 'collect':
        from breakout_world_model import collect
        groups = [('standard-fit', .25, 151001, 2048), ('standard-development', .25, 191001, 1024)]
        write(args.output/'declaration.json', dict(mode='collect', config=config, groups=groups,
            action_pool=list(range(18)), source_sha256=sha(__file__), started_unix=time.time()))
        collect(config, args.output, action_pool=tuple(range(18)), groups=groups)
        return
    if args.dataset is None or args.development is None: parser.error('dataset and development required')
    fitting, fit_manifest = load_dataset(args.dataset)
    development, dev_manifest = load_dataset(args.development)
    if min(fit_manifest.get('action_counts', [0]*18)) <= 0 or min(dev_manifest.get('action_counts', [0]*18)) <= 0:
        raise ValueError('all eighteen commands must be represented in fitting and development corpora')
    fit_seeds = {row['seed'] for row in fit_manifest['files']}
    assert not fit_seeds.intersection(row['seed'] for row in dev_manifest['files'])
    write(args.output/'declaration.json', dict(mode=args.mode, config=config, loss_names=report_names(args.pixel_loss, args.prediction_length),
        prediction_length=args.prediction_length, pixel_loss=args.pixel_loss, steps=args.steps, eval_every=args.eval_every,
        source_sha256=sha(__file__), fitting_manifest_sha256=sha(args.dataset/'manifest.json'),
        development_manifest_sha256=sha(args.development/'manifest.json'), started_unix=time.time()))
    if args.mode == 'initial-check': initial_check(args, config, fitting, development)
    elif args.mode == 'train': train(args, config, fitting, development)
    else:
        if args.checkpoint is None or args.reference is None: parser.error('restore-check needs checkpoint/reference')
        restore_check(args, config, development)


if __name__ == '__main__':
    main()
