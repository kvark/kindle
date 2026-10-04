"""Frozen-state localization with native GPU readouts, no actor learning/gameplay."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from atari import sha256_file
from fit_atari_probes import bytes32, checked_memory, mlp_probe
from fit_fixed_latents import (HEAD_SEED, NAMES, normalization, predict, state_report,
                              validate_corpus)
from kindle import _native
from kindle._representation_probe import regression_metrics
from probe_fixed_latents import assert_frozen_tensors


def prefix(data, limit):
    count = min(len(data['actions']), limit)
    arrivals = int(data['following'][count - 1]) + 1
    return {key: value[:arrivals if key in ('features', 'positions', 'phase') else count]
            for key, value in data.items()}


def collect(core, data, seed):
    posterior = np.full((len(data['features']), len(core.latent_feature)), np.nan, np.float32)
    adapter = np.full((len(data['features']), len(core.encoded_observation)), np.nan, np.float32)
    rows = {h: dict(origins=[], prior=[], predicted=[], unrelated_prior=[], unrelated_predicted=[])
            for h in (1, 15)}
    rng = np.random.default_rng(HEAD_SEED ^ seed)
    step, start = core.learner_step, core.environment_step
    max_alignment_error = 0.

    def record(index):
        posterior[index] = core.latent_feature
        adapter[index] = core.encoded_observation

    for i, action in enumerate(data['actions']):
        if i == 0 or data['episodes'][i] != data['episodes'][i - 1]:
            core.begin_episode(bytes32(data['features'][data['current'][i]]))
            record(data['current'][i])
        first_prior = None
        if i % 4 == 0:
            horizon = 15 if i + 15 <= len(data['actions']) and data['episodes'][i] == data['episodes'][i + 14] else 1
            states, observations = core.forecast_states(data['actions'][i:i + horizon].tolist())
            unrelated_states, unrelated_observations = core.forecast_states(rng.integers(18, size=horizon).tolist())
            first_prior = np.asarray(states[0][:core.deterministic_size], np.float32)
            for h, endpoint in ((1, 0), (15, 1)):
                if h > horizon:
                    continue
                rows[h]['origins'].append(i)
                for key, values in (('prior', states), ('predicted', observations),
                                    ('unrelated_prior', unrelated_states), ('unrelated_predicted', unrelated_observations)):
                    rows[h][key].append(np.asarray(values[endpoint], np.float32))
        core.observe(int(action), bytes32(data['features'][data['following'][i]]), float(data['rewards'][i]),
                     bool(data['terminated'][i]), bool(data['truncated'][i] or i == len(data['actions']) - 1))
        record(data['following'][i])
        if first_prior is not None:
            error = float(np.abs(first_prior - posterior[data['following'][i], :core.deterministic_size]).max())
            max_alignment_error = max(max_alignment_error, error)
            if error > 2e-5:
                raise RuntimeError(f'h1 prior consumed its target or differs from live transition: {error}')
    if core.learner_step != step or core.environment_step - start != len(data['actions']):
        raise RuntimeError('frozen collection accounting changed')
    values = dict(posterior=posterior, adapter=adapter)
    values.update({f'{key}_h{h}': np.asarray(value, dtype=np.int64 if key == 'origins' else np.float32)
                   for h, row in rows.items() for key, value in row.items()})
    if any(not np.isfinite(value).all() for value in values.values()):
        raise RuntimeError('nonfinite/incomplete state capture')
    return values, max_alignment_error


def load_data(corpus, output, files, split, stage, horizon, deter, smoke):
    pieces = []
    for row in files:
        if row['split'] != split:
            continue
        with np.load(corpus / row['file']) as source, np.load(output / row['file']) as capture:
            data, states = dict(source), dict(capture)
        if smoke:
            data = prefix(data, 128)
        if horizon == 0:
            origins = np.arange(len(data['actions']))
            following = data['following']
            x = states[stage][following]
            controls = {}
            crossing = np.zeros(len(origins), bool)
        else:
            origins = states[f'origins_h{horizon}']
            following = data['following'][origins + horizon - 1]
            field = 'predicted' if stage == 'predicted' else 'prior'
            x, unrelated = (states[f'{p}{field}_h{horizon}'] for p in ('', 'unrelated_'))
            if stage == 'deter':
                x, unrelated = x[:, :deter], unrelated[:, :deter]
            current = data['current'][origins]
            controls = dict(unrelated=unrelated, current_posterior=states['posterior'][current],
                            future_posterior=states['posterior'][following],
                            current_tiny=data['features'][current], future_tiny=data['features'][following])
            crossing = data['phase'][current] + horizon >= 16
        labels = np.column_stack((data['positions'][following], data['rewards'][origins + max(horizon, 1) - 1] > 0,
                                  data['terminated'][origins + max(horizon, 1) - 1]))
        if smoke:
            labels = labels[:, :2]
        pieces.append(dict(x=x, labels=labels, origins=origins, crossing=crossing,
                           seeds=np.full(len(origins), row['seed']), **controls))
    return {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}


def report(prediction, data):
    if prediction.shape[1] == len(NAMES):
        return state_report(prediction, data)
    # The excluded short smoke only fits player coordinates.
    def metrics(p, y):
        return dict(zip(NAMES[:2], regression_metrics(p, y)))
    return dict(all=metrics(prediction, data['labels']), by_trajectory={str(seed): metrics(
        prediction[data['seeds'] == seed], data['labels'][data['seeds'] == seed]) for seed in np.unique(data['seeds'])})


def fit_readouts(corpus, output, files, deter, tiny_readout, smoke, result, save):
    with np.load(tiny_readout) as data:
        tiny_norm = {key: data[key] for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale')}
        tiny_parameters = [data[f'parameter_{i}'].tolist() for i in range(4)]
    model = _native.RegressionProbe(196, 2 if smoke else len(NAMES), hidden=128, batch=64, seed=HEAD_SEED)
    checked_memory(model)
    posterior_parameters = posterior_norm = None
    stages = [('adapter', 0), ('posterior', 0), *[(s, h) for h in (1, 15) for s in ('deter', 'prior', 'predicted')]]
    for stage, horizon in stages:
        started = time.monotonic()
        train, val, test = [load_data(corpus, output, files, split, stage, horizon, deter, smoke)
                            for split in ('train', 'validation', 'test')]
        norm = normalization(train['x'], train['labels'])
        prediction, info = mlp_probe(train['x'], train['labels'], val['x'], val['labels'], test['x'], HEAD_SEED,
                                     steps=32 if smoke else 2048, model=model, validation_interval=128)
        parameters = model.parameters()
        training_prediction = predict(model, train['x'], norm)
        row = dict(stage=stage, horizon=horizon, fit=info, input_width=train['x'].shape[1],
                   train_examples=len(train['x']), validation_examples=len(val['x']), test_examples=len(test['x']),
                   training=report(training_prediction, train), readouts=dict(fitted=report(prediction, test)),
                   training_normalized_mse=float(np.nanmean(np.nanmean(
                       np.square((training_prediction - train['labels']) / norm['y_scale']), axis=0))))
        evidence = dict(origins=test['origins'], seeds=test['seeds'], crossing=test['crossing'],
                        labels=test['labels'], fitted=prediction)
        controls = dict(training_mean=np.broadcast_to(norm['y_mean'], prediction.shape))
        if horizon:
            controls['unrelated_actions'] = predict(model, test['unrelated'], norm)
        if model.parameters() != parameters:
            raise RuntimeError('fitted readout changed during frozen predictions')
        if stage == 'posterior':
            posterior_parameters, posterior_norm = parameters, norm
        head = output / f'head-{stage}-h{horizon}.npz'
        np.savez(head, **norm, **{f'parameter_{i}': np.asarray(p, np.float32) for i, p in enumerate(parameters)})
        if horizon and stage == 'prior':
            model.reset(test['current_posterior'].shape[1], test['labels'].shape[1], HEAD_SEED)
            model.set_parameters(posterior_parameters)
            for name, x in (('posterior_to_prior', test['x']), ('posterior_to_unrelated', test['unrelated']),
                            ('posterior_persistence', test['current_posterior']), ('real_future_posterior', test['future_posterior'])):
                controls[name] = predict(model, x, posterior_norm)
            if model.parameters() != posterior_parameters:
                raise RuntimeError('frozen posterior readout changed')
        if horizon and stage == 'predicted':
            model.reset(3136, len(NAMES), HEAD_SEED)
            model.set_parameters(tiny_parameters)
            for name, x in (('tiny_to_prediction', test['x']), ('tiny_to_unrelated', test['unrelated']),
                            ('tiny_persistence', test['current_tiny']), ('real_future_tiny', test['future_tiny'])):
                controls[name] = predict(model, x, tiny_norm)[:, :test['labels'].shape[1]]
            if model.parameters() != tiny_parameters:
                raise RuntimeError('frozen Tiny readout changed')
        for name, values in controls.items():
            row['readouts'][name] = report(values, test)
            evidence[name] = values
        evidence_file = output / f'evidence-{stage}-h{horizon}.npz'
        np.savez(evidence_file, **evidence)
        row.update(head_sha256=sha256_file(head), evidence_file=evidence_file.name,
                   evidence_sha256=sha256_file(evidence_file), seconds=time.monotonic() - started)
        result['heads'].append(row)
        save()
        print(json.dumps(dict(fitted=stage, horizon=horizon, selected_step=info['selected_step'], seconds=row['seconds'])), flush=True)
    result['memory'].append(checked_memory(model))


def run(corpus, checkpoint, tiny_readout, output, smoke):
    started = time.monotonic()
    manifest = json.loads((corpus / 'manifest.json').read_text())
    validate_corpus(corpus, manifest)
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    if metadata['config']['video_encoder'] is not None or metadata['config']['model_size'] != 'size1_m':
        raise ValueError('frozen Size1M saved-feature checkpoint required')
    previous = json.loads((checkpoint.parent / 'result.json').read_text())
    head_result = json.loads((tiny_readout.parent / 'result.json').read_text())
    if (previous['status'] != 'complete' or not previous['standardize'] or previous['updates'] != 2048
            or previous['corpus_manifest_sha256'] != sha256_file(corpus / 'manifest.json')
            or head_result['corpus_manifest_sha256'] != previous['corpus_manifest_sha256']
            or head_result['heads'][0]['head_sha256'] != sha256_file(tiny_readout)):
        raise ValueError('matched standardized checkpoint/corpus/Tiny readout required')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(protocol='kindle-frozen-rssm-belief-probes-v1', status='running', smoke=smoke,
                  seed=metadata['config']['seed'], checkpoint=str(checkpoint.resolve()),
                  checkpoint_metadata_sha256=sha256_file(checkpoint / 'metadata.json'),
                  corpus=str(corpus.resolve()), corpus_manifest_sha256=sha256_file(corpus / 'manifest.json'),
                  tiny_readout_sha256=sha256_file(tiny_readout), source_sha256=sha256_file(__file__),
                  native_sha256=sha256_file(_native.__file__), new_game_actions=0, actor_updates=0, files=[], heads=[])

    def save():
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    core = _native.FeatureCore.restore(str(checkpoint))
    step, start = core.learner_step, core.environment_step
    result.update(gpu_device=core.gpu_device, memory=[checked_memory(core)], deterministic_size=core.deterministic_size)
    core.save_checkpoint(str(output / 'before'))
    assert_frozen_tensors(checkpoint, output / 'before')
    seen = set()
    for row in manifest['files']:
        if smoke and row['split'] in seen:
            continue
        seen.add(row['split'])
        with np.load(corpus / row['file']) as values:
            data = dict(values)
        if smoke:
            data = prefix(data, 128)
        states, error = collect(core, data, row['seed'])
        file = output / row['file']
        np.savez(file, **states)
        result['files'].append(dict(file=row['file'], seed=row['seed'], split=row['split'], sha256=sha256_file(file),
                                    actions=len(data['actions']), arrivals=len(data['features']), max_h1_deter_error=error))
        result['memory'].append(checked_memory(core))
        save()
        del data, states
    if core.learner_step != step or core.environment_step - start != (384 if smoke else 32768):
        raise RuntimeError('actor changed or corpus incomplete')
    core.save_checkpoint(str(output / 'after'))
    result.update(tensor_counts=assert_frozen_tensors(output / 'before', output / 'after'),
                  frozen_tensor_bytes_unchanged=True, actor_learner_step=core.learner_step,
                  recorded_actions=core.environment_step - start, collection_seconds=time.monotonic() - started)
    del core
    save()
    fit_readouts(corpus, output, result['files'], result['deterministic_size'], tiny_readout, smoke, result, save)
    result.update(status='complete', seconds=time.monotonic() - started)
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('tiny_readout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.corpus, args.checkpoint, args.tiny_readout, args.output, args.smoke)


if __name__ == '__main__':
    main()
