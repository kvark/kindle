"""CPU estimates of AGC/LaProp contributions from fixed captured gradients."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from safetensors import safe_open


TERMS = ('total', 'cdp', 'dynamics', 'representation', 'reward', 'continuation',
         'replay_value', 'disagreement', 'expert')
UPDATE_TERMS = ('total_update', 'history', 'current', *TERMS[1:])
GROUPS = ('encoder', 'posterior', 'dynamics', 'other')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require_cpu():
    if any(name in sys.modules for name in ('kindle._native', 'jax', 'torch')):
        raise RuntimeError('CPU-only optimizer analysis forbids native/ML runtimes')


def group(name):
    for prefix, result in (('world.representation.encoder.', 'encoder'),
                           ('world.representation.posterior.', 'posterior'),
                           ('world.dynamics.', 'dynamics')):
        if name.startswith(prefix):
            return result
    return 'other'


def effective_rate(config, name, learner_step):
    base = np.float32(config['learning_rate'])
    warmup = config['learning_rate_warmup']
    fraction = min(np.float32(learner_step) / np.float32(warmup), np.float32(1)) if warmup else np.float32(1)
    rate = base * fraction
    override = None
    if name.startswith('world.representation.encoder.'):
        override = config.get('encoder_learning_rate')
    elif name.startswith(('world.dynamics.', 'world.representation.posterior.',
                          'world.future_predictor.', 'world.exploration.')):
        override = config.get('dynamics_learning_rate')
    if override is not None:
        rate *= np.float32(override) / base
    return float(rate)


def transform(parameter, momentum, variance, gradients, config, name, step, learner_step):
    """Return F64 conditional contributions; never apply them to the inputs."""
    p, m, v = [np.asarray(values, dtype=np.float64).reshape(-1) for values in
               (parameter, momentum, variance)]
    raw = np.asarray(gradients, dtype=np.float64)
    if (raw.shape != (len(TERMS), p.size) or m.shape != p.shape or v.shape != p.shape
            or not all(np.isfinite(a).all() for a in (p, m, v, raw)) or (v < 0).any()
            or step <= 0):
        raise ValueError('invalid finite parameter, moment, gradient or step')
    b1, b2, eps, clip, pmin = [float(np.float32(config[key])) for key in
        ('optimizer_beta1', 'optimizer_beta2', 'optimizer_epsilon', 'agc', 'agc_pmin')]
    if not (0 <= b1 < 1 and 0 <= b2 < 1 and eps > 0 and clip >= 0 and pmin >= 0):
        raise ValueError('invalid optimizer settings')
    one1 = float(np.float32(1) - np.float32(b1))
    one2 = float(np.float32(1) - np.float32(b2))
    correction1 = float(np.float32(1 / (1 - b1 ** step)))
    correction2 = float(np.float32(1 / (1 - b2 ** step)))
    weight_norm = float(np.linalg.norm(p))
    gradient_norm = float(np.linalg.norm(raw[0]))
    upper = clip * max(pmin, weight_norm)
    scale = upper / gradient_norm if 0 < upper < gradient_norm else 1.0
    clipped = raw * scale
    next_variance = b2 * v + one2 * clipped[0] ** 2
    denominator = np.sqrt(next_variance * correction2) + eps
    rate = effective_rate(config, name, learner_step)
    current = (-rate * correction1 * one1) * clipped / denominator
    history = (-rate * correction1 * b1) * m
    updates = np.vstack((history + current[0], history, current))
    if not all(np.isfinite(a).all() for a in (clipped, updates)):
        raise ValueError('nonfinite optimizer estimate')
    return dict(raw=raw, clipped=clipped, updates=updates), dict(
        elements=p.size, group=group(name), weight_norm=weight_norm,
        effective_learning_rate=rate, agc_scale=scale, agc_upper=upper,
        old_variance_zero=int(np.count_nonzero(v == 0)),
        old_variance_subnormal=int(np.count_nonzero((v > 0) & (v < np.finfo(np.float32).tiny))),
        minimum_denominator=float(denominator.min()), maximum_denominator=float(denominator.max()))


def matrix_summary(matrix, terms):
    norms = np.sqrt(np.maximum(np.diag(matrix), 0))
    denominator = norms[:, None] * norms[None, :]
    cosines = np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
    return dict(terms=list(terms), norms=norms.tolist(), dot_products=matrix.tolist(),
                cosines=[[float(v) if d else None for v, d in zip(row, valid, strict=True)]
                         for row, valid in zip(cosines, denominator > 0, strict=True)])


def error_stats(reference, candidate):
    difference = candidate - reference
    return dict(error_squared=float(np.dot(difference, difference)),
                reference_squared=float(np.dot(reference, reference)),
                maximum_error=float(np.max(np.abs(difference))),
                maximum_reference=float(np.max(np.abs(reference))))


def combine_error(total, value):
    for key in ('error_squared', 'reference_squared'):
        total[key] = total.get(key, 0) + value[key]
    for key in ('maximum_error', 'maximum_reference'):
        total[key] = max(total.get(key, 0), value[key])


def qualify_error(values):
    relative = np.sqrt(values['error_squared']) / max(np.sqrt(values['reference_squared']), 1e-30)
    return dict(**values, relative_l2=float(relative),
                passed=bool(relative <= 1e-3 and values['maximum_error'] <= 1e-3 * (1 + values['maximum_reference'])))


def analyze_batch(directory, checkpoint, config, learner_step, saved_step):
    layout = json.loads((directory / 'gradient-layout.json').read_text())
    prior = json.loads((directory / 'result.json').read_text())
    if prior['attribution']['terms'] != list(TERMS) or not prior['component_sum_passed']:
        raise ValueError('qualified raw-gradient input is required')
    if len(layout) != len(dict(layout)) or len(layout) != 86:
        raise ValueError('expected the complete 86-parameter gradient layout')
    count = sum(size for _, size in layout)
    captures = []
    for term in TERMS:
        file = directory / (term + '.f32')
        if file.stat().st_size != 4 * count:
            raise ValueError('gradient size mismatch: ' + term)
        captures.append(np.memmap(file, dtype='<f4', mode='r'))
    matrices = {g: {stage: np.zeros((n, n)) for stage, n in
                   (('raw', len(TERMS)), ('clipped', len(TERMS)), ('updates', len(UPDATE_TERMS)))} for g in GROUPS}
    errors = {g: {} for g in (*GROUPS, 'all')}
    parameters = {}
    offset = 0
    with safe_open(checkpoint, framework='np') as tensors:
        expected = {name for name in tensors.keys() if 'adam_m.' + name in tensors.keys()}
        if expected != set(dict(layout)) or int(tensors.metadata()['adam_step']) != saved_step:
            raise ValueError('gradient and complete optimizer state differ')
        for name, size in layout:
            p, m, v = [tensors.get_tensor(prefix + name) for prefix in ('', 'adam_m.', 'adam_v.')]
            if p.size != size or m.shape != p.shape or v.shape != p.shape:
                raise ValueError('tensor layout differs: ' + name)
            raw = np.stack([values[offset:offset + size] for values in captures]).astype(np.float64)
            stages, statistics = transform(p, m, v, raw, config, name, saved_step + 1, learner_step)
            stage_matrices = {key: values @ values.T for key, values in stages.items()}
            for key, value in stage_matrices.items():
                matrices[group(name)][key] += value
            residual = error_stats(stages['updates'][2], stages['updates'][3:].sum(axis=0))
            for selected in (group(name), 'all'):
                combine_error(errors[selected], residual)
            np.testing.assert_array_equal(stages['updates'][0], stages['updates'][1] + stages['updates'][2])
            statistics['matrices'] = {key: matrix_summary(value, UPDATE_TERMS if key == 'updates' else TERMS)
                                      for key, value in stage_matrices.items()}
            previous = prior['attribution']['parameters'][name]
            if previous['group'] != statistics['group'] or previous['elements'] != size:
                raise ValueError('prior parameter grouping differs: ' + name)
            np.testing.assert_allclose(stage_matrices['raw'], previous['dot_products'], rtol=1e-9, atol=1e-8)
            statistics['component_sum'] = qualify_error(residual)
            statistics['update_to_weight_norm'] = (statistics['matrices']['updates']['norms'][0] /
                statistics['weight_norm']) if statistics['weight_norm'] else None
            parameters[name] = statistics
            offset += size
    groups = {}
    for selected in GROUPS:
        np.testing.assert_allclose(matrices[selected]['raw'],
            prior['attribution']['groups'][selected]['dot_products'], rtol=1e-9, atol=1e-8)
        selected_parameters = [p for p in parameters.values() if p['group'] == selected]
        scales = [p['agc_scale'] for p in selected_parameters]
        summary = {key: matrix_summary(value, UPDATE_TERMS if key == 'updates' else TERMS)
                   for key, value in matrices[selected].items()}
        weight_norm = np.sqrt(sum(p['weight_norm'] ** 2 for p in selected_parameters))
        groups[selected] = dict(matrices=summary, parameters=len(scales),
            clipped_parameters=sum(value < 1 for value in scales),
            scale_minimum=min(scales), scale_median=float(np.median(scales)), scale_maximum=max(scales),
            weight_norm=float(weight_norm), update_to_weight_norm=summary['updates']['norms'][0] / weight_norm)
    qualification = {key: qualify_error(value) for key, value in errors.items()}
    return dict(groups=groups, parameters=parameters, qualification=qualification,
                passed=all(value['passed'] for value in qualification.values()), elements=count)


def run(declaration_path, output):
    require_cpu()
    started = time.monotonic()
    declaration = json.loads(declaration_path.read_text())
    for path, digest in declaration['pins'].items():
        if sha(path) != digest:
            raise ValueError('input pin differs: ' + path)
    source = Path(declaration['gradient_root'])
    metadata = json.loads((source / 'native/before/metadata.json').read_text())
    if (metadata['learner_step'], metadata['environment_step']) != (12477, 200000):
        raise ValueError('unexpected saved learner counters')
    checkpoint = source / 'native/before/world.safetensors'
    with safe_open(checkpoint, framework='np') as tensors:
        saved_step = int(tensors.metadata()['adam_step'])
    if saved_step != 12477:
        raise ValueError('unexpected optimizer step')
    output.mkdir(exist_ok=False)
    result = dict(status='running',source_commit=declaration['source_commit'], saved_step=saved_step,
        learner_updates=0, native_imported=False, batches=[], config=metadata['config'],
        limits=['F64 estimate, not measured native parameter movement or gameplay.',
                'Both batches independently use the same saved weights and moments.',
                'Component transforms hold total-gradient clipping and variance fixed; not independent optimizer steps.'])
    for index in range(2):
        batch = analyze_batch(source / f'native/batch-{index}', checkpoint, metadata['config'], 12477, saved_step)
        batch['index'] = index
        write(output / f'batch-{index}.json', batch)
        result['batches'].append({key: value for key, value in batch.items() if key != 'parameters'})
        if not batch['passed']:
            raise ValueError('transformed component sum failed; preserve result')
    require_cpu()
    for path, digest in declaration['pins'].items():
        if sha(path) != digest:
            raise ValueError('input changed during analysis: ' + path)
    result.update(status='complete',seconds=time.monotonic()-started,pins_unchanged=len(declaration['pins']))
    write(output / 'result.json', result)
    print(json.dumps(dict(status='complete',seconds=result['seconds'],pins=result['pins_unchanged'],learner_updates=0)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('declaration', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.declaration, args.output)
