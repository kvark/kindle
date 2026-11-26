"""CPU RBF action readouts on existing frozen Breakout captures."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from probe_breakout_actions import ACTIONS, attach_actions, evaluate
from probe_breakout_representation import read_features, require_cpu_only, sha, write


PROTOCOL = 'kindle-breakout-kernel-actions-v1'
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)


def squared_distances(left, right):
    return np.maximum(np.square(left).sum(1)[:, None]
                      + np.square(right).sum(1)[None, :] - 2 * left @ right.T, 0.)


def kernel_probe(train_x, train_y, validation_x, validation_y, test_x, alphas=ALPHAS):
    """Train-only normalization/bandwidth, centered RBF, validation-only penalties.

    Test labels are absent. Each one-hot target uses the same validation MSE
    selection rule as the existing linear readout; validation rows never refit.
    """
    xs = [np.asarray(x, dtype=np.float64) for x in (train_x, validation_x, test_x)]
    ys = [np.asarray(y, dtype=np.float64) for y in (train_y, validation_y)]
    if (any(x.ndim != 2 or not np.isfinite(x).all() for x in xs)
            or min(len(xs[0]), len(xs[1])) < 2 or not xs[0].shape[1]
            or any(x.shape[1] != xs[0].shape[1] for x in xs)
            or any(y.ndim != 2 or not np.isfinite(y).all() for y in ys)
            or ys[0].shape != (len(xs[0]), ys[1].shape[1]) or len(ys[1]) != len(xs[1])
            or not ys[0].shape[1] or not alphas
            or any(a <= 0 or not np.isfinite(a) for a in alphas)):
        raise ValueError('invalid kernel readout inputs')
    mean, scale = xs[0].mean(0), xs[0].std(0)
    scale = np.where(scale > 1e-6, scale, 1.) * np.sqrt(xs[0].shape[1])
    xs = [(x - mean) / scale for x in xs]
    distances = squared_distances(xs[0], xs[0])
    off_diagonal = distances[np.triu_indices(len(xs[0]), k=1)]
    positive = off_diagonal[off_diagonal > 1e-12]
    if not len(positive):
        raise ValueError('constant training features have no kernel bandwidth')
    bandwidth = float(np.median(positive))
    gram = np.exp(-distances / bandwidth)
    columns, average = gram.mean(0), gram.mean()
    centered = gram - columns[None, :] - columns[:, None] + average
    eigenvalues, vectors = np.linalg.eigh(centered)
    eigenvalues = np.maximum(eigenvalues, 0.)
    kernels = []
    for x in xs[1:]:
        kernel = np.exp(-squared_distances(x, xs[0]) / bandwidth)
        kernels.append(kernel - columns[None, :] - kernel.mean(1)[:, None] + average)
    val_kernel, test_kernel = kernels
    mean_y = ys[0].mean(0)
    basis = vectors.T @ (ys[0] - mean_y)
    predictions = np.empty((len(xs[2]), ys[0].shape[1]))
    selections = []
    for target in range(ys[0].shape[1]):
        coefficients = vectors @ (basis[:, target, None]
                                 / (eigenvalues[:, None] + np.asarray(alphas)[None, :]))
        val_predictions = val_kernel @ coefficients + mean_y[target]
        errors = np.square(val_predictions - ys[1][:, target, None]).mean(0)
        selected = int(np.argmin(errors))
        predictions[:, target] = test_kernel @ coefficients[:, selected] + mean_y[target]
        selections.append(dict(alpha=alphas[selected], validation_mse=errors.tolist()))
    if not np.isfinite(predictions).all():
        raise ValueError('nonfinite kernel predictions')
    return predictions, dict(bandwidth_squared=bandwidth, selection=selections)


def fit_actions(data, teacher, actual):
    if (len(teacher) != len(data['lane']) or len(actual) != len(teacher)
            or not np.isin(teacher, ACTIONS).all()):
        raise ValueError('invalid aligned action labels')
    masks = {name: data['split'] == name for name in ('train', 'validation', 'test')}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('missing action readout split')
    train, validation, test = masks.values()
    if (set(data['lane'][train]) - {0, 1, 2, 3}
            or set(data['lane'][validation]) - {4, 5}
            or set(data['lane'][test]) - {6, 7}):
        raise ValueError('lane roles differ')
    counts = np.array([(teacher[train] == action).sum() for action in ACTIONS])
    majority = int(ACTIONS[np.argmax(counts)])
    labels = (teacher[:, None] == ACTIONS).astype(float)
    result = dict(counts={key: int(mask.sum()) for key, mask in masks.items()},
                  training_class_counts=dict(zip(map(str, ACTIONS), map(int, counts))),
                  majority_action=majority, baselines={}, variants={})
    predictions = dict(majority=np.full(test.sum(), majority), recorded=actual[test])
    for name, values in predictions.items():
        result['baselines'][name] = evaluate(teacher[test], values, data['lane'][test], data['phase'][test])
    features = {key: data[key] for key in ('cnn', 'posterior')}
    features['combined'] = np.concatenate([features['cnn'], features['posterior']], axis=1)
    for name, values in features.items():
        scores, selection = kernel_probe(values[train], labels[train], values[validation],
                                         labels[validation], values[test])
        predictions[name] = ACTIONS[np.argmax(scores, axis=1)]
        predictions[name + '_scores'] = scores
        result['variants'][name] = dict(width=values.shape[1], **selection,
            metrics=evaluate(teacher[test], predictions[name], data['lane'][test], data['phase'][test]))
    return result, predictions


def run(features_root, phase_path, output):
    require_cpu_only()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    started = time.monotonic()
    corpus_path = features_root / 'corpus.json'
    corpus = json.loads(corpus_path.read_text())
    if corpus['phase_rows_sha256'] != sha(phase_path):
        raise ValueError('phase source identity differs')
    with phase_path.open() as source:
        phase_rows = [json.loads(line) for line in source]
    arrays = {}
    for name in ('final', 'initial'):
        capture = json.loads((features_root / name / 'result.json').read_text())
        if capture['corpus_sha256'] != sha(corpus_path):
            raise ValueError('capture corpus differs')
        arrays[name] = read_features(corpus, features_root / name)
    for field in ('lane', 'index', 'split', 'phase'):
        if not np.array_equal(arrays['final'][field], arrays['initial'][field]):
            raise ValueError('initial/final row identity differs')
    output.mkdir(exist_ok=False)
    result = dict(protocol=PROTOCOL, status='running', source_sha256=sha(__file__),
                  corpus_sha256=sha(corpus_path), phase_rows_sha256=sha(phase_path),
                  alphas=ALPHAS, fits={}, actor_updates=0, native_imported=False,
                  limits=['Teacher agreement is not teacher correctness or closed-loop competence.',
                          'Original vector policy draws differ from scalar re-observation features.',
                          'Reused development learner/cohort; test lanes are not learner confirmation.'])
    for name, data in arrays.items():
        teacher, actual = attach_actions(corpus, data, phase_rows)
        result['fits'][name], predictions = fit_actions(data, teacher, actual)
        test = data['split'] == 'test'
        with (output / (name + '-predictions.npz')).open('xb') as stream:
            np.savez_compressed(stream, teacher=teacher[test], lane=data['lane'][test],
                                index=data['index'][test], **predictions)
    require_cpu_only()
    result.update(status='complete', seconds=time.monotonic() - started,
                  prediction_hashes={name: sha(output / (name + '-predictions.npz')) for name in arrays})
    write(output / 'result.json', result)
    print(json.dumps(dict(status='complete', seconds=result['seconds'], actor_updates=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features_root', 'phase_path', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.features_root, args.phase_path, args.output)
