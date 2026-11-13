"""CPU nonlinear state readouts on existing frozen Breakout captures."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from probe_breakout_kernel_actions import ALPHAS, kernel_probe
from probe_breakout_representation import (
    SPLITS, TARGETS, evaluate, read_features, require_cpu_only, sha, write,
)


PROTOCOL = 'kindle-breakout-kernel-state-v1'


def fit_state(data):
    masks = {name: data['split'] == name for name in SPLITS}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('missing state readout split')
    train, validation, test = masks.values()
    if (set(data['lane'][train]) - {0, 1, 2, 3}
            or set(data['lane'][validation]) - {4, 5}
            or set(data['lane'][test]) - {6, 7}):
        raise ValueError('lane roles differ')
    targets = data['targets']
    if targets.shape != (len(data['lane']), len(TARGETS)) or np.isinf(targets).any():
        raise ValueError('invalid state targets')
    result = dict(counts={name: int(mask.sum()) for name, mask in masks.items()}, variants={})
    features = {name: data[name] for name in ('cnn', 'posterior')}
    features['combined'] = np.concatenate([features['cnn'], features['posterior']], axis=1)
    predictions = {}
    for name, values in features.items():
        prediction = np.empty((test.sum(), len(TARGETS)))
        selections = []
        for index, target in enumerate(TARGETS):
            valid = np.isfinite(targets[:, index])
            tr, va = train & valid, validation & valid
            if min(tr.sum(), va.sum()) < 2:
                raise ValueError('insufficient training/validation target support')
            fitted, selection = kernel_probe(values[tr], targets[tr, index:index+1],
                values[va], targets[va, index:index+1], values[test])
            prediction[:, index] = fitted[:, 0]
            selections.append(dict(target=target, training_count=int(tr.sum()),
                validation_count=int(va.sum()), **selection))
        predictions[name] = prediction
        result['variants'][name] = dict(width=values.shape[1], selection=selections,
            metrics=evaluate(prediction, targets[test], data['lane'][test], data['phase'][test]))
    return result, predictions


def run(features_root, output):
    require_cpu_only()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    started = time.monotonic()
    corpus_path = features_root / 'corpus.json'
    corpus = json.loads(corpus_path.read_text())
    arrays = {}
    for name in ('final', 'initial'):
        capture = json.loads((features_root / name / 'result.json').read_text())
        if capture['corpus_sha256'] != sha(corpus_path):
            raise ValueError('capture corpus differs')
        arrays[name] = read_features(corpus, features_root / name)
    for field in ('lane', 'index', 'split', 'phase', 'targets'):
        left, right = arrays['final'][field], arrays['initial'][field]
        equal = np.array_equal(left, right, equal_nan=True) if field == 'targets' else np.array_equal(left, right)
        if not equal:
            raise ValueError('initial/final row identity differs')
    output.mkdir(exist_ok=False)
    result = dict(protocol=PROTOCOL, status='running', source_sha256=sha(__file__),
        corpus_sha256=sha(corpus_path), alphas=ALPHAS, targets=TARGETS, fits={},
        actor_updates=0, native_imported=False,
        limits=['Teacher detections are reference labels, not independent physical truth.',
                'Reused weak-policy cohort and readout test lanes, not learner confirmation.',
                'Scalar captured draws differ from the original vector policy.',
                'Decoder accessibility is not causal policy use or all available information.'])
    for name, data in arrays.items():
        result['fits'][name], predictions = fit_state(data)
        test = data['split'] == 'test'
        with (output / (name + '-predictions.npz')).open('xb') as stream:
            np.savez_compressed(stream, lane=data['lane'][test], index=data['index'][test],
                targets=data['targets'][test], **predictions)
    require_cpu_only()
    result.update(status='complete', seconds=time.monotonic() - started,
        prediction_hashes={name: sha(output / (name + '-predictions.npz')) for name in arrays})
    write(output / 'result.json', result)
    print(json.dumps(dict(status='complete', seconds=result['seconds'], actor_updates=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features_root', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.features_root, args.output)
