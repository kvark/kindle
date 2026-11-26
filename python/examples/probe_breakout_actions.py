"""CPU-only action readouts from saved frozen Breakout features."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from kindle._representation_probe import ridge_probe
from probe_breakout_representation import read_features, require_cpu_only, sha, write


ACTIONS = np.array([0, 1, 3, 4], dtype=np.int64)
PROTOCOL = 'kindle-breakout-action-readout-v1'


def attach_actions(corpus, data, phase_rows):
    by_key = {}
    for row in phase_rows:
        key = (row['stream'], row['episode'], row['step'])
        if key in by_key:
            raise ValueError('duplicate phase identity')
        by_key[key] = row
    if len(by_key) != sum(map(len, corpus['streams'])):
        raise ValueError('incomplete phase source')
    labels, actual = {}, {}
    for lane, rows in enumerate(corpus['streams']):
        for index, row in enumerate(rows):
            phase = by_key[(lane, row['episode'], row['step'])]
            if (phase['action'] != row['action'] or phase['phase'] != row['phase']
                    or phase['input_pixel_sha256'] != row['input_pixel_sha256']
                    or phase['label'] not in ACTIONS):
                raise ValueError('phase label/input alignment differs')
            labels[lane, index], actual[lane, index] = phase['label'], phase['action']
    keys = list(zip(data['lane'].tolist(), data['index'].tolist(), strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate feature identity')
    return np.array([labels[key] for key in keys]), np.array([actual[key] for key in keys])


def classification_metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if (target.ndim != 1 or target.shape != prediction.shape
            or not np.isin(target, ACTIONS).all()
            or not np.isin(prediction, np.arange(18)).all()):
        raise ValueError('invalid teacher/predicted action vectors')
    confusion = np.zeros((4, 18), dtype=np.int64)
    for index, action in enumerate(ACTIONS):
        selected = target == action
        np.add.at(confusion[index], prediction[selected].astype(np.int64), 1)
    support = confusion.sum(axis=1)
    correct = confusion[np.arange(4), ACTIONS]
    recalls = {str(action): dict(count=int(n), correct=int(k), recall=float(k/n) if n else None)
               for action, n, k in zip(ACTIONS, support, correct, strict=True)}
    return dict(count=len(target), correct=int(correct.sum()),
        agreement=float(correct.sum()/len(target)) if len(target) else None,
        macro_recall=float(np.mean(correct[support>0]/support[support>0])) if support.any() else None,
        classes_with_support=int((support>0).sum()), per_class=recalls,
        teacher_actions=ACTIONS.tolist(), predicted_actions=list(range(18)), confusion=confusion.tolist())


def evaluate(target, prediction, lanes, phases):
    masks = dict(all=np.ones(len(target), dtype=bool), near_descending=phases == 'near_descending')
    masks.update({f'phase:{phase}': phases == phase for phase in np.unique(phases)})
    masks.update({f'lane{lane}': lanes == lane for lane in np.unique(lanes)})
    return {name:classification_metrics(target[mask], prediction[mask]) for name,mask in masks.items()}


def fit_actions(data, teacher, actual):
    if len(teacher) != len(data['lane']) or len(actual) != len(teacher) or not np.isin(teacher, ACTIONS).all():
        raise ValueError('invalid aligned action labels')
    masks = {name:data['split'] == name for name in ('train','validation','test')}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('missing action readout split')
    train, validation, test = [masks[name] for name in ('train','validation','test')]
    if (set(data['lane'][train]) - {0,1,2,3} or set(data['lane'][validation]) - {4,5}
            or set(data['lane'][test]) - {6,7}):
        raise ValueError('lane roles differ')
    frequencies = np.array([(teacher[train] == action).sum() for action in ACTIONS])
    majority = int(ACTIONS[np.argmax(frequencies)])
    labels = (teacher[:,None] == ACTIONS).astype(np.float64)
    result = dict(counts={key:int(mask.sum()) for key,mask in masks.items()},
        training_class_counts=dict(zip(map(str,ACTIONS),map(int,frequencies))), majority_action=majority,
        baselines={}, variants={})
    predictions = dict(majority=np.full(test.sum(),majority), recorded=actual[test])
    for name, values in predictions.items():
        result['baselines'][name] = evaluate(teacher[test], values, data['lane'][test], data['phase'][test])
    features = {key:data[key] for key in ('cnn','posterior')}
    features['combined'] = np.concatenate([features['cnn'],features['posterior']],axis=1)
    for name, values in features.items():
        scores, selection = ridge_probe(values[train],labels[train],values[validation],labels[validation],values[test])
        predictions[name] = ACTIONS[np.argmax(scores,axis=1)]
        predictions[name+'_scores'] = scores
        result['variants'][name] = dict(width=values.shape[1],selection=selection,
            metrics=evaluate(teacher[test],predictions[name],data['lane'][test],data['phase'][test]))
    return result, predictions


def run(features_root, phase_path, output):
    require_cpu_only()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    os.nice(10)
    started = time.monotonic()
    corpus_path = features_root/'corpus.json'
    corpus = json.loads(corpus_path.read_text())
    if corpus['phase_rows_sha256'] != sha(phase_path):
        raise ValueError('phase source identity differs')
    with phase_path.open() as source:
        phase_rows = [json.loads(line) for line in source]
    arrays = {}
    for name in ('final','initial'):
        capture = json.loads((features_root/name/'result.json').read_text())
        if capture['corpus_sha256'] != sha(corpus_path):
            raise ValueError('capture corpus differs')
        arrays[name] = read_features(corpus,features_root/name)
    for field in ('lane','index','split','phase'):
        if not np.array_equal(arrays['final'][field],arrays['initial'][field]):
            raise ValueError('initial/final row identity differs')
    output.mkdir(exist_ok=False)
    result = dict(protocol=PROTOCOL,status='running',source_sha256=sha(__file__),corpus_sha256=sha(corpus_path),
        phase_rows_sha256=sha(phase_path),fits={},actor_updates=0,native_imported=False,
        limits=['Offline teacher agreement, not correctness or closed-loop task performance.',
                'Recorded native baseline uses original vector draws; readout features use scalar re-observation.',
                'Same selected development learner/cohort; separate readout test lanes are not learner confirmation.'])
    for name,data in arrays.items():
        teacher,actual = attach_actions(corpus,data,phase_rows)
        result['fits'][name],predictions = fit_actions(data,teacher,actual)
        test = data['split'] == 'test'
        with (output/(name+'-predictions.npz')).open('xb') as stream:
            np.savez_compressed(stream,teacher=teacher[test],lane=data['lane'][test],index=data['index'][test],**predictions)
    require_cpu_only()
    result.update(status='complete',seconds=time.monotonic()-started,
        prediction_hashes={name:sha(output/(name+'-predictions.npz')) for name in arrays})
    write(output/'result.json',result)
    print(json.dumps(dict(status='complete',seconds=result['seconds'],actor_updates=0)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features_root','phase_path','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    run(args.features_root,args.phase_path,args.output)
