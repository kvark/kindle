"""Bounded native world learning with fitting-only paired action branches."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import breakout_world_corpus as corpus
import breakout_world_coverage as coverage
from breakout_world_model import sha, write


def load_fitting(dataset, supplement, paired):
    original, original_manifest = corpus.load_dataset(dataset)
    extra, extra_manifest = corpus.load_dataset(supplement)
    for lane, stream in enumerate(extra):
        with np.load(supplement/f'lane{lane}.npz', allow_pickle=False) as data:
            stream['source_wait'] = data['source_wait']
    streams = original+extra
    manifest = json.loads((paired/'manifest.json').read_text())
    if manifest['split'] != 'fitting paired branches' or len(manifest['files']) != 448:
        raise ValueError('qualified fitting-only paired dataset required')
    grouped = {}
    for row in manifest['files']:
        path = Path(row['path'])
        if sha(path) != row['sha256']: raise ValueError(f'paired data changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            stream = {key: data[key] for key in corpus.FIELDS}
        size = len(stream['action'])
        assert size == 32+row['future_arrivals'] and 32 < size <= 64
        assert not stream['terminal'][:-1].any() and (stream['action'][1:] >= 0).all()
        assert size == 64 or stream['terminal'][-1]
        key = (row['parent'], row['draw'])
        plans = grouped.setdefault(key, {})
        assert row['plan'] not in plans
        plans[row['plan']] = len(streams)
        streams.append(stream)
    groups = []
    for plans in grouped.values():
        names = ['noop', 'fire', 'leftfire', 'rightfire'] if 'fire' in plans else ['noop', 'left', 'right', 'reverse']
        assert set(plans) == set(names)
        lanes = [plans[name] for name in names]
        first = streams[lanes[0]]
        for lane in lanes[1:]:
            for key in corpus.FIELDS:
                np.testing.assert_array_equal(first[key][:32], streams[lane][key][:32])
        groups.append(lanes)
    assert len(groups) == 112 and len(streams) == 464
    return streams, np.asarray(groups, np.int32), original_manifest, extra_manifest


def paired_plan(streams, groups, config, steps, prediction_length=16):
    if config['batch_size'] != 8 or config['batch_length'] != 32 or prediction_length != 16:
        raise ValueError('paired training requires B8/T32/H16')
    groups = np.asarray(groups)
    if (groups.ndim != 2 or groups.shape[1] != 4 or not len(groups)
            or sorted(groups.ravel().tolist()) != list(range(16, len(streams)))):
        raise ValueError('each paired stream must belong to exactly one four-plan group')
    waits, endings = coverage.sampling_pools(streams[:16])
    paired_endings = []
    for lane, stream in enumerate(streams[16:], 16):
        if not 32 < len(stream['action']) <= 64 or (stream['action'][1:] < 0).any():
            raise ValueError('paired streams require a real prefix and non-reset future')
        terminal = np.flatnonzero(stream['terminal'])
        if len(terminal):
            if terminal.tolist() != [len(stream['action'])-1]:
                raise ValueError('paired terminal must be the final arrival')
            paired_endings.append((lane, int(terminal[0])-31))
    if paired_endings: endings = np.concatenate([endings, np.asarray(paired_endings, np.int32)])
    rng = np.random.default_rng(config['seed'] ^ 0x706169726564)
    endings = endings[rng.permutation(len(endings))]
    locations = []
    for step in range(steps):
        lanes = groups[rng.integers(len(groups))]
        limit = min(len(streams[lane]['action'])-32 for lane in lanes)
        start = min(16, limit) if step % 2 == 0 else int(rng.integers(limit+1))
        paired = [(int(lane), start) for lane in lanes]
        original = corpus.sample_locations(streams[:8], 32, 2, rng)
        wait = tuple(waits[rng.integers(len(waits))])
        locations.append(paired+original+[wait, tuple(endings[step % len(endings)])])
    locations = np.asarray(locations, np.int32)
    return locations, corpus.plan_counts(streams, config, locations, prediction_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'restore-check'])
    for name in ['config', 'development', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['dataset', 'supplement', 'paired', 'initial-reference', 'sampling-plan', 'reference', 'checkpoint']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--eval-every', type=int, default=1024)
    parser.add_argument('--dense-fitting', type=int, default=256)
    parser.add_argument('--dense-development', type=int, default=128)
    args = parser.parse_args()
    args.prediction_length, args.pixel_loss = 16, 'motion_cross_entropy'
    if min(args.steps, args.eval_every, args.dense_fitting, args.dense_development) <= 0:
        parser.error('positive learning/evaluation budgets required')
    if args.mode == 'train' and any(getattr(args, name) is None for name in
            ['dataset', 'supplement', 'paired', 'initial_reference', 'sampling_plan']):
        parser.error('training requires fitting data, matched initialization and fixed sampling plan')
    if args.mode == 'restore-check' and (args.checkpoint is None or args.reference is None):
        parser.error('restore requires checkpoint and evaluation reference')
    config = json.loads(args.config.read_text())
    development, dev_manifest = corpus.load_dataset(args.development)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'declaration.json', dict(config=config, pixel_loss=args.pixel_loss,
        prediction_length=16, steps=args.steps, eval_every=args.eval_every, mode=args.mode,
        source_sha256=sha(__file__), started_unix=time.time()))
    if args.mode == 'restore-check':
        corpus.restore_check(args, config, development)
        return
    fitting, groups, original_manifest, extra_manifest = load_fitting(args.dataset, args.supplement, args.paired)
    assert not {row['seed'] for row in original_manifest['files']+extra_manifest['files']}.intersection(
        row['seed'] for row in dev_manifest['files'])
    plan, counts = paired_plan(fitting, groups, config, args.steps)
    with np.load(args.sampling_plan, allow_pickle=False) as declared:
        np.testing.assert_array_equal(plan, declared['locations'])
    selections = dict(fitting=corpus.choose_evaluation(fitting[:16], 112003),
                      development=corpus.choose_evaluation(development, 112019))
    expected = json.loads((args.initial_reference/'selections.json').read_text())
    assert json.loads(json.dumps(selections)) == expected
    corpus.train(args, config, fitting, development, sampler=lambda *_: (plan, counts),
                 initial_reference=args.initial_reference, evaluation_fitting=fitting[:16])
    import kindle
    from safetensors.numpy import load_file
    learner = kindle.WorldModelLearner.restore(str(args.output/'final'))
    corpus.require_device(learner)
    output = args.output/'paired-fitting'; output.mkdir()
    locations = [(lane, 16) for lane in range(16, len(fitting))]
    corpus.evaluate_group(learner, fitting, locations, output, 'intervention', draws=corpus.DRAWS[:1])
    learner.save(str(output/'frozen'))
    before = load_file(args.output/'final/world.safetensors')
    after = load_file(output/'frozen/world.safetensors')
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
    write(output/'result.json', dict(status='complete', frozen_tensors=len(before), updates=learner.updates,
        origins=len(locations), source_streams='paired fitting only; all 448 intervention origins',
        world_model_proven=False))


if __name__ == '__main__': main()
