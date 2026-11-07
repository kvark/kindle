"""Warm-start an H16 forecast phase with an immutable observation filter."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import breakout_world_corpus as corpus
import breakout_world_coverage as coverage
from breakout_world_model import sha, write


def load_fitting(dataset, supplement):
    original, original_manifest = corpus.load_dataset(dataset)
    extra, extra_manifest = corpus.load_dataset(supplement)
    for lane, stream in enumerate(extra):
        with np.load(supplement/f'lane{lane}.npz', allow_pickle=False) as data:
            stream['source_wait'] = data['source_wait']
    return original+extra, original_manifest, extra_manifest


def config_f32(value):
    if isinstance(value, dict): return {key: config_f32(item) for key, item in value.items()}
    if isinstance(value, list): return [config_f32(item) for item in value]
    if isinstance(value, float):
        assert np.isfinite(value) and np.isfinite(np.float32(value))
        return float(np.float32(value))
    return value


def assert_phase(learner, metadata, pixel_loss=corpus.FIXED_TEACHER_OBJECTIVE):
    # Getters serialize F32 directly; checkpoint JSON first stores it as F64.
    assert config_f32(learner.config) == config_f32(metadata['config'])
    assert learner.phase_start_updates == metadata['updates']
    assert learner.updates >= learner.phase_start_updates
    assert learner.training_scope == 'fixed_teacher'
    assert learner.prediction_length == 16 and learner.pixel_loss == pixel_loss


def main(*, pixel_loss=corpus.FIXED_TEACHER_OBJECTIVE):
    factory = {corpus.FIXED_TEACHER_OBJECTIVE: 'start_fixed_teacher_phase',
               corpus.FIXED_IMAGE_OBJECTIVE: 'start_fixed_teacher_image_phase'}[pixel_loss]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'restore-check'])
    for name in ['output', 'development', 'source-checkpoint']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['dataset', 'supplement', 'sampling-plan', 'checkpoint', 'reference']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--eval-every', type=int, default=1024)
    parser.add_argument('--dense-fitting', type=int, default=256)
    parser.add_argument('--dense-development', type=int, default=128)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.dense_fitting, args.dense_development) <= 0:
        parser.error('positive phase/evaluation budgets required')
    if args.mode == 'train' and any(getattr(args, name) is None for name in ['dataset', 'supplement', 'sampling_plan']):
        parser.error('phase requires both original fitting sources and a pinned plan')
    if args.mode == 'restore-check' and (args.checkpoint is None or args.reference is None):
        parser.error('restore requires the new checkpoint and evaluation reference')
    source = json.loads((args.source_checkpoint/'metadata.json').read_text())
    if source['updates'] != 8192 or source['prediction_length'] != 0 or source['pixel_loss'] != 'motion_cross_entropy':
        parser.error('this experiment inherits the retained experienced H0 motion world')
    config = source['config']
    args.prediction_length, args.pixel_loss = 16, pixel_loss
    args.dense_initial = True
    development, dev_manifest = corpus.load_dataset(args.development)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'declaration.json', dict(mode=args.mode, config=config, steps=args.steps,
        eval_every=args.eval_every, prediction_length=16, pixel_loss=args.pixel_loss,
        source_checkpoint=str(args.source_checkpoint), source_sha256=sha(__file__),
        inherited_updates=source['updates'], training_scope='fixed_teacher', started_unix=time.time()))
    import kindle
    if args.mode == 'restore-check':
        learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
        assert_phase(learner, source, pixel_loss)
        corpus.restore_check(args, config, development, learner=learner)
        return
    fitting, original, extra = load_fitting(args.dataset, args.supplement)
    assert not {row['seed'] for row in original['files']+extra['files']}.intersection(row['seed'] for row in dev_manifest['files'])
    selections = dict(fitting=corpus.choose_evaluation(fitting, 112003),
                      development=corpus.choose_evaluation(development, 112019))
    expected = json.loads((args.source_checkpoint.parent/'selections.json').read_text())
    assert json.loads(json.dumps(selections)) == expected
    plan, counts = coverage.coverage_plan(fitting, config, args.steps, 16)
    with np.load(args.sampling_plan, allow_pickle=False) as declared:
        np.testing.assert_array_equal(plan, declared['locations'])
    learner = getattr(kindle.WorldModelLearner, factory)(str(args.source_checkpoint))
    assert_phase(learner, source, pixel_loss)
    assert learner.updates == source['updates']
    corpus.train(args, config, fitting, development, sampler=lambda *_: (plan, counts), learner=learner,
                 initial_checkpoint=args.source_checkpoint)


if __name__ == '__main__': main()
