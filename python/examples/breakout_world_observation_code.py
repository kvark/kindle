"""Bounded world-only learning with an observation-only RGB categorical code."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import breakout_world_corpus as corpus
import breakout_world_coverage as coverage
from breakout_world_frozen_visual import config_f32, load_fitting
from breakout_world_model import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'restore-check'])
    for name in ['config', 'development', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['dataset', 'supplement', 'sampling-plan', 'initial-reference', 'checkpoint', 'reference']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--eval-every', type=int, default=1024)
    parser.add_argument('--dense-fitting', type=int, default=256)
    parser.add_argument('--dense-development', type=int, default=128)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.dense_fitting, args.dense_development) <= 0:
        parser.error('positive training and evaluation budgets required')
    if args.mode == 'train' and any(getattr(args, key) is None for key in ['dataset', 'supplement', 'sampling_plan', 'initial_reference']):
        parser.error('training requires original data, plan and qualified actual initial reference')
    if args.mode == 'restore-check' and (args.checkpoint is None or args.reference is None):
        parser.error('restore requires checkpoint and final evaluation reference')
    config = json.loads(args.config.read_text())
    if not config.get('rgb_latent_only') or config['batch_size'] != 8 or config['batch_length'] != 32:
        parser.error('this protocol requires observation-only RGB at B8/T32')
    args.prediction_length, args.pixel_loss = 16, corpus.LATENT_OBJECTIVE
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'declaration.json', dict(mode=args.mode, config=config, steps=args.steps,
        eval_every=args.eval_every, prediction_length=16, pixel_loss=args.pixel_loss,
        source_sha256=sha(__file__), started_unix=time.time(), dense_initial=True))
    development, dm = corpus.load_dataset(args.development)
    import kindle
    if args.mode == 'restore-check':
        learner = kindle.WorldModelLearner.restore(str(args.checkpoint))
        assert config_f32(learner.config) == config_f32(config)
        assert learner.config['rgb_latent_only'] and learner.prediction_length == 16
        corpus.restore_check(args, config, development, learner=learner)
        return
    fitting, old, new = load_fitting(args.dataset, args.supplement)
    assert not {row['seed'] for row in old['files']+new['files']}.intersection(row['seed'] for row in dm['files'])
    plan, counts = coverage.coverage_plan(fitting, config, args.steps, 16)
    with np.load(args.sampling_plan, allow_pickle=False) as declared:
        np.testing.assert_array_equal(plan, declared['locations'])
    learner = kindle.WorldModelLearner(config, prediction_length=16, pixel_loss=args.pixel_loss)
    corpus.require_device(learner)
    assert learner.updates == 0 and config_f32(learner.config) == config_f32(config)
    initial = args.output/'dense-initial'
    initial.mkdir()
    for label, streams, count, seed in [('fitting', fitting, args.dense_fitting, 182003),
                                       ('development', development, args.dense_development, 182009)]:
        locations = corpus.sample_locations(streams, corpus.HISTORY+1, count, np.random.default_rng(seed))
        corpus.evaluate_group(learner, streams, locations, initial, label, draws=corpus.DRAWS[:1])
    # train saves/checks all initial tensors after the dense queries, against qualification.
    corpus.train(args, config, fitting, development, sampler=lambda *_: (plan, counts),
                 learner=learner, initial_reference=args.initial_reference)


if __name__ == '__main__': main()
