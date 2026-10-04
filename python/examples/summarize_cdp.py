"""Audit completed fixed-budget CDP/RGB runs without constructing a GPU context."""

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from safetensors import safe_open

from atari import sha256_file
from kindle._screening import mean_ci, summarize_curves
from kindle._vector_audit import audit
from summarize_representation_learning import SEEDS, plot_svg, read_run


def recipe(header, method, *, budget=200000):
    expected = dict(environment='ALE/Seaquest-v5', num_envs=8, steps=budget, full_action_space=True,
                    sticky_actions=.25, action_repeat=4, noop_max=0, max_episode_frames=100000,
                    mode='train', observation_size='native', starting_environment_step=0,
                    starting_learner_step=0, restored_checkpoint=None)
    if any(header.get(k) != v for k, v in expected.items()) or header.get('exploration'):
        raise ValueError('not the declared fresh Seaquest run')
    c = header['config']
    common = dict(model_size='size1_m', observation_kind='rgb64', video_encoder=None, action_count=18,
                  batch_size=8, batch_length=16, world_backprop_length=16, world_microbatch_size=8,
                  replay_context=1, replay_capacity=100000, train_ratio=32., imagination_length=15,
                  learning_rate=4e-5, learning_rate_warmup=1000, agc=.3, actor_unimix=0,
                  replay_value_gradient=True, actor_critic_gradient=False, intrinsic_reward_scale=0,
                  extrinsic_reward_scale=1, visitation_bonus=False)
    if any(c.get(k) != v for k, v in common.items()):
        raise ValueError('changed common learner recipe')
    if method not in ('rgb', 'cdp'):
        raise ValueError('unknown arm')
    cdp = method == 'cdp'
    if (c['encoder_learning_rate'] != (6e-6 if cdp else None)
            or c['dynamics_learning_rate'] != (4e-4 if cdp else None)
            or c['loss_scales']['future_prediction'] != (500 if cdp else 0)
            or c['loss_scales']['reconstruction'] != (0 if cdp else 1)):
        raise ValueError('CDP/RGB label disagrees with objective or rates')
    if header['environment_seeds'] != [(header['seed'] + i * 1000003) % 2**32 for i in range(8)]:
        raise ValueError('changed environment seeds')
    shared = {k: v for k, v in c.items() if k not in ('seed', 'encoder_learning_rate', 'dynamics_learning_rate', 'loss_scales')}
    shared['loss_scales'] = {k: v for k, v in c['loss_scales'].items() if k not in ('reconstruction', 'future_prediction')}
    shared['native_sha256'] = header['native_extension_sha256']
    shared['wrapper_sha256'] = header['wrapper_sha256']
    shared['runner_sha256'] = header['runner_sha256']
    shared['preprocessing'] = header['learned_rgb_preprocessing']
    shared['provenance'] = {k: v for k, v in header['model_provenance'].items() if k != 'future_head_revision'}
    return shared


def checkpoint_audit(path, header, updates):
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata['config'] != header['config'] or metadata['learner_step'] != updates:
        raise ValueError('final checkpoint configuration/counter differs')
    result = {}
    for name in ('world', 'behavior', 'slow_value'):
        file = path / f'{name}.safetensors'
        if sha256_file(file) != metadata['tensor_sha256'][name]:
            raise ValueError('checkpoint tensor identity changed')
        with safe_open(file, framework='np') as tensors:
            if not tensors.keys() or any(not np.isfinite(tensors.get_tensor(key)).all() for key in tensors.keys()):
                raise ValueError('nonfinite/empty checkpoint')
            result[name] = len(tensors.keys())
    return dict(sha256=sha256_file(path / 'metadata.json'), finite_tensor_counts=result)


def summarize(root):
    groups, common = dict(rgb=[], cdp=[]), None
    for method in groups:
        for seed in SEEDS:
            name = f'seaquest-{method}-{seed}'
            guard_file = root / 'learning-queue' / name / 'result.json'
            if not guard_file.exists():
                continue
            guard = json.loads(guard_file.read_text())
            if not guard.get('host_guard_passed'):
                raise ValueError(f'failed guard: {name}')
            path = root / f'{name}.jsonl'
            run = read_run(path, diagnostics=True)
            shared = recipe(run['header'], method)
            if common is not None and shared != common:
                raise ValueError('shared settings/implementation changed across runs')
            common = shared
            if run['seed'] != seed or run['final']['learner_updates'] != 49939 or run['final']['training_debt'] != 0:
                raise ValueError('unexpected seed/update count/debt')
            run['accounting_audit'] = audit(path)
            run['checkpoint_audit'] = checkpoint_audit(root / f'{name}-checkpoint', run['header'], 49939)
            windows = [row for row in run['curve'] if 'learner_mean' in row]
            count = sum(row['reported_updates'] for row in windows)
            if count != 49939:
                raise ValueError('missing diagnostic update reports')
            run['learner_mean'] = {section: {key: sum(row['reported_updates'] * row['learner_mean'][section][key]
                for row in windows) / count for key in windows[0]['learner_mean'][section]}
                for section in ('world', 'behavior', 'timing')}
            groups[method].append(run)
    aggregates = {}
    for method, runs in groups.items():
        if len(runs) == 3:
            aggregates[method] = dict(curves=summarize_curves(runs),
                final_score=mean_ci([r['curve'][-1]['score'] for r in runs]),
                run_seconds=mean_ci([r['final']['elapsed_seconds'] for r in runs]),
                mean_construction_seconds=fmean(r['header']['agent_construction_seconds'] for r in runs),
                mean_learner_timing={k: fmean(r['learner_mean']['timing'][k] for r in runs)
                                     for k in runs[0]['learner_mean']['timing']})
    paired = None
    if len(aggregates) == 2:
        paired = dict(score_difference=mean_ci([a['curve'][-1]['score'] - b['curve'][-1]['score']
                      for a, b in zip(groups['cdp'], groups['rgb'])]),
                      cdp_over_rgb_seconds=mean_ci([a['final']['elapsed_seconds'] / b['final']['elapsed_seconds']
                      for a, b in zip(groups['cdp'], groups['rgb'])]))
    return dict(protocol='kindle-cdp-learning-v1', status='complete' if paired else 'partial',
                root=str(root.resolve()), completed_runs=sum(map(len, groups.values())),
                shared_recipe=common, runs=groups, aggregate=aggregates, paired=paired,
                limitations=['online last-50 completed episode mean, not frozen competence',
                             'all completed episodes and unfinished tails retained',
                             'three paired learner seeds; bootstrap uncertainty is coarse',
                             'CDP changes the complete loss/rate package; not one isolated component',
                             'construction reported separately; no GPU utilization or physical/peak VRAM claim'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot = dict(comparison='cdp', methods=('rgb', 'cdp'), games=('Seaquest',), num_envs=8, action_budget=200000,
                results=[dict(method=method, game='Seaquest', runs=runs,
                              aggregate=result['aggregate'].get(method, {}).get('curves'))
                         for method, runs in result['runs'].items()])
    args.output.with_suffix('.svg').write_text(plot_svg(plot))
    print(json.dumps(dict(status=result['status'], completed_runs=result['completed_runs'], paired=result['paired'])))


if __name__ == '__main__':
    main()
