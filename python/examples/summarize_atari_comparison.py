"""CPU-only audits for the declared five-game CDP/RGB/frozen-Tiny comparison.

Audit each training run before frozen evaluation, then the complete pair before
another GPU invocation. This tool never launches, resumes or retries GPU work.
"""

import argparse
import json
from pathlib import Path
from statistics import fmean

from atari import checkpoint_identity, sha256_file
from audit_atari import read_run as read_frozen, require, verify_final_pair
from gpu_host_guard import audit as audit_guard
from kindle._screening import mean_ci, summarize_curves
from kindle._vector_audit import audit, episode_summary
from probe_fixed_latents import assert_frozen_tensors
from summarize_cdp import checkpoint_audit, recipe
from summarize_representation_learning import SEEDS, plot_svg, read_run


GAMES = ('Freeway', 'Boxing', 'Pong', 'Breakout', 'Qbert')
METHODS = ('cdp', 'rgb', 'pretrained_tiny')
BUDGET, UPDATES = 200000, 49939
NATIVE_SHA256 = '32353ffb5d4516aa9281e94004f7b7ca2f126c9d29bfd62e33c36f78c44f502a'


def run_name(game, method, seed):
    require(game in GAMES and method in METHODS and seed in SEEDS, 'undeclared game/method/seed')
    return f'{game.lower()}-{method}-{seed}'


def guard(root, name):
    result = audit_guard(root / f'{name}-queue' / name)
    require(result['host_guard_passed'], f'failed guard: {name}')
    return result


def training(root, game, method, seed):
    name = run_name(game, method, seed)
    checked_guard = guard(root, name)
    path = root / f'{name}.jsonl'
    run = read_run(path, diagnostics=True)
    header = run['header']
    shared = recipe(header, method, environment=f'ALE/{game}-v5')
    require(run['seed'] == header['config']['seed'] == seed, 'wrong learner seed')
    require(header['native_extension_sha256'] == NATIVE_SHA256, 'changed native implementation')
    for key, file in (('runner_sha256', 'atari_vector.py'), ('wrapper_sha256', 'atari.py')):
        require(header[key] == sha256_file(Path(__file__).with_name(file)), 'changed acting source')
    require(run['final']['learner_updates'] == UPDATES and run['final']['training_debt'] == 0,
            'wrong update count or unsettled learner debt')
    run['accounting'] = audit(path)
    checkpoint = root / f'{name}-checkpoint'
    run['checkpoint'] = checkpoint_audit(checkpoint, header, UPDATES)
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    require(metadata['environment_step'] == BUDGET and metadata['collection_streams'] == 8,
            'checkpoint is not the final training state')
    require(metadata['perception'] == header['model_provenance']['perception'], 'changed saved frontend')
    windows = [row for row in run['curve'] if 'learner_mean' in row]
    count = sum(row['reported_updates'] for row in windows)
    require(count == UPDATES, 'missing learner diagnostics')
    run['learner_mean'] = {section: {key: sum(row['reported_updates'] * row['learner_mean'][section][key]
        for row in windows) / count for key in windows[0]['learner_mean'][section]}
        for section in ('world', 'behavior', 'timing')}
    for row in run['curve']:
        row.pop('learner_mean', None)
    run['guard'] = checked_guard
    return dict(name=name, game=game, method=method, shared_recipe=shared, training=run)


def cohort(episodes, streams=8, target=3, *, natural_only=False):
    """Retain legacy completed cohorts; new natural quotas exclude, but retain, cutoffs."""
    counts = [0] * streams
    selected, excess = [], []
    for episode in episodes:
        stream = episode['stream']
        require(type(stream) is int and 0 <= stream < streams, 'invalid cohort stream')
        eligible = not natural_only or (episode['terminated'] and not episode['truncated'])
        (selected if eligible and counts[stream] < target else excess).append(episode)
        counts[stream] += eligible
    complete = min(counts) >= target
    natural = [e for e in selected if e['terminated'] and not e['truncated']]
    truncated = [e for e in selected if e['truncated']]
    return dict(complete=complete, per_stream_counts=counts, selected=selected, excess=excess,
                score=fmean(e['episode_return'] for e in selected) if complete else None,
                natural=episode_summary(natural), truncated=episode_summary(truncated))


def pair(root, game, method, seed, *, require_replay=False):
    result = training(root, game, method, seed)
    name = result['name']
    evaluation_name = f'{name}-frozen'
    checked_guard = guard(root, evaluation_name)
    train = read_frozen(root / f'{name}.jsonl')
    frozen = read_frozen(root / f'{evaluation_name}.jsonl', allow_capped_evaluation=True)
    verify_final_pair(train, frozen)
    h = frozen['start']
    expected = dict(num_envs=8, steps=BUDGET, seed=1000000000 + seed,
                    evaluation_episodes_per_stream=3, frozen_checkpoint_export=True,
                    observation_size='native', starting_environment_step=BUDGET, starting_learner_step=UPDATES)
    require(all(h.get(k) == v for k, v in expected.items()), 'changed frozen evaluation declaration')
    require(h['environment_seeds'] == [(h['seed'] + i * 1000003) % 2**32 for i in range(8)],
            'changed frozen environment seeds')
    source, after = root / f'{name}-checkpoint', root / f'{evaluation_name}-checkpoint'
    require(train['checkpoint']['identity'] == h['restored_checkpoint'] == checkpoint_identity(source),
            'frozen evaluation did not restore the final checkpoint')
    require(frozen['checkpoint'] is not None
            and frozen['checkpoint']['run_step'] == frozen['end']['run_step']
            and frozen['checkpoint']['identity'] == checkpoint_identity(after), 'missing final frozen export')
    frozen['checkpoint_audit'] = checkpoint_audit(after, h, UPDATES)
    frozen['unchanged_tensor_counts'] = assert_frozen_tensors(source, after)
    frozen['cohort'] = cohort(frozen['episodes'],
                              natural_only=frozen['accounting']['evaluation_episode_kind'] == 'natural')
    require(frozen['cohort']['complete'] == frozen['accounting']['budget_complete'], 'cohort status differs')
    frozen['guard'] = checked_guard
    if require_replay:
        import replay_atari
        from audit_atari_campaign import check_match_replay
        from audit_atari_tasks import check_replay
        replay_atari.gym.register_envs(replay_atari.ale_py)
        path = root / f'{evaluation_name}-replay.json'
        replay = json.loads(path.read_text())
        (check_match_replay if game in ('Boxing', 'Pong') else check_replay)(frozen, replay)
        video = replay['video']
        require(video is not None and video['stream'] == 0 and video['fps'] == 60
                and video['frames'] == frozen['end']['executed_action_frames'][0]
                and sha256_file(video['path']) == video['sha256'], 'missing or changed whole-stream video')
        frozen['replay'] = dict(path=str(path), sha256=sha256_file(path), video=video)
    result['evaluation'] = frozen
    return result


def aggregate(pairs):
    groups, paired = [], []
    for game in GAMES:
        by_method = {}
        for method in METHODS:
            rows = sorted((p for p in pairs if p['game'] == game and p['method'] == method),
                          key=lambda p: p['training']['seed'])
            seeds = [p['training']['seed'] for p in rows]
            require(len(set(seeds)) == len(seeds) and set(seeds) <= set(SEEDS), 'duplicate/undeclared learner seed')
            runs = [p['training'] for p in rows]
            complete = seeds == list(SEEDS)
            group = dict(game=game, method=method, runs=runs,
                         aggregate=summarize_curves(runs) if complete else None,
                         online_score=mean_ci([r['curve'][-1]['score'] for r in runs]) if complete else None,
                         seconds=mean_ci([r['final']['elapsed_seconds'] for r in runs]) if complete else None,
                         frozen_score=None)
            if complete and all(p['evaluation']['cohort']['complete'] for p in rows):
                group['frozen_score'] = mean_ci([p['evaluation']['cohort']['score'] for p in rows])
            groups.append(group)
            by_method[method] = rows if complete else None
        for control in ('rgb', 'pretrained_tiny'):
            candidate, reference = by_method['cdp'], by_method[control]
            if candidate is None or reference is None:
                continue
            frozen_complete = all(p['evaluation']['cohort']['complete'] for p in candidate + reference)
            paired.append(dict(game=game, candidate='cdp', control=control,
                online_difference=mean_ci([a['training']['curve'][-1]['score'] - b['training']['curve'][-1]['score']
                                           for a, b in zip(candidate, reference)]),
                seconds_ratio=mean_ci([a['training']['final']['elapsed_seconds'] / b['training']['final']['elapsed_seconds']
                                       for a, b in zip(candidate, reference)]),
                frozen_difference=mean_ci([a['evaluation']['cohort']['score'] - b['evaluation']['cohort']['score']
                                            for a, b in zip(candidate, reference)]) if frozen_complete else None))
    return dict(protocol='kindle-cdp-atari-comparison-v1', comparison='cdp', games=GAMES, methods=METHODS,
                num_envs=8, action_budget=BUDGET, completed_pairs=len(pairs), planned_pairs=45,
                status='complete' if len(pairs) == 45 else 'partial', results=groups, paired=paired,
                evaluations=[dict(name=p['name'], **p['evaluation']) for p in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--game', choices=GAMES)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--require-replay', action='store_true')
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    selection = (args.game, args.method, args.seed)
    if any(v is not None for v in selection):
        if not all(v is not None for v in selection) or args.plot:
            parser.error('a single-run audit needs game, method and seed, without plot')
        if args.training_only and args.require_replay:
            parser.error('replay is a frozen-evaluation audit')
        result = (training(args.root, *selection) if args.training_only else
                  pair(args.root, *selection, require_replay=args.require_replay))
    else:
        if args.training_only:
            parser.error('training-only needs a single run')
        pairs, shared = [], None
        for game in GAMES:
            for method in METHODS:
                for seed in SEEDS:
                    name = run_name(game, method, seed) + '-frozen'
                    if not (args.root / f'{name}-queue' / name / 'result.json').exists():
                        continue
                    row = pair(args.root, game, method, seed, require_replay=args.require_replay)
                    require(shared is None or shared == row['shared_recipe'], 'changed shared recipe')
                    shared = row['shared_recipe']
                    pairs.append(row)
        result = aggregate(pairs)
        result['shared_recipe'] = shared
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    if args.plot:
        with args.plot.open('x') as plot:
            plot.write(plot_svg(result))
    print(json.dumps(dict(output=str(args.output), name=result.get('name'), completed_pairs=result.get('completed_pairs'))))


if __name__ == '__main__':
    main()
