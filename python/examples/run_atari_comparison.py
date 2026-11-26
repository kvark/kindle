"""Execute the fixed five-game study serially, auditing each stage before advancing.

No retries, implicit resumes, GPU recovery or new learning settings. CPU audits
and exact replay use one CPU, a2GiB address-space limit and inherited zero swap.
Run this controller in a bounded systemd service with KillMode=control-group.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import ale_py
import ale_py._ale_py as ale_native
import gymnasium as gym

from atari import sha256_file
from check_atari_adapter import rom_identity
from run_gpu_queue import run as run_jobs
from summarize_atari_comparison import BUDGET, GAMES, METHODS, NATIVE_SHA256, SEEDS, UPDATES, run_name
from summarize_representation_learning import TINY_CHECKPOINTS


EXAMPLES = Path(__file__).resolve().parent
ENVIRONMENT = dict(MEGANEURA_DEVICE_ID='0x2c02', KINDLE_EXPECT_DEVICE_NAME='NVIDIA GeForce RTX 5080',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def schedule():
    return [(game, method, seed) for game in GAMES for i, seed in enumerate(SEEDS)
            for method in METHODS[i:] + METHODS[:i]]


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def job(root, encoder, game, method, seed, *, frozen=False):
    name = run_name(game, method, seed)
    training_name = name
    if frozen:
        name += '-frozen'
    command = [sys.executable, EXAMPLES / 'atari_vector.py', f'ALE/{game}-v5',
               '--output', root / f'{name}.jsonl', '--checkpoint', root / f'{name}-checkpoint',
               '--checkpoint-every', BUDGET, '--steps', BUDGET, '--num-envs', 8, '--report-every', 2000,
               '--seed', 1000000000 + seed if frozen else seed, '--atari-protocol', 'published',
               '--sticky-actions', .25, '--observation-size', 'native', '--min-gpu-budget-headroom-mib', 2048]
    if method == 'pretrained_tiny':
        command += ['--encoder-checkpoint', encoder]
        if not frozen:
            command += ['--encoder', 'levjepa-tiny']
    if frozen:
        command += ['--restore', root / f'{training_name}-checkpoint', '--evaluate', '--episodes-per-env', 3]
        expected = dict(event='run_end', learner_updates=0)
    else:
        command += ['--model-size', '1m', '--batch-size', 8, '--batch-length', 16,
                    '--world-microbatch-size', 8, '--train-ratio', 32, '--replay-capacity', 100000]
        if method == 'cdp':
            command += ['--cdp']
        expected = dict(event='run_end', reason='budget_complete', run_step=BUDGET,
                        learner_updates=UPDATES, training_debt=0)
    return dict(name=name, command=list(map(str, command)), executable_sha256=sha256_file(sys.executable),
                timeout_seconds=1800 if frozen else 3600, environment=ENVIRONMENT,
                result=str(root / f'{name}.jsonl'), result_format='jsonl_last', expected=expected)


def cpu(command, *, timeout=1800):
    subprocess.run(['taskset', '-c', str(min(os.sched_getaffinity(0))), 'prlimit', '--as=2147483648', '--',
                    sys.executable, *map(str, command)], check=True, timeout=timeout,
                   env={**os.environ, **ENVIRONMENT})


def wait_service(unit):
    deadline = time.monotonic() + 4000
    while True:
        text = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'LoadState',
                                        '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
        state = dict(line.split('=', 1) for line in text.splitlines())
        if state['LoadState'] == 'not-found':
            return  # The subsequent guard/counter audit must prove completion.
        if state['ActiveState'] not in ('active', 'activating', 'deactivating'):
            if state['ActiveState'] != 'inactive' or state['Result'] != 'success' or state['ExecMainStatus'] != '0':
                raise RuntimeError(f'initial training service failed: {state}')
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('initial training service did not finish')
        time.sleep(30)


def run(root, host, encoder, *, first_training_service=None):
    entries = schedule()
    native = Path(importlib.util.find_spec('kindle._native').origin)
    actor_identity = sha256_file(EXAMPLES / 'atari_vector.py')
    if host.get('record_allocation_warnings') is not True:
        raise ValueError('use the authorized warning-recording policy')
    if sha256_file(encoder) != TINY_CHECKPOINTS['pretrained_tiny']:
        raise ValueError('not the declared pretrained Tiny encoder')
    write(root / 'study-specification.json', dict(protocol='kindle-cdp-atari-comparison-v1', host=host,
          entries=entries, first_training_service=first_training_service, encoder=str(encoder),
          encoder_sha256=sha256_file(encoder), native_sha256=NATIVE_SHA256, runner_sha256=actor_identity,
          controller_sha256=sha256_file(__file__), training_actions=BUDGET, training_updates=UPDATES,
          frozen_episode_target=3, frozen_action_cap=BUDGET, automatic_retries=False))
    completed, stage, current = [], 'prepare', None
    try:
        gym.register_envs(ale_py)
        for game in GAMES:
            rom = rom_identity(f'ALE/{game}-v5')
            write(root / f'{game.lower()}-replay-manifest.json', dict(environment=f'ALE/{game}-v5',
                  atari_protocol='published', observation_size='native', sticky_actions=.25,
                  pins={rom['path']: rom['sha256'], str(ale_native.__file__): sha256_file(ale_native.__file__)}))
        for index, (game, method, seed) in enumerate(entries):
            current = run_name(game, method, seed)
            if sha256_file(native) != NATIVE_SHA256 or sha256_file(EXAMPLES / 'atari_vector.py') != actor_identity:
                raise ValueError('qualified native library or actor changed during study')
            stage = 'training'
            if index == 0 and first_training_service:
                wait_service(first_training_service)
            else:
                run_jobs(dict(host=host, jobs=[job(root, encoder, game, method, seed)]), root / f'{current}-queue')
            audit = [EXAMPLES / 'summarize_atari_comparison.py', root, '--game', game,
                     '--method', method, '--seed', seed]
            stage = 'training-audit'
            cpu([*audit, '--training-only', '--output', root / f'{current}-training-audit.json'])
            stage = 'frozen-evaluation'
            frozen = current + '-frozen'
            run_jobs(dict(host=host, jobs=[job(root, encoder, game, method, seed, frozen=True)]), root / f'{frozen}-queue')
            stage = 'exact-replay'
            cpu([EXAMPLES / 'replay_atari.py', root / f'{frozen}.jsonl', '--allow-capped-evaluation',
                 '--source-manifest', root / f'{game.lower()}-replay-manifest.json',
                 '--output', root / f'{frozen}-replay.json', '--video', root / f'{frozen}.mp4'])
            stage = 'pair-audit'
            cpu([*audit, '--require-replay', '--output', root / f'{current}-pair-audit.json'])
            completed.append(current)
            write(root / f'completed-{len(completed):02}.json', dict(completed=completed.copy()))
            print(f'Audited {len(completed)}/{len(entries)} pairs: {current}', flush=True)
            if (index + 1) % 9 == 0:
                stage = 'summary'
                cpu([EXAMPLES / 'summarize_atari_comparison.py', root, '--require-replay',
                     '--output', root / f'summary-{len(completed):02}.json',
                     '--plot', root / f'curves-{len(completed):02}.svg'])
        write(root / 'study-result.json', dict(status='complete', completed=completed))
    except BaseException as error:
        write(root / 'study-result.json', dict(status='stopped', stage=stage, current=current,
                                              completed=completed, reason=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--host-specification', required=True, type=Path)
    parser.add_argument('--encoder-checkpoint', required=True, type=Path)
    parser.add_argument('--first-training-service')
    args = parser.parse_args()
    def interrupted(_signal, _frame):
        raise InterruptedError('study controller interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.root.resolve(), json.loads(args.host_specification.read_text())['host'],
        args.encoder_checkpoint.resolve(), first_training_service=args.first_training_service)


if __name__ == '__main__':
    main()
