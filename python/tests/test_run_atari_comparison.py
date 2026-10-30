import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import run_atari_comparison as study


def test_schedule_covers_all_five_games_methods_and_seeds_with_balanced_order():
    rows = study.schedule()
    assert len(rows) == len(set(rows)) == 45
    assert rows[:3] == [('Freeway', m, 1009) for m in study.METHODS]
    assert rows[3:6] == [('Freeway', m, 2017) for m in ('rgb', 'pretrained_tiny', 'cdp')]
    assert rows[6:9] == [('Freeway', m, 3019) for m in ('pretrained_tiny', 'cdp', 'rgb')]
    assert {game for game, _, _ in rows} == {'Freeway', 'Boxing', 'Pong', 'Breakout', 'Qbert'}


@pytest.mark.parametrize('method', study.METHODS)
@pytest.mark.parametrize('frozen', [False, True])
def test_job_keeps_recipe_and_frozen_restore_options_separate(tmp_path, method, frozen):
    job = study.job(tmp_path, tmp_path / 'encoder', 'Pong', method, 2017, frozen=frozen)
    args = job['command']
    assert all(isinstance(v, str) for v in args)
    assert args[args.index('--steps') + 1] == '200000'
    assert args[args.index('--num-envs') + 1] == '8'
    assert args[args.index('--seed') + 1] == str(1000002017 if frozen else 2017)
    assert ('--encoder-checkpoint' in args) == (method == 'pretrained_tiny')
    assert ('--cdp' in args) == (method == 'cdp' and not frozen)
    assert ('--restore' in args) == ('--evaluate' in args) == frozen
    assert ('--model-size' in args) != frozen
    assert job['timeout_seconds'] == (1800 if frozen else 3600)
    assert job['expected']['learner_updates'] == (0 if frozen else 49939)


def test_cpu_analysis_is_bounded_to_one_cpu_and_two_gib(monkeypatch):
    calls = []
    monkeypatch.setattr(study.os, 'sched_getaffinity', lambda _: {3, 4})
    monkeypatch.setattr(study.subprocess, 'run', lambda *args, **kwargs: calls.append((args, kwargs)))
    study.cpu(['fixture.py', Path('/fixture')])
    args, options = calls[0]
    assert args[0][:6] == ['taskset', '-c', '3', 'prlimit', '--as=2147483648', '--']
    assert options['check'] and options['timeout'] == 1800


@pytest.mark.parametrize('state,failed', [
    ('LoadState=loaded\nActiveState=inactive\nResult=success\nExecMainStatus=0', False),
    ('LoadState=not-found\nActiveState=inactive\nResult=success\nExecMainStatus=0', False),
    ('LoadState=loaded\nActiveState=failed\nResult=exit-code\nExecMainStatus=1', True),
])
def test_terminal_service_requires_a_separate_artifact_audit(monkeypatch, state, failed):
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *_, **__: state)
    if failed:
        with pytest.raises(RuntimeError, match='failed'):
            study.wait_service('fixture.service')
    else:
        study.wait_service('fixture.service')


@pytest.mark.parametrize('failure', [None, 'training', 'training-audit', 'evaluation', 'replay', 'pair-audit'])
def test_study_advances_only_after_all_stage_checks_and_never_retries(tmp_path, monkeypatch, failure):
    calls = []
    monkeypatch.setattr(study, 'importlib', SimpleNamespace(util=SimpleNamespace(
        find_spec=lambda _: SimpleNamespace(origin=str(tmp_path / 'native')))))
    monkeypatch.setattr(study, 'schedule', lambda: [('Freeway', 'cdp', 1009), ('Freeway', 'rgb', 1009)])
    monkeypatch.setattr(study, 'sha256_file', lambda _: 'identity')
    monkeypatch.setattr(study, 'NATIVE_SHA256', 'identity')
    monkeypatch.setitem(study.TINY_CHECKPOINTS, 'pretrained_tiny', 'identity')
    monkeypatch.setattr(study, 'rom_identity', lambda _: dict(path='fixture.rom', sha256='fixture'))
    def stage(name):
        calls.append(name)
        if failure == name:
            raise subprocess.CalledProcessError(1, ['fixture', name])
    def native(spec, _output):
        stage('evaluation' if spec['jobs'][0]['name'].endswith('-frozen') else 'training')
    def cpu(command):
        stage('replay' if Path(command[0]).name == 'replay_atari.py' else
              'training-audit' if '--training-only' in command else 'pair-audit')
    monkeypatch.setattr(study, 'run_jobs', native)
    monkeypatch.setattr(study, 'cpu', cpu)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            study.run(tmp_path, dict(record_allocation_warnings=True), Path('encoder'))
    else:
        study.run(tmp_path, dict(record_allocation_warnings=True), Path('encoder'))
    result = json.loads((tmp_path / 'study-result.json').read_text())
    order = ['training', 'training-audit', 'evaluation', 'replay', 'pair-audit']
    assert calls == (order[:order.index(failure) + 1] if failure else order * 2)
    assert result['status'] == ('stopped' if failure else 'complete')
    assert len(result['completed']) == (0 if failure else 2)
    with pytest.raises(FileExistsError):
        study.run(tmp_path, dict(record_allocation_warnings=True), Path('encoder'))
