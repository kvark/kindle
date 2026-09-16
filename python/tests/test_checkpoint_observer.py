import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import observe_atari_checkpoint as observer
from test_checkpoint_retention import case, freeway, write_log


def append(path, row, newline=True):
    with path.open('ab') as output:
        output.write(json.dumps(row).encode() + (b'\n' if newline else b''))


def current(command=None):
    return dict(pid=12, state='S', parent_pid=11, start_ticks='1234',
                command=['synthetic reader fixture'] if command is None else command)


def observe(case, **options):
    arguments = {key: value for key, value in case.items() if key != 'rows'}
    return observer.observe_midpoint(**arguments, **options)


def test_partial_rows_wait_then_one_complete_checkpoint(case):
    path = case['log']
    path.unlink()
    reader = observer.Midpoint(path, case['expected_header'], case['action'])
    assert reader.poll() is None
    for row in case['rows']:
        position = reader.position
        append(path, row, newline=False)
        assert reader.poll() is None and reader.position == position
        with path.open('ab') as output:
            output.write(b'\n')
        result = reader.poll()
    assert result == case['rows'][-1]
    with pytest.raises(ValueError, match='terminal'):
        reader.poll()


@pytest.mark.parametrize('game', ['case', 'freeway'])
def test_observer_retains_complete_synthetic_state_once(request, game):
    case = request.getfixturevalue(game)
    child, events = current(), []
    before = {path: observer.retention.digest(path) for path in
              [case['log'], *(case['checkpoint'] / name for name in observer.retention.FILES)]}
    result = observe(case, bound=child, inspect=lambda _: child,
                     emit=lambda event, **fields: events.append((event, fields)))
    assert result['complete_finite_state']['optimizer_moments'] == 146
    assert result['complete_finite_state']['environment_step'] == 12
    assert result['complete_training'] is result['learner_called'] is False
    assert [event for event, _ in events] == ['midpoint_retention_start', 'midpoint_retention_complete']
    assert all(observer.retention.digest(path) == value for path, value in before.items())
    with pytest.raises(ValueError, match='already exists'):
        observe(case, bound=child, inspect=lambda _: child)


def test_runtime_timestamps_do_not_change_the_recipe(case):
    case['expected_header'].update(unix_time=1, agent_construction_seconds=2)
    case['rows'][0].update(unix_time=3, agent_construction_seconds=4)
    write_log(case)
    reader = observer.Midpoint(case['log'], case['expected_header'], case['action'])
    assert reader.poll() == case['rows'][-1]


def test_waits_for_new_rows_without_restarting_child(case, monkeypatch):
    case['log'].unlink()
    child, sleeps = current(), []
    def append_checkpoint(seconds):
        sleeps.append(seconds)
        write_log(case)
    monkeypatch.setattr(observer.time, 'sleep', append_checkpoint)
    result = observe(case, bound=child, inspect=lambda _: child)
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 1
    assert result['complete_finite_state']['optimizer_moments'] == 146


def test_exit_while_waiting_fails_before_copy(case, monkeypatch):
    case['log'].unlink()
    child = current()
    active = [child]
    def exit_child(_seconds):
        active[0] = None
    monkeypatch.setattr(observer.time, 'sleep', exit_child)
    with pytest.raises(ValueError, match='not live'):
        observe(case, bound=child, inspect=lambda _: active[0])
    assert not case['destination'].exists()


@pytest.mark.parametrize('field,value', [('seed', 1), ('steps', 30), ('mode', 'evaluate_sample'),
    ('protocol', 'kindle-vector-v4'), ('config', {}), ('model_provenance', {})])
def test_changed_runtime_header_is_terminal(case, field, value):
    case['rows'][0][field] = value
    write_log(case)
    reader = observer.Midpoint(case['log'], case['expected_header'], case['action'])
    with pytest.raises(ValueError, match='header differs'):
        reader.poll()
    with pytest.raises(ValueError, match='terminal'):
        reader.poll()


@pytest.mark.parametrize('row,message', [
    ({'event': 'transition', 'run_step': 18}, 'missed'),
    ({'event': 'checkpoint', 'run_step': 6}, 'cadence'),
    ({'event': 'run_end', 'run_step': 12}, 'ended'),
    ({'event': 'run_start'}, 'duplicate'),
    ({'event': 'learner', 'run_step': -1}, 'missed'),
    ({'event': 'learner', 'run_step': True}, 'missed'),
    ({'event': 'learner'}, 'missed')])
def test_missing_or_changed_midpoint_is_not_retried(case, row, message):
    case['rows'] = [case['rows'][0], row]
    write_log(case)
    reader = observer.Midpoint(case['log'], case['expected_header'], case['action'])
    with pytest.raises(ValueError, match=message):
        reader.poll()
    assert reader.failed


def test_reordered_steps_fail(case):
    case['rows'][2]['run_step'] = 0
    write_log(case)
    reader = observer.Midpoint(case['log'], case['expected_header'], case['action'])
    with pytest.raises(ValueError, match='reordered'):
        reader.poll()


@pytest.mark.parametrize('kind', ['replace', 'truncate', 'partial_truncate', 'disappear', 'symlink'])
def test_moving_log_is_terminal(case, kind):
    path = case['log']
    case['rows'] = case['rows'][:1]
    write_log(case)
    if kind == 'partial_truncate':
        with path.open('ab') as output:
            output.write(b'partial record')
    reader = observer.Midpoint(path, case['expected_header'], case['action'])
    assert reader.poll() is None
    if kind in ('replace', 'disappear', 'symlink'):
        old = path.with_suffix('.old')
        path.rename(old)
        if kind == 'replace':
            write_log(case)
        elif kind == 'symlink':
            path.symlink_to(old)
    elif kind == 'partial_truncate':
        write_log(case)
    else:
        path.write_bytes(b'')
    with pytest.raises(ValueError):
        reader.poll()
    assert reader.failed


@pytest.mark.parametrize('data', [b'x' * (observer.MAX_ROW_BYTES + 1), b'not json\n'])
def test_oversized_or_malformed_row_is_terminal(case, data):
    case['log'].write_bytes(data)
    reader = observer.Midpoint(case['log'], case['expected_header'], case['action'])
    with pytest.raises(ValueError):
        reader.poll()
    assert reader.failed


@pytest.mark.parametrize('action', [0, -6, 5, 12.0, True, 24, 30])
def test_invalid_or_final_action_is_not_observed(case, action):
    with pytest.raises(ValueError):
        observer.Midpoint(case['log'], case['expected_header'], action)


@pytest.mark.parametrize('field,value', [('pid', 13), ('start_ticks', '999'), ('parent_pid', 1),
    ('state', 'Z'), ('state', 'X'), ('state', 'x'), ('command', ['other'])])
def test_reused_dead_or_reparented_child_fails(field, value):
    original = current()
    with pytest.raises(ValueError):
        observer.require_child(original, dict(original, **{field: value}))


def test_process_binding_checks_twice():
    child = current()
    observed = iter([child, dict(child, start_ticks='changed')])
    with pytest.raises(ValueError, match='identity changed'):
        observer.bind_child(12, child['command'], 11, inspect=lambda _: next(observed))


@pytest.mark.parametrize('pid', [0, -1, True, 1.5])
def test_invalid_pid_fails_before_proc_access(pid):
    with pytest.raises(ValueError, match='invalid process PID'):
        observer.process(pid)


def test_actual_proc_binding_and_retention_on_harmless_child(case):
    command = [sys.executable, '-B', '-c', "import sys; print('ready', flush=True); sys.stdin.readline()"]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
        try:
            assert child.stdout.readline().strip() == 'ready'
            bound = observer.bind_child(child.pid, command, os.getpid())
            with pytest.raises(ValueError, match='owning controller'):
                observer.bind_child(child.pid, command, os.getpid() + 1)
            with pytest.raises(ValueError, match='command'):
                observer.bind_child(child.pid, ['wrong command'], os.getpid())
            result = observe(case, bound=bound)
            assert result['complete_finite_state']['environment_step'] == 12
            assert child.poll() is None
        finally:
            child.stdin.write('\n')
            child.stdin.flush()
            child.wait(timeout=10)
    assert observer.process(child.pid) is None


def test_dead_child_fails_before_copy(case):
    with pytest.raises(ValueError, match='not live'):
        observe(case, bound=current(), inspect=lambda _: None)
    assert not case['destination'].exists()


def test_failed_copy_propagates_without_retry(case, monkeypatch):
    calls, events = [], []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError('deliberate copy failure')
    monkeypatch.setattr(observer.retention, 'retain', fail)
    child = current()
    with pytest.raises(ValueError, match='deliberate'):
        observe(case, bound=child, inspect=lambda _: child, emit=lambda event, **_: events.append(event))
    assert len(calls) == 1 and events == ['midpoint_retention_start']


@pytest.mark.parametrize('failure', ['child_exit', 'event_changed'])
def test_failed_post_copy_check_emits_no_observer_success(case, monkeypatch, failure):
    child, events = current(), []
    active = [child]
    retain = observer.retention.retain
    def changed(*args, **kwargs):
        result = retain(*args, **kwargs)
        if failure == 'child_exit':
            active[0] = None
        else:
            result['source_checkpoint_event'] = dict(result['source_checkpoint_event'], learner_step=999)
        return result
    monkeypatch.setattr(observer.retention, 'retain', changed)
    with pytest.raises(ValueError):
        observe(case, bound=child, inspect=lambda _: active[0], emit=lambda event, **_: events.append(event))
    assert (case['destination'] / 'retained.json').is_file()
    assert events == ['midpoint_retention_start']


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), True, 64801])
def test_invalid_timeout_never_observes_or_writes(case, timeout):
    with pytest.raises(ValueError, match='timeout'):
        observe(case, bound={}, timeout_seconds=timeout)
    assert not case['destination'].exists()


def test_expired_wait_fails_without_copy(case, monkeypatch):
    clock = iter([0, 2])
    monkeypatch.setattr(observer.time, 'monotonic', lambda: next(clock))
    child = current()
    with pytest.raises(ValueError, match='wait expired'):
        observe(case, bound=child, inspect=lambda _: child, timeout_seconds=1)
    assert not case['destination'].exists()


def test_deadline_during_copy_is_not_observer_success(case, monkeypatch):
    clock = iter([0, 0, 2])
    monkeypatch.setattr(observer.time, 'monotonic', lambda: next(clock))
    child, events = current(), []
    with pytest.raises(ValueError, match='expired during copy'):
        observe(case, bound=child, inspect=lambda _: child, timeout_seconds=1,
                emit=lambda event, **_: events.append(event))
    assert (case['destination'] / 'retained.json').is_file()
    assert events == ['midpoint_retention_start']
