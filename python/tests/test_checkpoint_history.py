"""Checkpoint-path fixtures with a fake agent, not native state or gameplay results."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import kindle
import pytest
from kindle._vector_audit import audit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import atari_vector

FILES = ('metadata.json', 'world.safetensors', 'behavior.safetensors', 'slow_value.safetensors')


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    cases = []

    def run(*, history=False, steps=6, every=4, exploration=False, collision=None, fail_at=None):
        root = tmp_path / str(len(cases))
        output, checkpoint = root / 'run.jsonl', root / 'checkpoints'
        case = dict(root=root, output=output, checkpoint=checkpoint, saves=[], environments=[])
        cases.append(case)

        class Environment:
            action_space = SimpleNamespace(n=2)
            action_meanings = ['NOOP', 'FIRE']
            executed_action_frames = reset_noop_frames = emulator_resets = 0
            closed = False

            def reset(self, *, seed=None):
                self.length = 0
                self.emulator_resets += 1
                return None, {}

            def step(self, action):
                self.length += 1
                self.executed_action_frames += 4
                return None, float(action), self.length == 2, False, {}

            def close(self):
                self.closed = True

        class Agent:
            environment_step = learner_step = replay_len = 0
            training_debt = 0.0
            provenance = gpu_device = {'explicit_synthetic_fixture': True}
            cpu_worker_threads = 1
            trainable_parameter_counts = {'world': 0, 'behavior': 0}
            gpu_memory_budget = dict(usage_bytes=1024**3, budget_bytes=3 * 1024**3)

            def __init__(self, weights, streams, config):
                self.streams, self.config = streams, config

            def begin_episodes(self, ids, frames):
                self.replay_len += len(ids)

            def act(self, *, greedy, action_overrides=None):
                return [0 if value is None else value for value in
                        (action_overrides if action_overrides is not None else [None] * self.streams)]

            def observe(self, ids, frames, rewards, terminated, truncated):
                self.environment_step += len(ids)
                self.replay_len += len(ids)
                if collision and self.environment_step == 4:
                    checkpoint.mkdir(parents=True)
                    slot = checkpoint / '4'
                    if collision == 'file':
                        slot.write_text('preserve this existing file')
                    else:
                        target = slot if collision == 'directory' else root / 'untouched'
                        target.mkdir()
                        (target / 'sentinel').write_text('preserve this directory')
                        if collision == 'symlink':
                            slot.symlink_to(target, target_is_directory=True)
                return [[reward, 0.0] for reward in rewards]

            def learn_scheduled(self):
                return []

            def save_checkpoint(self, path):
                path = Path(path)
                case['saves'].append(path)
                path.mkdir(parents=True, exist_ok=True)
                for name in FILES:
                    (path / name).write_text(f'Explicit synthetic state identity at action {self.environment_step}\n')
                    if self.environment_step == fail_at:
                        raise RuntimeError('deliberate checkpoint write failure')

        def make(*_, **__):
            env = Environment()
            case['environments'].append(env)
            return env

        monkeypatch.setattr(atari_vector.gym, 'make', make)
        monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **_: env)
        monkeypatch.setattr(kindle, 'VectorAgent', Agent)
        monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'unused', 'ALE/Freeway-v5',
            '--output', str(output), '--steps', str(steps), '--num-envs', '2', '--train-ratio', '0',
            '--checkpoint', str(checkpoint), '--checkpoint-every', str(every),
            '--min-gpu-budget-headroom-mib', '2048',
            *(['--checkpoint-history'] if history else []),
            *(['--exploration-probability', '.5', '--exploration-hold', '64'] if exploration else [])])
        try:
            atari_vector.main()
        finally:
            case['rows'] = [json.loads(line) for line in output.read_text().splitlines()]
            case['memory'] = [json.loads(line) for line in output.with_suffix('.gpu-memory.jsonl').read_text().splitlines()]
        return case

    return run, cases


@pytest.mark.parametrize('steps,every,expected', [(6, 4, [4, 6]), (8, 4, [4, 8]), (6, 100, [6])])
@pytest.mark.parametrize('history', [False, True])
def test_default_replaces_last_save_and_history_preserves_every_slot(fake_runner, history, steps, every, expected):
    run, _ = fake_runner
    case = run(history=history, steps=steps, every=every)
    paths = [case['checkpoint'] / str(step) if history else case['checkpoint'] for step in expected]
    assert case['saves'] == paths
    assert audit(case['output'])['budget_complete']
    assert all(env.closed for env in case['environments'])
    header = case['rows'][0]
    assert header.get('checkpoint_history', False) is history
    if not history:
        assert 'checkpoint_history' not in header
    saves = [row for row in case['rows'] if row['event'] == 'checkpoint']
    assert [row['run_step'] for row in saves] == expected
    assert [row['identity']['path'] for row in saves] == [str(path) for path in paths]
    for step, path, event in zip(expected, paths, saves):
        if history or step == expected[-1]:
            assert event['identity'] == atari_vector.checkpoint_identity(path)
            assert (path / 'metadata.json').read_text().endswith(f'action {step}\n')
    samples = [row for row in case['memory'] if row['stage'] == 'checkpoint']
    assert [row['run_step'] for row in samples] == expected
    assert all(row['minimum_headroom_bytes'] == 2 * 1024**3 for row in samples)


@pytest.mark.parametrize('exploration', [False, True])
def test_history_changes_only_storage_not_interactions(fake_runner, exploration):
    run, _ = fake_runner
    default, history = run(exploration=exploration), run(history=True, exploration=exploration)
    def transitions(case):
        return [{key: value for key, value in row.items() if key != 'unix_time'} for row in case['rows']
                if row['event'] in ('transition', 'learner', 'episode', 'reset')]
    assert transitions(default) == transitions(history)
    left, right = audit(default['output']), audit(history['output'])
    assert left.pop('path') == str(default['output'])
    assert right.pop('path') == str(history['output'])
    assert left == right


@pytest.mark.parametrize('collision', ['file', 'directory', 'symlink'])
def test_history_never_overwrites_an_existing_slot(fake_runner, collision):
    run, cases = fake_runner
    with pytest.raises(FileExistsError):
        run(history=True, collision=collision)
    case = cases[-1]
    assert not case['saves']
    slot = case['checkpoint'] / '4'
    if collision == 'file':
        assert slot.read_text() == 'preserve this existing file'
    else:
        assert (slot / 'sentinel').read_text() == 'preserve this directory'
    assert all(row['event'] not in ('checkpoint', 'run_end') for row in case['rows'])
    assert all(env.closed for env in case['environments'])


def test_failed_save_keeps_prior_slot_and_emits_no_success(fake_runner):
    run, cases = fake_runner
    with pytest.raises(RuntimeError, match='deliberate'):
        run(history=True, every=2, fail_at=4)
    case = cases[-1]
    saves = [row for row in case['rows'] if row['event'] == 'checkpoint']
    assert len(saves) == 1 and saves[0]['run_step'] == 2
    assert saves[0]['identity'] == atari_vector.checkpoint_identity(case['checkpoint'] / '2')
    assert not any(row['event'] == 'run_end' for row in case['rows'])
    assert (case['checkpoint'] / '4' / 'metadata.json').is_file()
    assert not (case['checkpoint'] / '4' / 'world.safetensors').exists()
    assert all(env.closed for env in case['environments'])


def test_history_requires_a_destination_before_constructing_anything(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('environment or agent constructed before option refusal')
    monkeypatch.setattr(atari_vector.gym, 'make', unexpected)
    monkeypatch.setattr(kindle, 'VectorAgent', unexpected)
    output = tmp_path / 'absent.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'unused', '--output', str(output), '--checkpoint-history'])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
