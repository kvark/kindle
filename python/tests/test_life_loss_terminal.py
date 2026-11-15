"""Training-only life-loss terminals keep whole games in the emulator and the log; no GPU."""
import json
from pathlib import Path
import sys

import kindle
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import atari_vector
from kindle._vector_audit import audit
from test_replay_actor import command
from test_return_boundaries import RawEnvironment


@pytest.mark.parametrize('extra', [['--evaluate'], ['--restore', 'unused'],
    ['--replay-actor-scale', '100', '--disagreement-scale', '1', '--replay-actor-observed-lower-bound',
     '--breakout-return-life-boundaries']])
def test_invalid_life_terminal_options_fail_before_environment(monkeypatch, tmp_path, capsys, extra):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', command(tmp_path, ['--life-loss-terminal', *extra]))
    with pytest.raises(SystemExit): atari_vector.main()
    assert 'life-loss-terminal requires fresh training' in capsys.readouterr().err


@pytest.mark.parametrize('enabled', [False, True])
def test_life_terminals_reach_only_the_stored_stream(monkeypatch, tmp_path, enabled):
    environments, observed, begun = [], [], []

    class Agent:
        environment_step = learner_step = replay_len = training_debt = 0
        provenance = gpu_device = trainable_parameter_counts = {}
        cpu_worker_threads = 1

        @classmethod
        def learned_rgb(cls, streams, config):
            instance = cls()
            instance.config = config
            return instance

        def begin_episodes(self, streams, frames):
            begun.append((list(streams), [int(frame.max()) for frame in frames]))
            self.replay_len += len(streams)

        def act(self, *, greedy): return [0, 0]

        def observe(self, streams, frames, rewards, terminated, truncated):
            observed.append((list(terminated), list(truncated), [int(frame.max()) for frame in frames]))
            self.environment_step += len(streams)
            self.replay_len += len(streams)
            return [[reward, 0.] for reward in rewards]

        def learn_scheduled(self): return []

    def make(*args, **kwargs):
        environments.append(RawEnvironment())
        return environments[-1]

    kindle.default_config  # load the native module first; its lazy import would undo the patch
    monkeypatch.setitem(kindle.__dict__, 'VectorAgent', Agent)
    monkeypatch.setattr(atari_vector.gym, 'make', make)
    options = ['--num-envs', '2', '--steps', '6', '--train-ratio', '0', '--cdp']
    monkeypatch.setattr(sys, 'argv', command(tmp_path, options + (['--life-loss-terminal'] if enabled else [])))
    atari_vector.main()
    rows = [json.loads(line) for line in (tmp_path/'run.jsonl').read_text().splitlines()]
    transitions = [row for row in rows if row['event'] == 'transition']
    # The fixture loses a life in the first action of each game and ends on the second.
    assert [row['terminated'] for row in transitions] == [[False] * 2, [True] * 2, [False] * 2]
    assert rows[-1]['episode_counts'] == [1, 1] and rows[-1]['total_rewards'] == [12., 12.]
    assert [row['streams'] for row in rows if row['event'] == 'reset'] == [[0, 1]]
    assert (rows[0].get('life_loss_terminal_kind') == atari_vector.LIFE_TERMINAL_KIND) == enabled
    assert 'actor_return_boundary_kind' not in rows[0]
    if enabled:
        assert [stored[0] for stored in observed] == [[True] * 2] * 3
        assert [row['life_terminals'] for row in transitions] == [[True] * 2, [False] * 2, [True] * 2]
        # Stored episodes restart on the frame that showed the loss; the game is reset only when it ends.
        assert [streams for streams, _ in begun] == [[0, 1]] * 4
        assert begun[1][1] == observed[0][2] and begun[3][1] == observed[2][2]
        assert begun[2][1] != observed[1][2]
    else:
        assert [stored[0] for stored in observed] == [[False] * 2, [True] * 2, [False] * 2]
        assert all('life_terminals' not in row for row in transitions)
        assert [streams for streams, _ in begun] == [[0, 1]] * 2
    assert all(not any(stored[1]) for stored in observed)
    assert audit(tmp_path/'run.jsonl')['updates'] == 0
    if enabled:
        for mutation in ('undeclared', 'flag', 'boundary'):
            changed = json.loads(json.dumps(rows))
            target = next(row for row in changed if row['event'] == 'transition')
            if mutation == 'undeclared': del changed[0]['life_loss_terminal_kind']
            elif mutation == 'flag': target['life_terminals'] = [1, 1]
            else: target['life_terminals'], target['terminated'] = [True, True], [True, True]
            (tmp_path/'changed.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in changed))
            with pytest.raises(ValueError): audit(tmp_path/'changed.jsonl')
