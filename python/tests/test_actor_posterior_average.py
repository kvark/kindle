import json
import sys
from types import SimpleNamespace

import pytest
import kindle
import atari_vector
from kindle._vector_audit import audit
from test_episode_evaluation import frozen_run


@pytest.mark.parametrize('args,message', [
    *[(['--actor-posterior-samples', value], 'between 2 and 16') for value in ['-1', '0', '1', '17', str(2**64)]],
    (['--actor-posterior-samples', '8'], 'requires frozen restore'),
    (['--actor-posterior-samples', '8', '--evaluate'], 'requires frozen restore'),
    (['--actor-posterior-samples', '8', '--restore', 'unused'], 'requires frozen restore'),
    (['--actor-posterior-samples', '8', '--evaluate', '--restore', 'unused', '--greedy-posterior'], 'sampled live posteriors'),
    (['--actor-posterior-samples', '8', '--evaluate', '--restore', 'unused', '--action-temperature', '1'], 'no action temperature'),
])
def test_actor_average_rejects_invalid_cli_before_construction(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('unexpected emulator construction'))
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    output = tmp_path / 'absent' / 'run.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err
    assert not output.parent.exists()


def test_actor_average_rejects_old_native_before_construction(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(kindle, 'VectorAgent', SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *_, **__: pytest.fail('unexpected emulator construction'))
    output = tmp_path / 'run.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', '--output', str(output), '--evaluate',
                                    '--restore', 'unused', '--actor-posterior-samples', '8'])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert 'native runtime does not support actor posterior averaging' in capsys.readouterr().err


@pytest.mark.parametrize('samples', [None, 2, 8, 16])
def test_actor_average_runner_metadata_and_frozen_lifecycle(frozen_run, samples):
    path, rows = frozen_run(actor_samples=samples)
    assert rows[0].get('actor_posterior_samples') == samples
    assert rows[0].get('posterior_mode', 'sample') == 'sample'
    result = audit(path)
    assert result['updates'] == 0


@pytest.mark.parametrize('change,message', [
    ({'actor_posterior_samples': True}, 'invalid actor posterior sample count'),
    ({'actor_posterior_samples': 8.0}, 'invalid actor posterior sample count'),
    ({'actor_posterior_samples': 1}, 'invalid actor posterior sample count'),
    ({'actor_posterior_samples': 17}, 'invalid actor posterior sample count'),
    ({'mode': 'train'}, 'requires frozen restore'),
    ({'restored_checkpoint': None}, 'requires frozen restore'),
    ({'posterior_mode': 'greedy'}, 'sampled live posteriors'),
    ({'action_temperature': 1.0}, 'no action temperature'),
])
def test_actor_average_audit_rejects_ambiguous_headers(frozen_run, change, message):
    path, rows = frozen_run(actor_samples=8)
    rows[0].update(change)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=message):
        audit(path)
