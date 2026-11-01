"""Training action selection must be explicit and must not alter frozen CLI modes."""
import sys

import pytest
import atari_vector
from kindle._vector_audit import audit
from test_vector import fixture_events, write_log


@pytest.mark.parametrize('extra,message', [
    (['--training-action-mode', 'greedy', '--evaluate'], 'requires fresh training'),
    (['--training-action-mode=sample', '--restore', 'unused'], 'requires fresh training'),
    (['--training-action-mode', 'greedy', '--restore', 'unused', '--evaluate'], 'requires fresh training'),
    (['--training-action-mode', 'invalid'], 'invalid choice'),
    (['--greedy'], 'only supported for frozen evaluation'),
])
def test_invalid_training_mode_fails_before_environment(monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **kw: pytest.fail('environment created'))
    output = tmp_path / 'run.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector.py', 'ALE/Qbert-v5', '--output', str(output), *extra])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('mode', ['sample', 'greedy'])
def test_training_modes_preserve_action_and_update_accounting(tmp_path, mode):
    rows = fixture_events()
    rows[0]['training_action_mode'] = mode
    result = audit(write_log(tmp_path, rows))
    assert result['actions'] == 6 and result['updates'] == 3


@pytest.mark.parametrize('fields,message', [
    ({'training_action_mode': None}, 'unknown training action mode'),
    ({'training_action_mode': 'random'}, 'unknown training action mode'),
    ({'training_action_mode': 'greedy', 'mode': 'evaluate_greedy'}, 'requires fresh training'),
    ({'training_action_mode': 'sample', 'restored_checkpoint': {}}, 'requires fresh training'),
])
def test_audit_rejects_invalid_or_restored_training_mode(tmp_path, fields, message):
    rows = fixture_events()
    rows[0].update(fields)
    with pytest.raises(ValueError, match=message):
        audit(write_log(tmp_path, rows))
