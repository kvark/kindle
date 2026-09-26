import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari_vector


def test_exact_budget_boundary():
    atari_vector.require_gpu_budget(dict(usage_bytes=1024**3, budget_bytes=3*1024**3), 2*1024**3)


@pytest.mark.parametrize("snapshot", [
    {}, {"usage_bytes": 1}, {"usage_bytes": -1, "budget_bytes": 4},
    {"usage_bytes": True, "budget_bytes": 4}, {"usage_bytes": 1.0, "budget_bytes": 4},
    {"usage_bytes": 0, "budget_bytes": 0}, {"usage_bytes": 4, "budget_bytes": 3},
    {"usage_bytes": 1024**3 + 1, "budget_bytes": 3*1024**3},
])
def test_invalid_unsupported_or_insufficient_budget(snapshot):
    with pytest.raises(ValueError):
        atari_vector.require_gpu_budget(snapshot, 2*1024**3)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_minimum_refuses_before_output(monkeypatch, tmp_path, value):
    output = tmp_path / "run.jsonl"
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "unused", "--output", str(output),
                                    "--min-gpu-budget-headroom-mib", value])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()


def test_existing_memory_report_is_preserved(monkeypatch, tmp_path):
    output = tmp_path / "run.jsonl"
    memory = output.with_suffix(".gpu-memory.jsonl")
    memory.write_text("existing evidence")
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "unused", "--output", str(output),
                                    "--min-gpu-budget-headroom-mib", "2048"])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert memory.read_text() == "existing evidence"
