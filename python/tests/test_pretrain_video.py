"""Adapter lifecycle only; fake trainer deliberately contains no learning."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import kindle
from kindle import _video_pretrain as video


@pytest.fixture
def runner(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "pretrain_video.py"
    spec = importlib.util.spec_from_file_location("pretrain_video", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(model=dict(batch=2, local_views=1, projector_hidden=8,
                             projector_output=7, directions=3),
                  seed=7, steps=3, warmup_steps=1, learning_rate=1e-4,
                  weight_decay=0.04, ema_decay=0.99)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    records = []
    for index, split in enumerate(("train", "validation")):
        rng = np.random.default_rng(index)
        arrays = dict(frames=rng.integers(256, size=(40, 16, 16, 3), dtype=np.uint8),
                      episodes=np.zeros(40, dtype=np.uint32))
        record = dict(split=split, observations=40)
        for kind, array in arrays.items():
            path = tmp_path / f"{split}-{kind}.npy"
            np.save(path, array)
            record[kind] = dict(file=path.name, sha256=video.sha256_file(path))
        records.append(record)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(dict(format=1, recordings=records)))
    called = []

    class Trainer:
        gpu_device = dict(device_name="test", driver_info="test-driver", is_software_emulated=False)

        def __init__(self, text):
            called.append("construct")
            self.config, self.completed_steps = json.loads(text), 0

        @classmethod
        def restore(cls, path):
            saved = json.loads((Path(path) / "fake.json").read_text())
            instance = cls(json.dumps(saved["config"]))
            instance.completed_steps = saved["step"]
            return instance

        def step(self, **arrays):
            assert arrays["global_patches"].shape == (157, 2, 768)
            self.completed_steps += 1
            return dict(step=self.completed_steps, loss=0.0)

        def save(self, path):
            destination = Path(path)
            destination.mkdir()
            (destination / "fake.json").write_text(json.dumps(dict(
                step=self.completed_steps, config=self.config)))

        def export_encoder(self, path):
            Path(path).write_bytes(b"not model weights")

    monkeypatch.setitem(kindle.__dict__, "_native", SimpleNamespace(
        __file__=str(source), LeVJepaTrainer=Trainer))
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test-only")

    def run(output, *extra):
        monkeypatch.setattr(sys, "argv", [str(source), str(config_path), str(corpus), str(output),
            "--checkpoint-every", "2", "--expected-device", "test", "--expected-driver", "test-driver", *extra])
        module.main()

    return run, called


def test_adapter_exact_budget_checkpoints_and_deterministic_resume(runner, tmp_path):
    run, called = runner
    output = tmp_path / "first"
    run(output)
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert sorted(path.name for path in output.glob("step*")) == [
        "step000000", "step000002", "step000003", "steps.jsonl"]
    assert json.loads((output / "result.json").read_text())["complete"]
    restored = tmp_path / "restored"
    run(restored, "--restore", str(output / "step000002"))
    resumed = json.loads((restored / "steps.jsonl").read_text())
    assert resumed["step"] == 3 and resumed["examples"] == rows[2]["examples"]
    assert called == ["construct", "construct"]
    with pytest.raises(FileExistsError):
        run(output)
    assert called == ["construct", "construct"]


def test_restore_changed_identity_refuses_before_native_constructor(runner, tmp_path):
    run, called = runner
    checkpoint = tmp_path / "changed"
    checkpoint.mkdir()
    (checkpoint / "adapter.json").write_text("{}")
    with pytest.raises(ValueError, match="restore corpus"):
        run(tmp_path / "unused", "--restore", str(checkpoint))
    assert not called and not (tmp_path / "unused").exists()
