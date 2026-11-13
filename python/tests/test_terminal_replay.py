import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import kindle
from kindle._vector_audit import audit_terminal_replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari_vector


@pytest.mark.parametrize("fraction", [None, 0, .25, 1])
def test_terminal_replay_config_reaches_native(monkeypatch, tmp_path, fraction):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    def construct(streams, config):
        assert config.get("terminal_replay_fraction") == fraction
        assert config["loss_scales"]["continuation"] == 1
        assert config["batch_size"] == config["batch_length"] == 16
        raise RuntimeError("checked before GPU")
    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(
        learned_rgb=construct, SUPPORTS_TERMINAL_REPLAY=True))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "run.jsonl"),
        "--batch-length", "16", *(["--terminal-replay-fraction", str(fraction)] if fraction is not None else [])])
    with pytest.raises(RuntimeError, match="checked before GPU"):
        atari_vector.main()


@pytest.mark.parametrize("args,message", [
    *[(["--terminal-replay-fraction", value], "finite and in [0, 1]")
      for value in ["-0.01", "1.01", "nan", "inf"]],
    (["--terminal-replay-fraction", ".25", "--restore", "unused"], "requires fresh non-video training"),
    (["--terminal-replay-fraction=.25", "--evaluate"], "requires fresh non-video training"),
    (["--terminal-replay-fraction", ".25", "--encoder-training", "joint"], "requires fresh non-video training"),
])
def test_terminal_replay_rejects_invalid_cli_before_environment(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: pytest.fail("environment constructed"))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"), *args])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("restore", [False, True])
def test_terminal_replay_rejects_old_native(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: pytest.fail("environment constructed"))
    if restore:
        (tmp_path / "metadata.json").write_text(json.dumps({"config": {"terminal_replay_fraction": .25}}))
    args = (["--restore", str(tmp_path), "--observation-size", "native"] if restore
            else ["--terminal-replay-fraction", ".25"])
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"), *args])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert "requires native terminal-replay support" in capsys.readouterr().err


def fixture():
    config = dict(terminal_replay_fraction=.25, batch_size=16, batch_length=16, horizon=333)
    sampling = dict(fresh_sequences=2, uniform_sequences=10, terminal_sequences=4,
                    eligible_terminal_sequences=37, terminal_rows=5)
    report = dict(replay_sampling=sampling,
                  world=dict(replay_continuation_target_mean=(1-1/333)*(1-5/256)))
    return config, report


def test_terminal_report_accepts_routes_and_targets():
    config, report = fixture()
    audit_terminal_replay(config, report)
    audit_terminal_replay({}, {})
    report = copy.deepcopy(report)
    report["replay_sampling"]["uniform_sequences"] += 4
    report["replay_sampling"]["terminal_sequences"] = 0
    config["terminal_replay_fraction"] = 0
    audit_terminal_replay(config, report)


@pytest.mark.parametrize("key,value", [
    ("fresh_sequences", 3), ("uniform_sequences", -1), ("terminal_sequences", True),
    ("eligible_terminal_sequences", 0), ("terminal_rows", 3), ("terminal_rows", 257),
    ("terminal_rows", 6),
])
def test_terminal_report_rejects_corrupt_counts(key, value):
    config, report = fixture()
    report["replay_sampling"][key] = value
    with pytest.raises(ValueError):
        audit_terminal_replay(config, report)


def test_terminal_report_rejects_missing_undeclared_and_invalid_routes():
    config, report = fixture()
    for fraction in [None, 0, 1, -1, float("nan")]:
        bad = dict(config, terminal_replay_fraction=fraction)
        with pytest.raises(ValueError):
            audit_terminal_replay(bad, report)
    with pytest.raises(ValueError):
        audit_terminal_replay(config, {})
    with pytest.raises(ValueError):
        audit_terminal_replay(dict(config, video_encoder="joint"), report)
