import json
import sys
from pathlib import Path
from types import SimpleNamespace

import kindle
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari_vector


@pytest.mark.parametrize("weight", [None, .5, 1, 10])
def test_terminal_weight_reaches_native_without_changing_sampler(monkeypatch, tmp_path, weight):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    def construct(streams, config):
        assert config.get("terminal_continuation_weight") == weight
        assert config["loss_scales"]["continuation"] == 1
        assert config.get("terminal_replay_fraction") is None
        assert config.get("open_loop_supervision") is None
        if weight is None:
            assert "terminal_continuation_weight" not in config
        raise RuntimeError("checked before GPU")
    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(
        learned_rgb=construct, SUPPORTS_TERMINAL_CONTINUATION_WEIGHT=True))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "run.jsonl"),
        "--cdp", *(["--terminal-continuation-weight", str(weight)] if weight is not None else [])])
    with pytest.raises(RuntimeError, match="checked before GPU"):
        atari_vector.main()


@pytest.mark.parametrize("extra,message", [
    *[(["--cdp", "--terminal-continuation-weight", value], "finite and positive in F32")
      for value in ["0", "-1", "nan", "inf", "1e39", "1e-50"]],
    (["--terminal-continuation-weight", "10"], "requires fresh CDP"),
    (["--cdp", "--terminal-continuation-weight", "10", "--evaluate"], "requires fresh CDP"),
    (["--cdp", "--terminal-continuation-weight", "10", "--restore", "unused"], "requires fresh CDP"),
    (["--cdp", "--terminal-continuation-weight", "10", "--open-loop-length", "3"], "requires fresh CDP"),
    (["--cdp", "--terminal-continuation-weight", "10", "--open-loop-scale", "1"], "requires fresh CDP"),
    (["--cdp", "--terminal-continuation-weight", "1e38", "--continuation-scale", "1e38"], "effective terminal-continuation"),
])
def test_invalid_terminal_weight_stops_before_environment(monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: pytest.fail("environment constructed"))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"), *extra])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("restore", [False, True])
def test_terminal_weight_rejects_old_native_before_environment(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: pytest.fail("environment constructed"))
    if restore:
        (tmp_path / "metadata.json").write_text(json.dumps({"config": {"terminal_continuation_weight": 10}}))
    args = (["--restore", str(tmp_path), "--observation-size", "native"] if restore
            else ["--cdp", "--terminal-continuation-weight", "10"])
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"), *args])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert "requires native terminal-weight support" in capsys.readouterr().err
