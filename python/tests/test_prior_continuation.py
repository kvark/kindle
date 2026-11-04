import json
import sys
from pathlib import Path
from types import SimpleNamespace

import kindle
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import atari_vector


@pytest.mark.parametrize("weight", [None, .5, 1, 10])
def test_prior_continuation_reaches_native_without_changing_sampler(monkeypatch, tmp_path, weight):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    def construct(streams, config):
        assert config.get("prior_continuation_scale") == weight
        assert config["loss_scales"]["continuation"] == 1
        assert config.get("terminal_replay_fraction") is None
        assert config.get("open_loop_supervision") is None
        if weight is None:
            assert "prior_continuation_scale" not in config
        raise RuntimeError("checked before GPU")
    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(
        learned_rgb=construct, SUPPORTS_PRIOR_CONTINUATION=True))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "run.jsonl"),
        "--cdp", *(["--prior-continuation-scale", str(weight)] if weight is not None else [])])
    with pytest.raises(RuntimeError, match="checked before GPU"):
        atari_vector.main()


@pytest.mark.parametrize("extra,message", [
    *[(["--cdp", "--prior-continuation-scale", value], "finite and positive in F32")
      for value in ["0", "-1", "nan", "inf", "1e39", "1e-50"]],
    (["--prior-continuation-scale", "10"], "requires fresh CDP"),
    (["--cdp", "--prior-continuation-scale", "10", "--evaluate"], "requires fresh CDP"),
    (["--cdp", "--prior-continuation-scale", "10", "--restore", "unused"], "requires fresh CDP"),
    (["--cdp", "--prior-continuation-scale", "10", "--open-loop-length", "3"], "requires fresh CDP"),
    (["--cdp", "--prior-continuation-scale", "10", "--open-loop-scale", "1"], "requires fresh CDP"),
    (["--cdp", "--prior-continuation-scale", "1", "--terminal-continuation-weight", "10"], "requires fresh CDP"),
])
def test_invalid_prior_continuation_stops_before_environment(monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: pytest.fail("environment constructed"))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"), *extra])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("restore", [False, True])
def test_prior_continuation_rejects_old_native_before_environment(monkeypatch, tmp_path, capsys, restore):
    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: pytest.fail("environment constructed"))
    if restore:
        (tmp_path / "metadata.json").write_text(json.dumps({"config": {"prior_continuation_scale": 10}}))
    args = (["--restore", str(tmp_path), "--observation-size", "native"] if restore
            else ["--cdp", "--prior-continuation-scale", "10"])
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"), *args])
    with pytest.raises(SystemExit):
        atari_vector.main()
    assert "requires native prior-continuation support" in capsys.readouterr().err
