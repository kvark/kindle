import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import atari_vector
import kindle


@pytest.mark.parametrize("args,message", [
    *[(["--discount-horizon", value], "between 2 and 33554430")
      for value in ("0", "1", "-1", "33554431", "33554432", "18446744073709551616")],
    *[(["--discount-horizon", value], "invalid int value") for value in ("1.5", "nan")],
    (["--restore", "unused", "--discount-horizon", "100"], "training overrides"),
    (["--restore", "unused", "--discount-horizon=100"], "training overrides"),
    (["--evaluate", "--discount-horizon", "100"], "requires fresh training"),
    (["--evaluate", "--greedy", "--discount-horizon", "100"], "requires fresh training"),
])
def test_discount_override_rejects_invalid_runs(monkeypatch, tmp_path, capsys, args, message):
    output = tmp_path / "never.jsonl"
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("cdp", [False, True])
def test_discount_override_preserves_rollout_and_other_config(monkeypatch, tmp_path, cdp):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError("config checked before GPU")

    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(learned_rgb=construct))
    for horizon in (None, 333, 100, 2, 33554430):
        monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / f"discount-{horizon}.jsonl"),
            *(["--cdp", "--cdp-centered"] if cdp else []),
            *(["--discount-horizon", str(horizon)] if horizon is not None else [])])
        with pytest.raises(RuntimeError, match="config checked before GPU"):
            atari_vector.main()
    default, explicit_default, *modified = configs
    assert default["horizon"] == kindle.default_config(18)["horizon"] == 333
    assert explicit_default == default
    for config, horizon in zip(modified, (100, 2, 33554430)):
        assert config["horizon"] == horizon
        gamma = np.float32(1) - np.float32(1) / np.float32(horizon)
        assert 0 < gamma < 1
        config["horizon"] = 333
        assert config == default
