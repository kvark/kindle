import copy
import sys
from types import SimpleNamespace

import pytest

import atari_vector
import kindle


@pytest.mark.parametrize("args,message", [
    *[(["--latent-unimix=" + value], "representable in F32")
      for value in ("-0.1", "1", "2", "nan", "inf", "-inf", "1e-50", "0.99999999")],
    (["--restore", "unused", "--latent-unimix", ".001"], "training overrides"),
    (["--restore", "unused", "--latent-unimix=.001"], "training overrides"),
    (["--evaluate", "--latent-unimix", ".001"], "requires fresh training"),
    (["--evaluate", "--greedy", "--latent-unimix", ".001"], "requires fresh training"),
])
def test_latent_override_rejects_invalid_runs(monkeypatch, tmp_path, capsys, args, message):
    output = tmp_path / "never.jsonl"
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("cdp", [False, True])
def test_latent_override_preserves_other_config(monkeypatch, tmp_path, cdp):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError("config checked before GPU")

    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(learned_rgb=construct))
    for unimix in (None, .01, .001, 0, .999):
        monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / f"latent-{unimix}.jsonl"),
            *(["--cdp", "--cdp-centered"] if cdp else []),
            *(["--latent-unimix", str(unimix)] if unimix is not None else [])])
        with pytest.raises(RuntimeError, match="config checked before GPU"):
            atari_vector.main()
    default, explicit_default, *modified = configs
    assert default["unimix"] == kindle.default_config(18)["unimix"] == .01
    assert explicit_default == default
    for config, unimix in zip(modified, (.001, 0, .999)):
        assert config["unimix"] == unimix
        config["unimix"] = .01
        assert config == default
