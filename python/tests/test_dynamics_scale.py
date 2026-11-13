import copy
import sys
from types import SimpleNamespace

import pytest

import atari_vector
import kindle


@pytest.mark.parametrize("args,message", [
    *[(["--dynamics-scale", value], "finite and positive in F32")
      for value in ("0", "-1", "nan", "inf", "1e300", "1e-300")],
    (["--restore", "unused", "--dynamics-scale", "3"], "training overrides"),
    (["--restore", "unused", "--dynamics-scale=3"], "training overrides"),
    (["--evaluate", "--dynamics-scale", "3"], "requires fresh training"),
    (["--evaluate", "--greedy", "--dynamics-scale", "3"], "requires fresh training"),
])
def test_dynamics_override_rejects_invalid_runs(monkeypatch, tmp_path, capsys, args, message):
    output = tmp_path / "never.jsonl"
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("cdp", [False, True])
def test_dynamics_override_changes_only_requested_loss(monkeypatch, tmp_path, cdp):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError("config checked before GPU")

    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(learned_rgb=construct))
    for scale in (None, 1., 3.):
        monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / f"dynamics-{scale}.jsonl"),
            *(["--cdp", "--cdp-centered"] if cdp else []),
            *(["--dynamics-scale", str(scale)] if scale is not None else [])])
        with pytest.raises(RuntimeError, match="config checked before GPU"):
            atari_vector.main()
    default, explicit_default, modified = configs
    assert default["loss_scales"]["dynamics"] == kindle.default_config(18)["loss_scales"]["dynamics"] == 1.
    assert explicit_default == default
    assert modified["loss_scales"]["dynamics"] == 3.
    modified["loss_scales"]["dynamics"] = 1.
    assert modified == default
