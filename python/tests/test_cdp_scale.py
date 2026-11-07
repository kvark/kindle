import copy
import json
import sys
from types import SimpleNamespace

import pytest

import atari_vector
import kindle


@pytest.mark.parametrize("args,message", [
    *[(["--cdp", "--cdp-scale", value], "finite and positive in F32")
      for value in ("0", "-1", "nan", "inf", "1e300", "1e-300")],
    (["--cdp-scale", "1000"], "requires fresh CDP"),
    (["--cdp", "--encoder-checkpoint", "unused", "--cdp-scale", "1000"], "requires fresh CDP"),
    (["--restore", "unused", "--cdp-scale", "1000"], "training overrides"),
    (["--restore", "unused", "--cdp-scale=1000"], "training overrides"),
    (["--cdp", "--evaluate", "--cdp-scale", "1000"], "requires fresh CDP"),
    (["--cdp", "--evaluate", "--greedy", "--cdp-scale", "1000"], "requires fresh CDP"),
])
def test_cdp_scale_rejects_invalid_runs(monkeypatch, tmp_path, capsys, args, message):
    output = tmp_path / "never.jsonl"
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(output), *args])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("centered", [False, True])
def test_cdp_scale_changes_only_requested_loss(monkeypatch, tmp_path, centered):
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    configs = []

    def construct(streams, config):
        configs.append(copy.deepcopy(config))
        raise RuntimeError("config checked before GPU")

    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(learned_rgb=construct))
    for scale in (None, 500, 1000):
        monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / f"cdp-{scale}.jsonl"),
            "--cdp", *(["--cdp-centered"] if centered else []),
            *(["--cdp-scale", str(scale)] if scale is not None else [])])
        with pytest.raises(RuntimeError, match="config checked before GPU"):
            atari_vector.main()
    default, explicit_default, candidate = configs
    assert default["loss_scales"]["future_prediction"] == 500
    assert explicit_default == default
    assert candidate["loss_scales"]["future_prediction"] == 1000
    candidate["loss_scales"]["future_prediction"] = 500
    assert candidate == default


def test_cdp_scale_restore_uses_checkpoint_value(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = kindle.default_config(18)
    config.update(observation_kind="rgb64", cdp_centered=True)
    config["loss_scales"].update(reconstruction=0, future_prediction=1000)
    (checkpoint / "metadata.json").write_text(json.dumps(dict(config=config, action_count=18)))
    environment = SimpleNamespace(action_space=SimpleNamespace(n=18),
                                  reset=lambda **_: (None, {}), close=lambda: None)
    monkeypatch.setattr(atari_vector.gym, "make", lambda *_, **__: environment)
    monkeypatch.setattr(atari_vector, "DreamerAtariPreprocessing", lambda env, **_: env)
    monkeypatch.setattr(atari_vector, "checkpoint_identity", lambda _: {})

    def restore(path, streams):
        assert path == str(checkpoint) and streams == 4
        assert json.loads((checkpoint / "metadata.json").read_text())["config"] == config
        raise RuntimeError("restore checked before GPU")

    monkeypatch.setattr(kindle, "VectorAgent", SimpleNamespace(restore_rgb=restore))
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "restore.jsonl"),
                                     "--restore", str(checkpoint), "--observation-size", "native"])
    with pytest.raises(RuntimeError, match="restore checked before GPU"):
        atari_vector.main()


def test_cdp_scale_checks_effective_open_loop_weight(monkeypatch, tmp_path, capsys):
    if "--open-loop-scale" not in atari_vector.main.__code__.co_consts:
        pytest.skip("retained private runner predates optional open-loop supervision")
    monkeypatch.setattr(sys, "argv", ["atari_vector.py", "--output", str(tmp_path / "never.jsonl"),
                                     "--cdp", "--cdp-scale", "1e38",
                                     "--open-loop-length", "3", "--open-loop-scale", "4"])
    with pytest.raises(SystemExit) as error:
        atari_vector.main()
    assert error.value.code == 2
    assert "positive finite scaled F32 loss" in capsys.readouterr().err
