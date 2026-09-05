import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import run_upstream_control as control  # noqa: E402


def test_upstream_accepts_only_the_declared_config_patch(tmp_path, monkeypatch) -> None:
    original = "defaults:\n  env:\n    atari100k: {clip_reward: False}\n"
    directory = tmp_path / "dreamerv3"
    directory.mkdir()
    config = directory / "configs.yaml"
    config.write_text(control.configured_source(original))

    def git(source, *args):
        return {
            ("rev-parse", "HEAD"): control.REVISION + "\n",
            ("diff", "--name-only", "HEAD"): "dreamerv3/configs.yaml\n",
            ("show", f"{control.REVISION}:dreamerv3/configs.yaml"): original,
            ("diff", "HEAD", "--", "dreamerv3/configs.yaml"): "declared diff",
        }[args]

    monkeypatch.setattr(control, "git", git)
    assert control.validate_source(tmp_path) == "declared diff"
    config.write_text(control.configured_source(original).replace("noops: 0", "noops: 30"))
    with pytest.raises(ValueError, match="PUBLISHED_CONFIG"):
        control.validate_source(tmp_path)


def test_upstream_rejects_a_different_revision(monkeypatch) -> None:
    monkeypatch.setattr(control, "git", lambda *args: "wrong revision")
    with pytest.raises(ValueError, match="upstream must be at"):
        control.validate_source(Path("/unused"))
