import sys
from pathlib import Path

import pytest

from kindle._atari_scores import ATARI_PROFILES


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from atari import ATARI_PROTOCOLS  # noqa: E402
import atari  # noqa: E402


def test_published_minimal_changes_only_action_vocabulary() -> None:
    published = ATARI_PROTOCOLS["published"]
    minimal = ATARI_PROTOCOLS["published-minimal"]

    assert published.full_action_space is True
    assert minimal.full_action_space is False
    assert minimal.noop_max == published.noop_max == 0
    assert minimal.max_episode_frames == published.max_episode_frames == 100_000


def test_runner_and_score_protocol_metadata_agree() -> None:
    assert set(ATARI_PROTOCOLS) == set(ATARI_PROFILES)
    for name, protocol in ATARI_PROTOCOLS.items():
        assert ATARI_PROFILES[name] == {
            "action_repeat": 4,
            "full_action_space": protocol.full_action_space,
            "noop_max": protocol.noop_max,
            "max_episode_frames": protocol.max_episode_frames,
        }


@pytest.mark.parametrize("mode", [["--restore", "/unused"], ["--random-policy"]])
@pytest.mark.parametrize("option", ["--visitation-bonus", "--future-prediction-loss-scale=.25", "--agc=0"])
def test_training_overrides_are_not_silently_ignored(monkeypatch, capsys, mode, option) -> None:
    monkeypatch.setattr(sys, "argv", ["atari.py", "/unused", *mode, option])
    with pytest.raises(SystemExit) as error:
        atari.main()
    assert error.value.code == 2
    assert "training overrides require a fresh Dreamer run" in capsys.readouterr().err
