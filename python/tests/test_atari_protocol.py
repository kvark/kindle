import sys
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from atari import ATARI_PROTOCOLS  # noqa: E402


def test_published_minimal_changes_only_action_vocabulary() -> None:
    published = ATARI_PROTOCOLS["published"]
    minimal = ATARI_PROTOCOLS["published-minimal"]

    assert published.full_action_space is True
    assert minimal.full_action_space is False
    assert minimal.noop_max == published.noop_max == 0
    assert minimal.max_episode_frames == published.max_episode_frames == 100_000
