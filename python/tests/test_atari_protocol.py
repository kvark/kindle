import sys
import hashlib
from pathlib import Path

import pytest
import gymnasium as gym
import numpy as np

from kindle._atari_scores import ATARI_PROFILES


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from atari import ATARI_PROTOCOLS  # noqa: E402
import atari  # noqa: E402


def test_checkpoint_identity_fingerprints_actual_files(tmp_path) -> None:
    names = ("metadata.json", "world.safetensors", "behavior.safetensors", "slow_value.safetensors")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    identity = atari.checkpoint_identity(tmp_path)
    assert identity["path"] == str(tmp_path.resolve())
    assert identity["metadata_sha256"] == hashlib.sha256(b"metadata.json").hexdigest()
    assert identity["tensor_sha256"] == {
        Path(name).stem: hashlib.sha256(name.encode()).hexdigest() for name in names[1:]
    }
    (tmp_path / "world.safetensors").write_bytes(b"changed weights")
    assert atari.checkpoint_identity(tmp_path)["tensor_sha256"]["world"] != identity["tensor_sha256"]["world"]


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


@pytest.mark.parametrize("terminal_at, frame_limit, expected_frames", [(2, 100, 2), (100, 5, 5)])
def test_exact_frame_counts_survive_resets(terminal_at, frame_limit, expected_frames) -> None:
    class Environment(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def get_action_meanings(self):
            return ["NOOP", "FIRE"]

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros((8, 8, 3), dtype=np.uint8), {}

        def step(self, action):
            self.steps += 1
            return (np.zeros((8, 8, 3), dtype=np.uint8), 0.0,
                    self.steps >= terminal_at, False, {})

    environment = atari.DreamerAtariPreprocessing(
        Environment(), noop_max=0, max_episode_frames=frame_limit)
    environment.reset(seed=0)
    ended = False
    while not ended:
        _, _, terminal, truncated, _ = environment.step(1)
        ended = terminal or truncated
    assert environment.executed_action_frames == expected_frames
    assert environment.reset_noop_frames == 0
    environment.reset()
    assert environment.executed_action_frames == expected_frames
    assert environment.emulator_resets == 2


@pytest.mark.parametrize("screen_size", [None, 64])
def test_native_rgb_preserves_fine_detail_and_rgb64_remains_an_explicit_control(screen_size):
    class Environment(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (210, 160, 3), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def get_action_meanings(self):
            return ["NOOP", "FIRE"]

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            checker = (np.indices((210, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)
            self.frame = np.repeat(checker[..., None], 3, axis=2)
            return self.frame.copy(), {}

        def step(self, action):
            self.steps += 1
            self.frame = np.zeros((210, 160, 3), dtype=np.uint8)
            self.frame[30, 40 + self.steps] = 255
            return self.frame.copy(), float(action), False, False, {}

    raw = Environment()
    env = atari.DreamerAtariPreprocessing(raw, noop_max=0, screen_size=screen_size)
    image, _ = env.reset(seed=123)
    if screen_size is None:
        assert image.shape == env.observation_space.shape == (210, 160, 3)
        np.testing.assert_array_equal(image, raw.frame)
        assert set(np.unique(image)) == {0, 255}
    else:
        assert image.shape == env.observation_space.shape == (64, 64, 3)
        expected = atari.Image.fromarray(raw.frame).resize((64, 64), atari.Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(image, expected)
        assert image.min() > 0 and image.max() < 255
    image, reward, terminal, cutoff, _ = env.step(1)
    expected = np.zeros((210, 160, 3), dtype=np.uint8)
    expected[30, 43:45] = 255
    if screen_size is not None:
        expected = np.asarray(atari.Image.fromarray(expected).resize((64, 64), atari.Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(image, expected)
    assert image.flags.c_contiguous and image.dtype == np.uint8
    assert reward == 4 and not terminal and not cutoff and env.executed_action_frames == 4


@pytest.mark.parametrize("screen_size", [0, -1, True, 64.0, "native"])
def test_invalid_resize_is_rejected_before_environment_use(screen_size):
    with pytest.raises(ValueError, match="screen_size"):
        atari.DreamerAtariPreprocessing(gym.Env(), screen_size=screen_size)


@pytest.mark.parametrize("game", ["Pong", "Boxing", "Freeway", "Breakout", "Qbert"])
@pytest.mark.parametrize("sticky_actions", [0.0, 0.25])
def test_native_and_rgb64_real_ale_share_interactions_not_pixel_loss(game, sticky_actions):
    gym.register_envs(atari.ale_py)
    environments = [atari.DreamerAtariPreprocessing(gym.make(
        f"ALE/{game}-v5", frameskip=1, repeat_action_probability=sticky_actions,
        full_action_space=True), noop_max=0, max_episode_frames=100000, screen_size=size)
        for size in (None, 64)]

    def compare(frames):
        native, rgb64 = frames
        assert native.shape == (210, 160, 3) and rgb64.shape == (64, 64, 3)
        resized = atari.Image.fromarray(native).resize((64, 64), atari.Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(resized, rgb64)

    try:
        compare([env.reset(seed=17029)[0] for env in environments])
        for action in np.random.default_rng(1783).integers(0, 18, size=512):
            results = [env.step(int(action)) for env in environments]
            compare([row[0] for row in results])
            assert results[0][1:4] == results[1][1:4]
            assert environments[0].executed_action_frames == environments[1].executed_action_frames
            if results[0][2] or results[0][3]:
                compare([env.reset()[0] for env in environments])
    finally:
        for env in environments:
            env.close()
