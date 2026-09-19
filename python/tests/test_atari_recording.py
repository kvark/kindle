import hashlib
import io
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import atari  # noqa: E402
import record_atari  # noqa: E402


class Environment(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)
    closed = False

    def get_action_meanings(self):
        return ["NOOP", "FIRE"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        frame = self.np_random.integers(256, size=(8, 8, 3), dtype=np.uint8)
        return frame, float(action), self.steps == 5, False, {"step": self.steps}

    def close(self):
        self.closed = True


class Encoder:
    def __init__(self, *args, **kwargs):
        self.stdin = io.BytesIO()
        self.waits = 0

    def wait(self):
        self.waits += 1
        return 0


@pytest.mark.parametrize("noop_max", [0, 3])
def test_recording_preserves_transitions_and_rng(monkeypatch, noop_max):
    monkeypatch.setattr(record_atari.subprocess, "Popen", Encoder)
    recorded = record_atari.AtariVideo(Environment(), Path("unused.mp4"), "ffmpeg")
    plain = atari.DreamerAtariPreprocessing(Environment(), noop_max=noop_max)
    video = atari.DreamerAtariPreprocessing(recorded, noop_max=noop_max)
    a, a_info = plain.reset(seed=17)
    b, b_info = video.reset(seed=17)
    np.testing.assert_array_equal(a, b)
    assert a_info == b_info
    for action in (0, 1, 1, 0, 1):
        a, b = plain.step(action), video.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        if a[2] or a[3]:
            a, a_info = plain.reset()
            b, b_info = video.reset()
            np.testing.assert_array_equal(a, b)
            assert a_info == b_info
    assert recorded.frame_count == video.executed_action_frames + video.reset_noop_frames
    assert plain.executed_action_frames == video.executed_action_frames
    assert plain.reset_noop_frames == video.reset_noop_frames
    assert plain.emulator_resets == video.emulator_resets
    pixels = recorded._encoder.stdin.getvalue()
    assert len(pixels) == recorded.frame_count * 8 * 8 * 3
    assert recorded.frame_sha256.hexdigest() == hashlib.sha256(pixels).hexdigest()
    video.close()
    video.close()
    assert recorded._encoder.waits == 1


def test_recording_refuses_training(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["record_atari.py", "unused",
                                     "--video", str(tmp_path / "video.mp4"),
                                     "--output", str(tmp_path / "run.jsonl")])
    with pytest.raises(SystemExit) as error:
        record_atari.main()
    assert error.value.code == 2
    assert "record only frozen evaluation" in capsys.readouterr().err


@pytest.mark.parametrize("existing", ["video.mp4", "run.jsonl", "video.mp4.json"])
def test_recording_preserves_existing_artifacts(monkeypatch, tmp_path, capsys, existing):
    artifact = tmp_path / existing
    artifact.write_bytes(b"existing data")
    monkeypatch.setattr(sys, "argv", ["record_atari.py", "unused", "--random-policy",
                                     "--video", str(tmp_path / "video.mp4"),
                                     "--output", str(tmp_path / "run.jsonl")])
    with pytest.raises(SystemExit) as error:
        record_atari.main()
    assert error.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert artifact.read_bytes() == b"existing data"


def test_encoder_failure_is_reported_and_closes_environment(monkeypatch):
    monkeypatch.setattr(record_atari.subprocess, "Popen", Encoder)
    monkeypatch.setattr(Encoder, "wait", lambda self: 7)
    environment = Environment()
    recorder = record_atari.AtariVideo(environment, Path("unused.mp4"), "ffmpeg")
    with pytest.raises(RuntimeError, match="ffmpeg failed with exit status 7"):
        recorder.close()
    assert environment.closed
    assert recorder._encoder.stdin.closed
    recorder.close()


@pytest.mark.parametrize("failure", ["encoder_startup", "runner"])
def test_failure_restores_runner_globals_and_closes_environment(monkeypatch, tmp_path, failure):
    environment = Environment()
    original_make = lambda *args, **kwargs: environment
    argv = ["record_atari.py", "unused", "--random-policy",
            "--video", str(tmp_path / "video.mp4"),
            "--output", str(tmp_path / "run.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(atari.gym, "make", original_make)
    monkeypatch.setattr(record_atari.shutil, "which", lambda executable: "ffmpeg")

    def failing_encoder(*args, **kwargs):
        raise RuntimeError("synthetic encoder failure")

    def failing_runner():
        atari.gym.make("synthetic")
        raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(record_atari.subprocess, "Popen",
                        failing_encoder if failure == "encoder_startup" else Encoder)
    monkeypatch.setattr(atari, "main", failing_runner)
    with pytest.raises(RuntimeError, match="synthetic"):
        record_atari.main()
    assert environment.closed
    assert atari.gym.make is original_make
    assert sys.argv is argv
    assert not (tmp_path / "video.mp4.json").exists()
