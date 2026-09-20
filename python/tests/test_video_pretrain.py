from types import SimpleNamespace
from pathlib import Path
import json
import sys

import gymnasium as gym
import numpy as np
import pytest

from kindle import _video_pretrain as video


def recording(seed=7):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(256, size=(80, 24, 24, 3), dtype=np.uint8)
    episodes = np.repeat(np.array([0, 1], dtype=np.uint32), 40)
    return video.Recording(pixels, episodes), episodes


def test_clips_never_cross_episode_boundary():
    record, episodes = recording()
    rng = np.random.default_rng(19)
    seen = set()
    for _ in range(1000):
        clip, start = record.sample(rng)
        indices = start + np.arange(16) * 2
        assert len(np.unique(episodes[indices])) == 1
        np.testing.assert_array_equal(clip, record.frames[indices])
        seen.add(start)
    assert seen == set(range(10)) | set(range(40, 50))


def test_short_and_nonmonotonic_recordings_are_refused():
    pixels = np.zeros((32, 24, 24, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="no complete"):
        video.Recording(pixels, np.arange(32, dtype=np.uint32))
    with pytest.raises(ValueError, match="monotonic"):
        video.Recording(pixels, np.arange(32, dtype=np.uint32)[::-1])


def test_patch_ids_select_original_frame_and_channel_major_pixels():
    clip = np.zeros((16, 96, 96, 3), dtype=np.uint8)
    for frame in range(16):
        clip[frame, :, :, 0] = frame * 10
    clip[..., 1] = np.arange(96)[None, :, None]
    clip[..., 2] = np.arange(96)[None, None, :]
    ids = np.array([15 * 36 + 35, 0, 3 * 36 + 2 * 6 + 1], dtype=np.uint32)
    patches = video.retained_patches(clip, 96, (0, 0, 96, 96), ids).reshape(3, 3, 16, 16)
    reconstructed = np.rint((patches * video.STD + video.MEAN) * 255).astype(np.uint8)
    for index, token in enumerate(ids):
        frame, patch = divmod(int(token), 36)
        y, x = divmod(patch, 6)
        expected = clip[frame, y * 16:(y + 1) * 16, x * 16:(x + 1) * 16].transpose(2, 0, 1)
        np.testing.assert_array_equal(reconstructed[index], expected)


def test_batches_reproduce_from_step_and_keep_each_view_token_major():
    train, _ = recording()
    held_out, _ = recording(11)
    corpus = SimpleNamespace(recordings={"train": [train], "validation": [held_out]})
    model = dict(batch=3, local_views=2, projector_output=17, directions=7)
    first, examples = video.batch(corpus, model, 99, 7)
    repeated, repeated_examples = video.batch(corpus, model, 99, 7)
    changed, _ = video.batch(corpus, model, 99, 8)
    validation, _ = video.batch(corpus, model, 99, 7, "validation")
    assert examples == repeated_examples
    for name, value in first.items():
        np.testing.assert_array_equal(value, repeated[name])
        assert value.flags.c_contiguous
        assert np.isfinite(value).all()
        assert not np.array_equal(value, changed[name])
        assert not np.array_equal(value, validation[name])
    assert first["global_patches"].shape == (157, 3, 768)
    assert first["local_patches"].shape == (29, 6, 768)
    for name in ("global", "local"):
        for column in first[name + "_ids"].T:
            assert len(np.unique(column)) == len(column)
    np.testing.assert_allclose(np.linalg.norm(first["directions"], axis=0), 1, atol=1e-6)


def test_crops_are_bounded_and_ids_reject_duplicates():
    rng = np.random.default_rng(4)
    for _ in range(100):
        left, top, right, bottom = video.crop_box(64, 64, (0.1, 0.4), rng)
        assert 0 <= left < right <= 64 and 0 <= top < bottom <= 64
    with pytest.raises(ValueError, match="token IDs"):
        video.retained_patches(np.zeros((16, 64, 64, 3), dtype=np.uint8),
                               224, (0, 0, 64, 64), np.array([0, 0], dtype=np.uint32))


def test_recording_counts_frames_and_corpus_refuses_leakage_and_tampering(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from collect_pretraining import collect_recording
    from atari import DreamerAtariPreprocessing

    class Environment(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (16, 16, 3), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def get_action_meanings(self):
            return ["NOOP", "FIRE"]

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros((16, 16, 3), dtype=np.uint8), {}

        def step(self, action):
            self.steps += 1
            pixels = self.np_random.integers(256, size=(16, 16, 3), dtype=np.uint8)
            return pixels, 99.0, self.steps == 160, False, {}

    records = []
    for seed, split in [(91, "train"), (92, "validation")]:
        environment = DreamerAtariPreprocessing(Environment(), noop_max=0, max_episode_frames=100000)
        record = collect_recording(environment, tmp_path, split, seed, 80, split)
        environment.close()
        assert record["executed_emulator_frames"] == 320
        assert record["reset_noop_frames"] == 0
        assert record["emulator_resets"] == 2
        assert record["terminated_episodes"] == 2
        assert record["truncated_episodes"] == 0
        assert record["shape"] == [80, 64, 64, 3]
        assert "reward" not in json.dumps(record)
        episodes = np.load(tmp_path / record["episodes"]["file"])
        np.testing.assert_array_equal(episodes, np.repeat([0, 1], 40))
        records.append(record)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps(dict(format=1, recordings=records)))
    corpus = video.Corpus(manifest)
    assert len(corpus.recordings["train"]) == len(corpus.recordings["validation"]) == 1
    duplicate = dict(records[0], split="validation")
    manifest.write_text(json.dumps(dict(format=1, recordings=[records[0], duplicate])))
    with pytest.raises(ValueError, match="duplicate"):
        video.Corpus(manifest)
    manifest.write_text(json.dumps(dict(format=1, recordings=records)))
    artifact = tmp_path / records[0]["frames"]["file"]
    with artifact.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        video.Corpus(manifest)
