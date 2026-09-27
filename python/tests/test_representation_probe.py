import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kindle._representation_probe import CLIP_LENGTH, SPLITS, clip_targets, fit_pca, fixed_projection, positions, spatial_features, target_names

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from collect_atari_probes import collect
from extract_atari_probe_features import token_batch


def test_split_seeds_are_disjoint():
    seeds = [seed for group in SPLITS.values() for seed in group]
    assert len(seeds) == len(set(seeds))


def test_probe_constructor_rejects_invalid_batch_before_loading_or_gpu():
    from kindle import _native
    with pytest.raises(ValueError, match="num_streams"):
        _native.LeVJepaPerception("absent.safetensors", num_streams=0)


def test_ram_coordinates_absence_and_no_uint8_underflow():
    ram = np.zeros(128, dtype=np.uint8)
    assert np.isnan(positions("Breakout", ram)).all()
    ram[[49, 54, 51, 50]] = [100, 80, 100, 120]
    np.testing.assert_array_equal(positions("Pong", ram), [51, 66, 87, 105])
    ram[[99, 101, 72]] = [100, 80, 120]
    np.testing.assert_array_equal(positions("Breakout", ram), [51, 89, 73])
    ram[[70, 97, 30, 36]] = [70, 50, 40, 4]
    sea = positions("Seaquest", ram)
    np.testing.assert_array_equal(sea[:3], [70, 83, 40])
    assert np.isnan(sea[3:]).all()
    ram[36], ram[105] = 2, 7
    assert np.isnan(positions("Seaquest", ram)).all()


def test_velocity_is_last_action_displacement_and_masks_respawns():
    ram = np.zeros((CLIP_LENGTH, 128), dtype=np.uint8)
    ram[:, [49, 54, 51, 50]] = [100, 80, 100, 120]
    ram[-1, [49, 54, 51, 50]] = [104, 72, 105, 30]
    targets = clip_targets("Pong", ram, [4] * CLIP_LENGTH)
    np.testing.assert_array_equal(targets[4:7], [1, -2, 1.25])
    assert np.isnan(targets[-1])
    ram[-1, 49] = 150
    assert np.isnan(clip_targets("Pong", ram, [4] * CLIP_LENGTH)[4])
    with pytest.raises(ValueError):
        clip_targets("Pong", ram[:-1], [4] * 15)
    with pytest.raises(ValueError):
        clip_targets("Pong", ram, [0] * CLIP_LENGTH)
    assert len(target_names("Seaquest")) == 12


@pytest.mark.parametrize("batch", [(), (3,)])
def test_pooling_matches_explicit_spatial_reference(batch):
    rng = np.random.default_rng(92)
    tokens = rng.normal(size=(*batch, 14, 14, 9)).astype(np.float32)
    for pooling, channels in [("mean", 64), ("space_to_depth", 16)]:
        projection = rng.normal(size=(9, channels)).astype(np.float32)
        patches = tokens @ projection
        expected = np.empty((*batch, 7, 7, 64), dtype=np.float32)
        for y in range(7):
            for x in range(7):
                corners = patches[..., 2*y:2*y+2, 2*x:2*x+2, :]
                expected[..., y, x, :] = (corners.mean(axis=(-3, -2)) if pooling == "mean"
                                          else corners.reshape(*batch, 64))
        np.testing.assert_allclose(spatial_features(tokens, projection, pooling=pooling), expected,
                                   rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        spatial_features(tokens, np.eye(9), pooling="bad")


def test_fixed_projection_and_pca():
    projection = fixed_projection(192)
    assert projection.shape == (192, 64)
    assert set(projection.flatten()) == {-.125, .125}
    np.testing.assert_array_equal(projection, fixed_projection(192))
    rng = np.random.default_rng(22)
    train = rng.normal(size=(200, 9)) * np.arange(1, 10)
    center, axes, eigenvalues = fit_pca(train, 5)
    np.testing.assert_allclose(center, train.mean(0), atol=1e-6)
    np.testing.assert_allclose(axes.T @ axes, np.eye(5), atol=1e-6)
    assert (np.diff(eigenvalues) <= 0).all()
    assert np.linalg.norm((train-center) @ axes) <= np.linalg.norm(train-center)
    with pytest.raises(ValueError):
        fit_pca(train, 12)


def test_phase_contrast_uses_same_final_frame_with_correct_causal_prefix():
    class Encoder:
        patch_token_shape = (3, 14, 14, 4)

        def __init__(self):
            self.tokens = np.zeros(self.patch_token_shape, dtype="<f4")
            self.arrivals = []

        def encode_batch(self, streams, frames, resets):
            self.arrivals.append((streams, [int(f[0, 0, 0]) for f in frames], resets))
            for stream, frame in zip(streams, frames):
                self.tokens[stream].fill(frame[0, 0, 0])
            values = spatial_features(self.tokens, fixed_projection(4), pooling="mean")
            return values[streams].reshape(len(streams), -1).tolist()

        def patch_tokens(self):
            return self.tokens.tobytes()

    clips = np.broadcast_to(np.arange(16, dtype=np.uint8)[None, :, None, None, None], (2, 16, 2, 2, 3))
    encoder = Encoder()
    parity = []
    first = token_batch(encoder, clips, "native", phase=15, parity=parity)
    assert len(encoder.arrivals) == 16
    assert encoder.arrivals[0] == ([0, 1], [0, 0], [True, True])
    assert encoder.arrivals[-1] == ([0, 1], [15, 15], [False, False])
    second = token_batch(encoder, clips, "native", phase=0, parity=parity)
    assert encoder.arrivals[-1] == ([0, 1], [15, 15], [True, True])
    np.testing.assert_array_equal(first, second)
    assert parity == [0, 0]


def test_collection_never_crosses_reset_and_does_not_select_by_labels():
    class Environment:
        action_space = SimpleNamespace(n=18)
        executed_action_frames = 0
        tick = 0
        episode = 0

        def __init__(self):
            self.unwrapped = SimpleNamespace(ale=SimpleNamespace(getRAM=lambda: np.zeros(128, dtype=np.uint8)))

        def reset(self, **kwargs):
            self.tick = 0
            self.episode += 1

        def step(self, action):
            self.tick += 1
            self.executed_action_frames += 4
            terminal = self.episode == 1 and self.tick == 3
            return np.full((2, 2, 3), self.episode, dtype=np.uint8), 0, terminal, False, {}

    arrays, stats = collect(Environment(), "Breakout", 19, 2)
    assert stats["actions"] == 35 and stats["discarded_tail_actions"] == 3
    assert stats["completed_episodes"] == 1
    assert arrays["rgb"].shape == (2, 16, 2, 2, 3)
    assert (arrays["rgb"] == 2).all()
    assert np.isnan(arrays["targets"]).all()
    assert (arrays["executed_frames"] == 4).all()
