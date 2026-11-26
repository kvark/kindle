import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kindle._representation_probe import CLIP_LENGTH, SPLITS, clip_targets, fit_pca, fixed_projection, positions, regression_metrics, ridge_probe, spatial_features, target_names

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from collect_atari_probes import collect
from extract_atari_probe_features import token_batch
from audit_atari_probes import visible_targets
import fit_atari_probes
from prepare_atari_pixel_probes import pixel_features
import train_atari_reconstruction


def test_split_seeds_are_disjoint():
    seeds = [seed for group in SPLITS.values() for seed in group]
    assert len(seeds) == len(set(seeds))


def test_pixel_controls_preserve_order_and_current_frame():
    frames = np.zeros((2, 16, 8, 8, 3), dtype=np.uint8)
    frames[:, -2] = 51
    frames[:, -1] = 204
    features = pixel_features(frames)
    assert features["rgb56/single_frame"].shape == (2, 9408)
    assert features["rgb56/two_frames"].shape == (2, 18816)
    np.testing.assert_allclose(features["rgb56/single_frame"], .8)
    np.testing.assert_allclose(features["rgb56/two_frames"][:, :9408], .2)
    np.testing.assert_array_equal(features["rgb56/two_frames"][:, 9408:], features["rgb56/single_frame"])


def test_fit_reuse_requires_identical_shapes_dtypes_and_all_splits():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    key = fit_atari_probes.feature_identity(a, a, a)
    assert key == fit_atari_probes.feature_identity(a.copy(), a.copy(), a.copy())
    assert key != fit_atari_probes.feature_identity(a, a, a+1)
    assert key != fit_atari_probes.feature_identity(a, a.reshape(4, 3), a)
    assert key != fit_atari_probes.feature_identity(a.astype(np.float64), a, a)


def test_reconstruction_loader_only_reads_declared_split_and_arrivals(tmp_path):
    from atari import sha256_file
    frames = np.zeros((2, 16, 210, 160, 3), dtype=np.uint8)
    for index in range(16):
        frames[:, index] = index
    path = tmp_path / "train.npz"
    np.savez_compressed(path, rgb=frames)
    manifest = dict(clips_per_seed=2, files=[dict(split="train", file=path.name, sha256=sha256_file(path)),
                                            dict(split="test", file="absent-test.npz", sha256="unused")])
    loaded = train_atari_reconstruction.load_frames(tmp_path, manifest, "train", [3, 7, 11, 15])
    assert loaded.shape == (8, 210, 160, 3)
    np.testing.assert_array_equal(loaded[:, 0, 0, 0], [3, 7, 11, 15, 3, 7, 11, 15])


def test_reconstruction_checkpoint_uses_validation_only(tmp_path, monkeypatch):
    class Model:
        step = 0
        saved = []

        def process(self, frames, *, learning_rate=None):
            if learning_rate is not None:
                assert set(frames) <= {0, 1}
                self.step += 1
                return 1
            assert frames == [3]
            return {0: 2, 512: 1, 1024: 3}[self.step]

        def save(self, path):
            self.saved.append((Path(path).name, self.step))

    model = Model()
    monkeypatch.setattr(train_atari_reconstruction, "checked_memory", lambda _: {})
    result = train_atari_reconstruction.train(model, np.array([0, 1]), np.array([3]), tmp_path,
                                              steps=1024, seed=1, batch=1)
    assert result["selected_step"] == 512 and model.step == 1024
    assert model.saved[-1] == ("encoder.safetensors", 512)


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


def test_ridge_matches_independent_primal_solve_with_missing_targets():
    rng = np.random.default_rng(902)
    x, val, test = [rng.normal(size=(n, 5)) for n in (23, 9, 11)]
    weight = rng.normal(size=(5, 2))
    y, vy = x @ weight + 3, val @ weight + 3
    y[::4, 1] = np.nan
    predicted, selection = ridge_probe(x, y, val, vy, test, alphas=(0.1,))
    for column in range(2):
        mask = np.isfinite(y[:, column])
        tx = x[mask]
        mean, scale = tx.mean(0), tx.std(0)*np.sqrt(5)
        tx = (tx-mean)/scale
        target = y[mask, column]
        weights = np.linalg.solve(tx.T @ tx + 0.1*np.eye(5), tx.T @ (target-target.mean()))
        expected = (test-mean)/scale @ weights + target.mean()
        np.testing.assert_allclose(predicted[:, column], expected, atol=1e-10)
        assert selection[column]["train_count"] == mask.sum()


def test_regression_metrics_keep_missing_and_constant_targets_explicit():
    target = np.array([[1, 3, np.nan], [2, 3, np.nan], [3, 3, np.nan]], dtype=float)
    prediction = np.array([[1, 3, 0], [2, 3, 0], [3, 3, 0]], dtype=float)
    metrics = regression_metrics(prediction, target)
    assert metrics[0]["r2"] == 1
    assert metrics[1] == dict(count=3, r2=None, mae=0, rmse=0)
    assert metrics[2] == dict(count=0, r2=None, mae=None, rmse=None)


@pytest.mark.parametrize("interval", [32, 128])
@pytest.mark.parametrize("standardize", [False, True])
def test_mlp_checkpoint_selection_uses_validation_not_test(monkeypatch, interval, standardize):
    class Model:
        gpu_device = dict(device_name="NVIDIA GeForce RTX 5080", driver_info="580.178.04")
        gpu_memory_budget = dict(budget_bytes=4 << 30, usage_bytes=0)
        step = 0
        learns = 0

        def learn(self, x, y, mask, **kwargs):
            assert set(np.frombuffer(x, dtype="<f4")) <= ({-1, 1} if standardize else {0, 1})
            assert set(np.frombuffer(y, dtype="<f4")) <= {-1, 1}
            self.step += 1
            self.learns += 1
            return 1.0

        def predict(self, x):
            return [float(self.step)] * 64

        def parameters(self):
            return [[float(self.step)]]

        def set_parameters(self, values):
            self.step = values[0][0]

    model = Model()
    monkeypatch.setattr(fit_atari_probes._native, "RegressionProbe", lambda *a, **kw: model, raising=False)
    prediction, info = fit_atari_probes.mlp_probe(
        np.array([[0.0], [1.0]]), np.array([[-1.0], [1.0]]),
        np.array([[2.0]]), np.array([[float(interval)]]), np.array([[100.0]]), 92,
        steps=2 * interval, validation_interval=interval, standardize_inputs=standardize)
    assert info["selected_step"] == interval and model.learns == 2 * interval
    assert info["input_normalization"] == ("training_mean_std" if standardize else "identity")
    np.testing.assert_array_equal(prediction, [[interval]])


def test_mlp_reuses_device_but_resets_every_fit():
    class Model:
        gpu_device = dict(device_name="NVIDIA GeForce RTX 5080", driver_info="580.178.04")
        gpu_memory_budget = dict(budget_bytes=4 << 30, usage_bytes=0)
        resets = []

        def reset(self, inputs, targets, seed):
            self.resets.append((inputs, targets, seed))
            self.value = 0.

        def learn(self, *args, **kwargs):
            self.value += 1
            return 1.

        def predict(self, x):
            return [self.value]*64

        def parameters(self):
            return [[self.value]]

        def set_parameters(self, values):
            self.value = values[0][0]

    model = Model()
    data = np.array([[0.], [1.]])
    first, _ = fit_atari_probes.mlp_probe(data, data, data, data, data, 1009, steps=32, model=model)
    second, _ = fit_atari_probes.mlp_probe(data, data, data, data, data, 2017, steps=32, model=model)
    assert model.resets == [(1, 1, 1009), (1, 1, 2017)]
    np.testing.assert_array_equal(first, second)


def test_mlp_constant_training_columns_do_not_gain_false_variance():
    training = np.full((1024, 4), .8, dtype=np.float32)
    training[:, 0] = np.arange(1024) % 2
    validation = np.array([[1., .5, 1.1, .8]], dtype=np.float32)
    train, valid = fit_atari_probes.standardized_features(training, validation)
    np.testing.assert_array_equal(train[:, 1:], 0.)
    np.testing.assert_allclose(train[:, 0], 2*training[:, 0]-1)
    np.testing.assert_allclose(valid, [[1., -.3, .3, 0.]], atol=1e-7)
    assert train.dtype == valid.dtype == np.float32


def test_fixed_pixel_range_does_not_amplify_rare_training_variation():
    training = np.full((768, 2), -.5, dtype=np.float32)
    training[0, 0] += 1 / 255
    held_out = np.array([[.1, .2]], dtype=np.float32)
    _, whitened = fit_atari_probes.standardized_features(training, held_out)
    assert whitened[0, 0] > 4000
    actual = fit_atari_probes.standardized_features(training, held_out, standardize=False)
    for original, normalized in zip((training, held_out), actual):
        np.testing.assert_array_equal(normalized, original)
        assert normalized.dtype == np.float32


def test_secondary_visible_metrics_require_current_and_previous_sprite_for_velocity():
    frames = np.zeros((16, 210, 160, 3), dtype=np.uint8)
    ram = np.zeros((16, 128), dtype=np.uint8)
    ram[:, [49, 54, 51, 50]] = [100, 80, 100, 120]
    frames[-1, 66:70, 51:53] = (236, 236, 236)
    masks = visible_targets("Pong", frames, ram)
    np.testing.assert_array_equal(masks, [True, True, False, False, False, False, False, False])
    frames[-2] = frames[-1]
    np.testing.assert_array_equal(visible_targets("Pong", frames, ram)[4:6], [True, True])


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
