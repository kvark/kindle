import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import probe_fixed_latents as probe
import fit_fixed_latents as fit


class Environment:
    action_space = SimpleNamespace(n=3)

    def __init__(self):
        self.tick = 0
        self.unwrapped = SimpleNamespace(ale=SimpleNamespace(getRAM=self.ram))

    def ram(self):
        value = np.zeros(128, dtype=np.uint8)
        value[70] = self.tick
        return value

    def reset(self, *, seed=None):
        if seed is not None:
            self.tick = 0
        return self.tick, {}

    def step(self, action):
        assert 0 <= action < 3
        self.tick += 1
        return self.tick, float(self.tick == 2), self.tick in (3, 7), self.tick == 5, {}


class Agent:
    learner_step = 1987
    environment_step = 8192
    active = False

    def begin_episode(self, frame):
        assert not self.active, "reset requires an episode boundary"
        self.active = True
        self.visual_observation = [frame, -frame]

    def act(self, *, action_mask):
        return action_mask.index(True)

    def observe(self, frame, **kwargs):
        assert self.active
        self.environment_step += 1
        self.visual_observation = [frame, -frame]
        self.active = not (kwargs["terminated"] or kwargs["truncated"])


def test_collection_keeps_terminal_targets_and_separate_reset_arrivals():
    actor, checks = Agent(), []
    data = probe.collect_trace(actor, Environment(), 123, 8, lambda: checks.append(True))
    assert actor.learner_step == 1987 and actor.environment_step == 8200
    np.testing.assert_array_equal(data["episodes"], [0, 0, 0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(data["features"][data["following"], 0], np.arange(1, 9))
    np.testing.assert_array_equal(data["positions"][data["following"], 0], np.arange(1, 9))
    np.testing.assert_array_equal(data["phase"][data["current"]], [0, 1, 2, 0, 1, 0, 1, 0])
    assert len(data["features"]) == 12 and len(checks) == 2
    assert data["collection_cut"].tolist() == [False] * 7 + [True]
    assert not data["truncated"][-1] and not data["terminated"][-1] and not actor.active
    rng = random.Random(123 ^ probe.ACTION_SEED_XOR)
    assert data["actions"].tolist() == [rng.randrange(3) for _ in range(8)]
    for i in range(7):
        assert (data["following"][i] == data["current"][i + 1]) == (data["episodes"][i] == data["episodes"][i + 1])


def test_chunk_wrap_does_not_reset_episode_or_drop_actions():
    actor = Agent()
    data = probe.collect_trace(actor, Environment(), 123, 26, lambda: None)
    assert data["episodes"][-1] == 3
    np.testing.assert_array_equal(data["phase"][data["following"]][7:], np.arange(1, 20) % 16)
    assert len(data["actions"]) == 26


@pytest.mark.parametrize("failure", ["action", "updates", "nonfinite"])
def test_collection_rejects_invalid_actor(failure):
    actor = Agent()
    if failure == "action":
        actor.act = lambda **_: -1
    else:
        original = actor.observe

        def observe(*args, **kwargs):
            original(*args, **kwargs)
            if failure == "updates":
                actor.learner_step += 1
            else:
                actor.visual_observation = [float("nan"), 0]

        actor.observe = observe
    with pytest.raises(RuntimeError):
        probe.collect_trace(actor, Environment(), 123, 8, lambda: None)


def test_trajectory_split_is_disjoint_and_test_has_three_seeds():
    seeds = [seed for group in probe.SPLITS.values() for seed in group]
    assert len(seeds) == len(set(seeds)) and len(probe.SPLITS["test"]) == 3


def test_recording_cutoff_is_not_an_environment_truncation():
    actor, env = Agent(), Environment()
    first = probe.collect_trace(actor, env, 123, 2, lambda: None)
    second = probe.collect_trace(actor, env, 456, 2, lambda: None)
    for data in (first, second):
        assert not data["truncated"].any() and not data["terminated"].any()
        assert data["collection_cut"].tolist() == [False, True]
    assert actor.environment_step == 8196 and actor.learner_step == 1987


@pytest.mark.parametrize("change", [None, "value", "shape", "dtype", "key", "signed_zero"])
def test_freeze_check_compares_tensor_bytes_not_file_serialization(tmp_path, change):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    for name in ("world", "behavior", "slow_value"):
        a = dict(weight=np.array([0., 1.], dtype=np.float32))
        b = {key: value.copy() for key, value in a.items()}
        if name == "world":
            if change == "value":
                b["weight"][1] += 1
            elif change == "signed_zero":
                b["weight"][0] = -0.
            elif change == "shape":
                b["weight"] = b["weight"].reshape(1, 2)
            elif change == "dtype":
                b["weight"] = b["weight"].astype(np.float64)
            elif change == "key":
                b["bias"] = b.pop("weight")
        save_file(a, before / f"{name}.safetensors", metadata={"serialization": "before"})
        save_file(b, after / f"{name}.safetensors", metadata={"serialization": "after"})
    assert (before / "world.safetensors").read_bytes() != (after / "world.safetensors").read_bytes()
    if change is None:
        assert probe.assert_frozen_tensors(before, after) == dict(world=1, behavior=1, slow_value=1)
    else:
        with pytest.raises(RuntimeError, match="frozen collection changed"):
            probe.assert_frozen_tensors(before, after)


def synthetic_trace(count=39):
    arrival = np.arange(count + 1)
    return dict(features=np.column_stack((arrival, -arrival)).astype(np.float32),
                positions=np.tile(arrival[:, None], (1, 6)).astype(np.float32),
                current=np.arange(count), following=np.arange(1, count + 1),
                episodes=np.zeros(count, int), actions=np.arange(count) % 18,
                phase=(arrival % 16).astype(np.uint8), rewards=(np.arange(count) % 7 == 0).astype(float),
                terminated=np.arange(count) == count - 1)


def test_fixed_dynamics_inputs_are_causal_and_retain_terminal_targets():
    data = synthetic_trace()
    rows = fit.examples(data, 15)
    origins = np.arange(0, 25, 4)
    np.testing.assert_array_equal(rows["origins"], origins)
    np.testing.assert_array_equal(rows["x"][:, :2], data["features"][np.maximum(origins - 1, 0)])
    np.testing.assert_array_equal(rows["x"][:, 2:4], data["features"][origins])
    assert rows["x"].shape == (7, 2 * 2 + 15 * 18 + 16)
    np.testing.assert_array_equal(rows["y"], np.tile([15, -15], (7, 1)))
    np.testing.assert_array_equal(rows["crossing"], [False, True, True, True, False, True, True])
    assert rows["labels"][-1, -1] == 1
    data["positions"][:] = 999
    data["rewards"][:] = 123
    changed = fit.examples(data, 15)
    np.testing.assert_array_equal(rows["x"], changed["x"])
    np.testing.assert_array_equal(rows["y"], changed["y"])


def test_fixed_future_windows_never_cross_resets():
    data = probe.collect_trace(Agent(), Environment(), 123, 40, lambda: None)
    rows = fit.examples(data, 15)
    assert rows["origins"].tolist() == [8, 12, 16, 20, 24]
    assert np.all(data["episodes"][rows["origins"]] == data["episodes"][rows["origins"] + 14])
    state = fit.examples(data, 0)
    assert len(state["x"]) == 40
    np.testing.assert_array_equal(state["labels"][:, -1], data["terminated"])


def test_unrelated_action_control_preserves_features_and_phase():
    rows = fit.examples(synthetic_trace(), 15)
    control = fit.unrelated_actions(rows["x"], 15, 2)
    np.testing.assert_array_equal(control[:, :4], rows["x"][:, :4])
    np.testing.assert_array_equal(control[:, -16:], rows["x"][:, -16:])
    np.testing.assert_array_equal(control[:, 4:-16].reshape(-1, 15, 18).sum(-1), 1)
    assert not np.array_equal(control[:, 4:-16], rows["x"][:, 4:-16])
    np.testing.assert_array_equal(control, fit.unrelated_actions(rows["x"], 15, 2))


def test_fixed_normalization_and_error_metrics_use_all_dimensions():
    x = np.array([[1., 5.], [3., 5.]], np.float32)
    y = np.array([[2., np.nan], [4., 6.], [6., 8.]], np.float32)
    norm = fit.normalization(x, y)
    np.testing.assert_array_equal(norm["x_mean"], [2, 5])
    np.testing.assert_array_equal(norm["x_scale"], [1, 1])
    np.testing.assert_array_equal(norm["y_mean"], [4, 7])
    expected = np.zeros((4, 3136), np.float32)
    prediction = expected.copy()
    prediction[:, -1] = 2
    error = fit.latent_errors(prediction, expected, np.full(3136, 2.))
    np.testing.assert_allclose(error, np.tile([4 / 3136, 1 / 3136], (4, 1)))


def test_latent_uncertainty_resamples_whole_trajectories():
    errors = np.array([[1., 2.], [1., 2.], [7., 8.], [7., 8.]])
    data = dict(seeds=np.array([10, 10, 20, 20]), crossing=np.array([False, True, False, True]))
    result = fit.latent_report(errors, data)
    assert result["trajectory_bootstrap_95"] == [[1., 7.], [2., 8.]]
    assert result["all"] == dict(count=4, raw_mse=4., normalized_mse=5.)


def test_fixed_fit_pipeline_preserves_splits_and_reuses_frozen_readout(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = dict(protocol=probe.PROTOCOL, status="complete", tensor_bytes_unchanged=True,
                    learner_updates=0, steps_per_trajectory=39, files=[])
    for split, seeds in probe.SPLITS.items():
        for seed in seeds:
            data = synthetic_trace()
            data["features"] = np.tile(data["features"], (1, 1568))
            data["truncated"] = np.zeros(39, bool)
            data["collection_cut"] = np.arange(39) == 38
            path = corpus / f"{split}-{seed}.npz"
            np.savez(path, **data)
            manifest["files"].append(dict(file=path.name, split=split, seed=seed,
                                          sha256=probe.sha256_file(path), actions=39, arrivals=40))
    (corpus / "manifest.json").write_text(json.dumps(manifest))

    class Model:
        gpu_device = dict(device_name="NVIDIA GeForce RTX 5080", driver_info="580.178.04")
        gpu_memory_budget = dict(budget_bytes=4 << 30, usage_bytes=0)
        dimensions = []
        updates = 0

        def __init__(self, inputs, targets, **kwargs):
            self.reset(inputs, targets, 0)

        def reset(self, inputs, targets, seed):
            self.dimensions.append((inputs, targets))
            self.targets = targets

        def predict(self, _):
            return [0.] * (64 * self.targets)

        def learn(self, *args, **kwargs):
            self.updates += 1
            return 1.

        def parameters(self):
            return [[0.]]

        def set_parameters(self, parameters):
            assert parameters == [[0.]]

    model = Model(3136, 8)
    monkeypatch.setattr(fit._native, "RegressionProbe", lambda *args, **kwargs: model)
    fit.fit(corpus, tmp_path / "fits", steps=2)
    result = json.loads((tmp_path / "fits/result.json").read_text())
    assert result["status"] == "complete" and model.updates == 6
    assert [row["horizon"] for row in result["heads"]] == [0, 1, 15]
    assert model.dimensions == [(3136, 8), (3136, 8), (6306, 3136), (3136, 8), (6558, 3136), (3136, 8)]
    for row in result["heads"]:
        assert row["train_examples"] == row["test_examples"]
        with np.load(tmp_path / "fits" / row["evidence_file"]) as evidence:
            assert set(evidence["trajectory_seed"]) == set(probe.SPLITS["test"])
        if row["horizon"]:
            assert set(row["decoded"]) == {"real_future", "prediction", "persistence", "unrelated_actions"}
    # Tampered data must fail before any native construction or fitting.
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed corpus file"):
        fit.fit(corpus, tmp_path / "tampered", steps=2)
    assert model.updates == 6
