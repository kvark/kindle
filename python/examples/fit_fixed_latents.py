"""Fit small GPU readout/dynamics heads on frozen Tiny trajectories.

No actor learns here. RAM labels are readout targets only. Train/validation/test
are whole disjoint trajectories; validation alone selects each head checkpoint.
Run under gpu_host_guard.py. Full latent dimensions are retained.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from atari import sha256_file
from fit_atari_probes import bytes32, checked_memory, mlp_probe
from kindle import _native
from kindle._representation_probe import regression_metrics, target_names
from kindle._reward_probe import roc_auc
from probe_fixed_latents import PROTOCOL, SPLITS


HEAD_SEED = 20261003
NAMES = (*target_names("Seaquest")[:6], "reward_event", "terminal")


def examples(data, horizon):
    """Outcome readouts, or causal residual prediction with stride-four origins."""
    current, following, episodes = (data[k] for k in ("current", "following", "episodes"))
    features = data["features"]
    labels = np.column_stack((data["positions"][following], data["rewards"] > 0, data["terminated"]))
    if horizon == 0:
        origins = np.arange(len(current))
        return dict(x=features[following], y=labels, labels=labels, origins=origins,
                    crossing=np.zeros(len(origins), bool))
    origins = np.arange(0, len(current) - horizon + 1, 4)
    origins = origins[episodes[origins] == episodes[origins + horizon - 1]]
    end = origins + horizon - 1
    previous = np.maximum(origins - 1, 0)
    previous = np.where(episodes[previous] == episodes[origins], previous, origins)
    actions = data["actions"][origins[:, None] + np.arange(horizon)]
    phase = data["phase"][current[origins]]
    x = np.concatenate((features[current[previous]], features[current[origins]],
                        np.eye(18, dtype=np.float32)[actions].reshape(len(origins), -1),
                        np.eye(16, dtype=np.float32)[phase]), axis=1)
    return dict(x=x, y=features[following[end]] - features[current[origins]],
                labels=labels[end], origins=origins, crossing=phase + horizon >= 16)


def load_split(root, manifest, split, horizon):
    parts, seeds = [], []
    for row in manifest["files"]:
        if row["split"] != split:
            continue
        with np.load(root / row["file"]) as data:
            part = examples(data, horizon)
        parts.append(part)
        seeds.extend([row["seed"]] * len(part["x"]))
    return {**{key: np.concatenate([p[key] for p in parts]) for key in parts[0]},
            "seeds": np.asarray(seeds)}


def normalization(x, y):
    mean, scale = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
    target_mean = np.nanmean(y, 0, dtype=np.float64)
    target_scale = np.nanstd(y, 0, dtype=np.float64)
    return dict(x_mean=mean, x_scale=np.where(scale > 1e-6, scale, 1.), y_mean=target_mean,
                y_scale=np.where(target_scale > 1e-6, target_scale, 1.))


def predict(model, x, norm):
    result = []
    for offset in range(0, len(x), 64):
        part = x[offset:offset + 64]
        padded = np.zeros((64, x.shape[1]), np.float32)
        padded[:len(part)] = (part - norm["x_mean"]) / norm["x_scale"]
        values = np.asarray(model.predict(bytes32(padded)), np.float32).reshape(64, -1)
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite fixed-head prediction")
        result.append(values[:len(part)] * norm["y_scale"] + norm["y_mean"])
    checked_memory(model)
    return np.concatenate(result)


def state_metrics(prediction, target):
    result = dict(zip(NAMES, regression_metrics(prediction, target)))
    for i in (6, 7):
        result[NAMES[i]].update(positive_count=int(target[:, i].sum()),
                               auc=roc_auc(target[:, i].astype(bool).tolist(), prediction[:, i].tolist()))
    return result


def state_report(prediction, data):
    return dict(all=state_metrics(prediction, data["labels"]),
                by_trajectory={str(seed): state_metrics(prediction[data["seeds"] == seed],
                    data["labels"][data["seeds"] == seed]) for seed in np.unique(data["seeds"])})


def latent_errors(prediction, target, scale):
    delta = prediction.astype(np.float64) - target
    return np.column_stack((np.square(delta).mean(1), np.square(delta / scale).mean(1)))


def latent_report(errors, data):
    def summary(rows):
        if not len(rows):
            return dict(count=0, raw_mse=None, normalized_mse=None)
        return dict(count=len(rows), raw_mse=float(rows[:, 0].mean()),
                    normalized_mse=float(rows[:, 1].mean()))

    by_trajectory = {str(seed): summary(errors[data["seeds"] == seed]) for seed in np.unique(data["seeds"])}
    # Resample whole trajectories, never correlated individual frames.
    means = np.array([[row["raw_mse"], row["normalized_mse"]] for row in by_trajectory.values()])
    draws = np.random.default_rng(HEAD_SEED).integers(len(means), size=(10000, len(means)))
    interval = np.quantile(means[draws].mean(1), [.025, .975], axis=0)
    return dict(all=summary(errors), by_trajectory=by_trajectory,
                equal_trajectory_mean=means.mean(0).tolist(),
                trajectory_bootstrap_95=interval.T.tolist(),
                by_chunk_crossing={str(flag): summary(errors[data["crossing"] == flag]) for flag in (False, True)})


def unrelated_actions(x, horizon, width):
    result = x.copy()
    rng = np.random.default_rng(HEAD_SEED ^ horizon)
    actions = rng.integers(18, size=(len(x), horizon))
    result[:, 2 * width:2 * width + 18 * horizon] = np.eye(18, dtype=np.float32)[actions].reshape(len(x), -1)
    return result


def validate_corpus(root, manifest):
    if (manifest["protocol"] != PROTOCOL or manifest["status"] != "complete"
            or not manifest["tensor_bytes_unchanged"] or manifest["learner_updates"] != 0):
        raise ValueError("complete bitwise-frozen corpus required")
    expected = {(split, seed) for split, seeds in SPLITS.items() for seed in seeds}
    if {(row["split"], row["seed"]) for row in manifest["files"]} != expected or len(manifest["files"]) != 8:
        raise ValueError("whole declared disjoint trajectory splits required")
    for row in manifest["files"]:
        if sha256_file(root / row["file"]) != row["sha256"]:
            raise ValueError(f"changed corpus file: {row['file']}")
        with np.load(root / row["file"]) as data:
            current, following, episodes = (data[k] for k in ("current", "following", "episodes"))
            if len(current) != manifest["steps_per_trajectory"] or len(current) != row["actions"]:
                raise ValueError("incomplete action trace")
            if data["features"].shape != (row["arrivals"], 3136) or not np.isfinite(data["features"]).all():
                raise ValueError("invalid production latents")
            boundary = data["terminated"] | data["truncated"]
            if (not np.array_equal(episodes[1:] - episodes[:-1], boundary[:-1].astype(int))
                    or not np.array_equal(following[:-1] == current[1:], ~boundary[:-1])
                    or not np.array_equal(data["collection_cut"], np.arange(len(current)) == len(current) - 1)
                    or np.any(data["actions"] < 0) or np.any(data["actions"] >= 18)
                    or not np.array_equal(data["phase"][following], (data["phase"][current] + 1) % 16)):
                raise ValueError("inconsistent trajectory boundaries/actions/encoder phase")


def fit(root, output, steps):
    manifest = json.loads((root / "manifest.json").read_text())
    validate_corpus(root, manifest)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(protocol="kindle-fixed-latent-fits-v1", status="running", heads=[], head_seed=HEAD_SEED,
                  corpus_manifest_sha256=sha256_file(root / "manifest.json"), corpus=str(root.resolve()),
                  source_sha256=sha256_file(__file__), helper_sha256=sha256_file(Path(__file__).with_name("fit_atari_probes.py")),
                  native_sha256=sha256_file(_native.__file__), steps=steps,
                  normalization="training-only F64 statistics; std<=1e-6 uses scale1; F32 GPU inputs",
                  limitations=["offline frozen-target diagnostic, not RL or RSSM forecasts",
                               "one fixed head seed; only three held-out trajectories per encoder",
                               "RAM positions omit bullets and do not establish control sufficiency",
                               "residual predictors know proposed actions, not hidden sticky-action execution"])
    model, readout_parameters, readout_norm = None, None, None
    for horizon in (0, 1, 15):
        train, val, test = [load_split(root, manifest, split, horizon) for split in SPLITS]
        norm = normalization(train["x"], train["y"])
        if model is None:
            model = _native.RegressionProbe(train["x"].shape[1], train["y"].shape[1], hidden=128, batch=64)
        prediction, info = mlp_probe(train["x"], train["y"], val["x"], val["y"], test["x"], HEAD_SEED,
                                     steps=steps, model=model, validation_interval=128)
        parameters = model.parameters()
        training_prediction = predict(model, train["x"], norm)
        info["training_normalized_mse"] = float(np.nanmean(np.nanmean(
            np.square((training_prediction - train["y"]) / norm["y_scale"]), axis=0)))
        del training_prediction
        head_file = output / f"head-h{horizon}.npz"
        np.savez(head_file, **norm, **{f"parameter_{i}": np.asarray(p, np.float32) for i, p in enumerate(parameters)})
        row = dict(horizon=horizon, fit=info, train_examples=len(train["x"]), validation_examples=len(val["x"]),
                   test_examples=len(test["x"]), head_file=head_file.name, head_sha256=sha256_file(head_file))
        evidence = dict(trajectory_seed=test["seeds"], origin=test["origins"], chunk_crossing=test["crossing"],
                        labels=test["labels"])
        if horizon == 0:
            readout_parameters, readout_norm = parameters, norm
            row["readout"] = state_report(prediction, test)
            row["constant"] = state_report(np.broadcast_to(norm["y_mean"], prediction.shape), test)
            evidence["readout"] = prediction
        else:
            width = test["y"].shape[1]
            unrelated = predict(model, unrelated_actions(test["x"], horizon, width), norm)
            errors = {"prediction": latent_errors(prediction, test["y"], norm["y_scale"]),
                      "persistence": latent_errors(np.zeros_like(test["y"]), test["y"], norm["y_scale"]),
                      "unrelated_actions": latent_errors(unrelated, test["y"], norm["y_scale"])}
            row["latents"] = {name: latent_report(error, test) for name, error in errors.items()}
            evidence.update({f"error_{name}": error for name, error in errors.items()})
            model.reset(width, len(NAMES), HEAD_SEED)
            model.set_parameters(readout_parameters)
            current = test["x"][:, width:2 * width]
            row["decoded"] = {}
            for name, features in (("real_future", current + test["y"]), ("prediction", current + prediction),
                                   ("persistence", current), ("unrelated_actions", current + unrelated)):
                decoded = predict(model, features, readout_norm)
                row["decoded"][name] = state_report(decoded, test)
                evidence[f"decoded_{name}"] = decoded
            if model.parameters() != readout_parameters:
                raise RuntimeError("frozen readout changed during evaluation")
        evidence_file = output / f"evidence-h{horizon}.npz"
        np.savez(evidence_file, **evidence)
        row.update(evidence_file=evidence_file.name, evidence_sha256=sha256_file(evidence_file),
                   seconds=time.monotonic() - started)
        result["heads"].append(row)
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(horizon=horizon, selected_step=info["selected_step"], seconds=row["seconds"])), flush=True)
        del train, val, test, prediction
    result.update(status="complete", seconds=time.monotonic() - started, gpu_device=model.gpu_device)
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=2048)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("positive step budget required")
    fit(args.corpus, args.output, args.steps)


if __name__ == "__main__":
    main()
