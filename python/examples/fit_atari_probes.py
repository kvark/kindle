"""Fit offline ridge and native-GPU MLP probes, with validation-only selection.

Run MLP mode under gpu_host_guard.py. Ridge-only mode constructs no GPU context.
Every result includes all targets/variants and per-held-out-trajectory metrics;
head seeds measure fit variability, not independent RL agents or test datasets.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from atari import sha256_file
from kindle import _native
from kindle._representation_probe import PROTOCOL, regression_metrics, ridge_probe, target_names


def checked_memory(model):
    device, memory = model.gpu_device, model.gpu_memory_budget
    if device["device_name"] != "NVIDIA GeForce RTX 5080" or device["driver_info"] != "580.178.04":
        raise RuntimeError(f"unexpected probe GPU: {device}")
    if memory is None or memory["budget_bytes"] - memory["usage_bytes"] < 2 << 30:
        raise RuntimeError(f"less than 2GiB estimated headroom: {memory}")
    return memory


def bytes32(values):
    return np.asarray(values, dtype="<f4").tobytes()


def feature_identity(*arrays):
    return tuple((x.shape, x.dtype.str, hashlib.sha256(x.tobytes()).digest()) for x in arrays)


def standardized_features(training, *others):
    # Float32 axis-0 accumulation can invent >1e-6 variance in a constant
    # column. Compute training statistics in F64, then upload F32 features.
    mean, scale = training.mean(0, dtype=np.float64), training.std(0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return [np.asarray((x-mean)/scale, dtype=np.float32) for x in (training, *others)]


def mlp_probe(train_x, train_y, validation_x, validation_y, test_x, seed, *, steps=512, model=None,
              validation_interval=32):
    if steps <= 0 or validation_interval <= 0:
        raise ValueError("positive fit and validation intervals required")
    batch, hidden, interval = 64, 128, validation_interval
    xs = standardized_features(train_x, validation_x, test_x)
    mask = np.isfinite(train_y)
    y_mean, y_scale = np.nanmean(train_y, 0, dtype=np.float64), np.nanstd(train_y, 0, dtype=np.float64)
    y_scale = np.where(y_scale > 1e-6, y_scale, 1.0)
    if not np.isfinite(y_mean).all() or np.any(mask.sum(0) < 2):
        raise ValueError("insufficient MLP training targets")
    y = np.asarray(np.where(mask, (train_y-y_mean)/y_scale, 0), dtype=np.float32)
    # Equal target weighting in expectation despite different missingness.
    weights = mask.astype(np.float32) / mask.mean(0)
    if model is None:
        model = _native.RegressionProbe(xs[0].shape[1], y.shape[1], hidden=hidden, batch=batch, seed=seed)
    else:
        model.reset(xs[0].shape[1], y.shape[1], seed)
    memory = [checked_memory(model)]
    rng = np.random.default_rng(seed ^ 0xA72A)

    def predict(x):
        outputs = []
        for start in range(0, len(x), batch):
            part = x[start:start+batch]
            padded = np.zeros((batch, x.shape[1]), dtype=np.float32)
            padded[:len(part)] = part
            result = np.asarray(model.predict(bytes32(padded)), dtype=np.float32).reshape(batch, y.shape[1])
            if not np.isfinite(result).all():
                raise RuntimeError("nonfinite MLP predictions")
            outputs.append(result[:len(part)] * y_scale + y_mean)
        return np.concatenate(outputs)

    best, best_parameters, selected_step, curve = float("inf"), None, None, []
    for step in range(1, steps+1):
        rows = rng.integers(len(train_x), size=batch)
        loss = model.learn(bytes32(xs[0][rows]), bytes32(y[rows]), bytes32(weights[rows]),
                           learning_rate=1e-3, regularization=1e-4/len(train_x))
        if step % interval == 0 or step == steps:
            prediction = predict(xs[1])
            error = float(np.nanmean(np.nanmean(np.square((prediction-validation_y)/y_scale), axis=0)))
            if not np.isfinite(error):
                raise RuntimeError("nonfinite MLP validation error")
            curve.append(dict(step=step, training_loss=float(loss), validation_normalized_mse=error))
            if error < best:
                best, selected_step, best_parameters = error, step, model.parameters()
    model.set_parameters(best_parameters)
    prediction = predict(xs[2])
    memory.append(checked_memory(model))
    return prediction, dict(seed=seed, selected_step=selected_step, curve=curve,
                            batch=batch, hidden=hidden, steps=steps, validation_interval=interval, learning_rate=1e-3,
                            weight_penalty=1e-4/len(train_x), optimizer="Adam .9/.999/1e-8",
                            memory=memory, selection="validation normalized MSE; no refit or test selection")


def load_split(dataset, features, rows, split, variant):
    xs, ys, seeds = [], [], []
    for row in rows:
        if row["split"] != split:
            continue
        with np.load(features / row["file"]) as data:
            values = data[variant].reshape(row["count"], -1)
        with np.load(dataset / row["file"]) as data:
            labels = data["targets"][:len(values)]
        xs.append(values)
        ys.append(labels)
        seeds.extend([row["seed"]] * len(values))
    return np.concatenate(xs), np.concatenate(ys), np.asarray(seeds)


def evaluate(prediction, targets, seeds, names, visible):
    return dict(all=dict(zip(names, regression_metrics(prediction, targets))),
                visible=dict(zip(names, regression_metrics(prediction, np.where(visible, targets, np.nan)))),
                by_trajectory={str(seed): dict(zip(names, regression_metrics(prediction[seeds==seed], targets[seeds==seed])))
                               for seed in np.unique(seeds)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("ridge", "mlp"), required=True)
    parser.add_argument("--head-seeds", nargs="+", type=int, default=(1009, 2017, 3019))
    parser.add_argument("--mlp-steps", type=int, default=512)
    parser.add_argument("--visibility", type=Path, required=True,
                        help="independent sprite-coverage masks, only for secondary test reporting")
    args = parser.parse_args()
    if args.mlp_steps <= 0 or len(set(args.head_seeds)) != len(args.head_seeds):
        parser.error("positive step budget and unique head seeds required")
    feature_manifest = json.loads((args.features / "manifest.json").read_text())
    if feature_manifest["status"] != "complete" or feature_manifest["protocol"] != PROTOCOL:
        parser.error("complete matching feature extraction required")
    if feature_manifest["dataset_manifest_sha256"] != sha256_file(args.dataset / "manifest.json"):
        parser.error("feature/dataset identity mismatch")
    for row in feature_manifest["files"]:
        if sha256_file(args.features / row["file"]) != row["sha256"]:
            parser.error(f"feature bytes changed: {row['file']}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(protocol=PROTOCOL, method=args.method, status="running", results=[],
                  normalization=("training-only F64 statistics; std<=1e-6 uses scale1; F32 GPU inputs"
                                 if args.method == "mlp" else "training-only F64 ridge statistics"),
                  encoder_sha256=feature_manifest["encoder_sha256"],
                  feature_manifest_sha256=sha256_file(args.features / "manifest.json"),
                  native_sha256=sha256_file(_native.__file__), limit_clips=feature_manifest["limit_clips"],
                  visibility_sha256=sha256_file(args.visibility),
                  limits=["offline diagnostic, not gameplay or forecasts", "head seeds are fit variability, not RL seeds"])
    model = None
    for game in dict.fromkeys(r["game"] for r in feature_manifest["files"]):
        rows = [r for r in feature_manifest["files"] if r["game"] == game]
        fitted = {}
        with np.load(args.visibility) as visibility:
            visible = np.concatenate([visibility[r["file"]][:r["count"]] for r in rows if r["split"] == "test"])
        for variant in rows[0]["variants"]:
            train_x, train_y, _ = load_split(args.dataset, args.features, rows, "train", variant)
            val_x, val_y, _ = load_split(args.dataset, args.features, rows, "validation", variant)
            test_x, test_y, test_seeds = load_split(args.dataset, args.features, rows, "test", variant)
            names = target_names(game)
            identity = feature_identity(train_x, val_x, test_x)
            reused = fitted.get(identity)
            if reused is not None:
                # The stateless CNN has identical phase0/15 inputs. Reuse only
                # byte-identical complete splits, never "similar" features or
                # favorable labels. Same initialization, data and training budget.
                fits = reused[1]
            else:
                fits = []
                for seed in ((None,) if args.method == "ridge" else args.head_seeds):
                    if seed is None:
                        prediction, info = ridge_probe(train_x, train_y, val_x, val_y, test_x)
                    else:
                        if model is None:
                            model = _native.RegressionProbe(train_x.shape[1], train_y.shape[1], hidden=128, batch=64)
                        prediction, info = mlp_probe(train_x, train_y, val_x, val_y, test_x, seed,
                                                     steps=args.mlp_steps, model=model)
                    fits.append(dict(fit=info, test=evaluate(prediction, test_y, test_seeds, names, visible)))
                fitted[identity] = (variant, fits)
            constant = np.broadcast_to(np.nanmean(train_y, axis=0), test_y.shape)
            row = dict(game=game, variant=variant, fits=fits, reuses_fit_from=reused[0] if reused else None,
                       constant=evaluate(constant, test_y, test_seeds, names, visible), seconds=time.monotonic()-started)
            result["results"].append(row)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(dict(game=game, variant=variant, seconds=row["seconds"])), flush=True)
    result.update(status="complete", seconds=time.monotonic()-started)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
