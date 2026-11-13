"""One-factor production RSSM target-conditioning screen on saved frozen features.

No gameplay or encoder updates. RAM labels enter only the final frozen readout
diagnostics. Run under gpu_host_guard.py; see the experiment declaration.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open

from atari import sha256_file
from fit_atari_probes import bytes32, checked_memory
from fit_fixed_latents import (HEAD_SEED, NAMES, latent_errors, latent_report,
                              predict, state_report, validate_corpus)
from kindle import _native
from kindle._reward_probe import roc_auc
from probe_fixed_latents import assert_frozen_tensors


def training_statistics(root, manifest):
    features, rewards = [], []
    for row in manifest["files"]:
        if row["split"] == "train":
            with np.load(root / row["file"]) as data:
                features.append(data["features"][data["following"]])
                rewards.append(data["rewards"])
    values = np.concatenate(features)
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    scale = values.std(0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.).astype(np.float32)
    return mean, scale, float(np.concatenate(rewards).mean(dtype=np.float64))


def learner_config(base, mean, scale, standardize, posterior_targets=False):
    if posterior_targets and not standardize:
        raise ValueError("posterior-target screen requires standardized future targets")
    config = copy.deepcopy(base)
    config.update(video_encoder=None, replay_capacity=16384, train_ratio=0.,
                  future_target_standardization=(dict(mean=mean.tolist(), scale=scale.tolist())
                                                 if standardize else None))
    config['reconstruction_target_standardization'] = None
    if posterior_targets:
        config['loss_scales']['reconstruction'] = .25
        config['reconstruction_target_standardization'] = copy.deepcopy(config['future_target_standardization'])
    return config


def replay_trace(core, data, limit=None, before=None):
    """Replay real boundaries; close an artificial prefix separately from terminal."""
    count = min(len(data["actions"]), limit) if limit is not None else len(data["actions"])
    arrivals = 0
    for i in range(count):
        if i == 0 or data["episodes"][i] != data["episodes"][i - 1]:
            core.begin_episode(bytes32(data["features"][data["current"][i]]))
            arrivals += 1
        if before is not None:
            before(i, count)
        core.observe(int(data["actions"][i]), bytes32(data["features"][data["following"][i]]),
                     float(data["rewards"][i]), bool(data["terminated"][i]),
                     bool(data["truncated"][i] or i == count - 1))
        arrivals += 1
    return count, arrivals


def compare_initial(reference, candidate):
    counts = {}
    for group in ('world', 'behavior', 'slow_value'):
        with safe_open(reference / f'{group}.safetensors', framework='np') as old, \
                safe_open(candidate / f'{group}.safetensors', framework='np') as new:
            extra = set(new.keys()) - set(old.keys())
            if not set(old.keys()) <= set(new.keys()) or any('world.decoder.' not in k for k in extra):
                raise RuntimeError('unexpected initial tensor keys')
            for key in old.keys():
                a, b = old.get_tensor(key), new.get_tensor(key)
                if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                    raise RuntimeError(f'initial control mismatch: {group}/{key}')
            counts[group] = dict(shared=len(old.keys()), added=len(extra))
    return counts


def forecast_trace(core, data, seed, stride, limit=None):
    """Paired proposed/unrelated actions, retaining terminal outcomes, never resets."""
    rows = {h: dict(origins=[], prediction=[], unrelated_actions=[],
                   reward_prediction=[], reward_unrelated_actions=[]) for h in (1, 15)}
    rng = np.random.default_rng(HEAD_SEED ^ seed)
    step = core.learner_step

    def before(i, count):
        if i % stride:
            return
        horizon = 15 if i + 15 <= count and data["episodes"][i] == data["episodes"][i + 14] else 1
        actions = data["actions"][i:i + horizon].tolist()
        rewards, features = core.forecast(actions)
        unrelated_rewards, unrelated_features = core.forecast(rng.integers(18, size=horizon).tolist())
        for h, endpoint in ((1, 0), (15, 1)):
            if h > horizon:
                continue
            row = rows[h]
            row["origins"].append(i)
            for name, values in (("prediction", features), ("unrelated_actions", unrelated_features),
                                 ("reward_prediction", rewards), ("reward_unrelated_actions", unrelated_rewards)):
                row[name].append(np.asarray(values[endpoint], np.float32))

    replay_trace(core, data, limit, before)
    if core.learner_step != step:
        raise RuntimeError("evaluation updated learner")
    return {h: {k: np.asarray(v) for k, v in row.items()} for h, row in rows.items()}


def reward_report(prediction, target, seeds):
    def metrics(p, y):
        return dict(count=len(y), positive_count=int((y > 0).sum()),
                    mae=float(np.abs(p - y).mean()), rmse=float(np.sqrt(np.square(p - y).mean())),
                    event_auc=roc_auc((y > 0).tolist(), p.tolist()))
    return dict(all=metrics(prediction, target), by_trajectory={str(seed): metrics(
        prediction[seeds == seed], target[seeds == seed]) for seed in np.unique(seeds)})


def score_trace(rows, data, seed, mean, scale, readout, norm):
    evidence = {}
    for h, row in rows.items():
        if not len(row["origins"]):
            continue
        origins = row["origins"].astype(np.int64)
        end = origins + h - 1
        current, following = data["current"][origins], data["following"][end]
        target = data["features"][following]
        part = dict(origins=origins, seeds=np.full(len(origins), seed),
                    crossing=data["phase"][current] + h >= 16,
                    labels=np.column_stack((data["positions"][following], data["rewards"][end] > 0,
                                            data["terminated"][end])), reward_target=data["rewards"][end],
                    reward_prediction=row["reward_prediction"],
                    reward_unrelated_actions=row["reward_unrelated_actions"])
        for name, features in (("real_future", target), ("prediction", row["prediction"]),
                               ("unrelated_actions", row["unrelated_actions"]),
                               ("persistence", data["features"][current]),
                               ("training_mean", np.broadcast_to(mean, target.shape))):
            if not np.isfinite(features).all():
                raise RuntimeError("nonfinite forecast")
            part["decoded_" + name] = predict(readout, features, norm)
            if name != "real_future":
                part["error_" + name] = latent_errors(features, target, scale)
        evidence[h] = part
    return evidence


def evaluate(core, root, manifest, output, mean, scale, reward_mean, readout_path, smoke):
    with np.load(readout_path) as head:
        norm = {key: head[key] for key in ("x_mean", "x_scale", "y_mean", "y_scale")}
        parameters = [head[f"parameter_{i}"].tolist() for i in range(4)]
    readout = _native.RegressionProbe(len(mean), len(NAMES), hidden=128, batch=64, seed=HEAD_SEED)
    checked_memory(readout)
    readout.set_parameters(parameters)
    result = {}
    for split in (("test",) if smoke else ("train", "validation", "test")):
        pieces = {1: [], 15: []}
        files = [row for row in manifest["files"] if row["split"] == split]
        for row in files[:1] if smoke else files:
            with np.load(root / row["file"]) as archive:
                data = dict(archive)
            forecasts = forecast_trace(core, data, row["seed"], 4 if split == "test" else 16,
                                       128 if smoke else (512 if split == "train" else None))
            scored = score_trace(forecasts, data, row["seed"], mean, scale, readout, norm)
            for h, part in scored.items():
                pieces[h].append(part)
            checked_memory(core)
            del data, forecasts, scored
        result[split] = {}
        for h, parts in pieces.items():
            merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            errors = {key.removeprefix("error_"): latent_report(values, merged)
                      for key, values in merged.items() if key.startswith("error_")}
            decoded = {key.removeprefix("decoded_"): state_report(values, merged)
                       for key, values in merged.items() if key.startswith("decoded_")}
            reward = {name: reward_report(values, merged["reward_target"], merged["seeds"])
                      for name, values in (("prediction", merged["reward_prediction"]),
                          ("unrelated_actions", merged["reward_unrelated_actions"]),
                          ("zero", np.zeros_like(merged["reward_target"])),
                          ("training_mean", np.full_like(merged["reward_target"], reward_mean)))}
            path = output / f"{split}-h{h}.npz"
            np.savez(path, **merged)
            result[split][str(h)] = dict(latents=errors, decoded=decoded, rewards=reward,
                                        evidence_file=path.name, evidence_sha256=sha256_file(path))
        print(json.dumps(dict(evaluated=split, learner_step=core.learner_step)), flush=True)
    if readout.parameters() != parameters:
        raise RuntimeError("frozen state readout changed")
    return result


def run(root, readout_path, output, standardize, smoke, posterior_targets=False, initial_reference=None):
    started = time.monotonic()
    manifest = json.loads((root / "manifest.json").read_text())
    validate_corpus(root, manifest)
    checkpoint = Path(manifest["checkpoint"]["path"]) / "metadata.json"
    if sha256_file(checkpoint) != manifest["checkpoint"]["metadata_sha256"]:
        raise ValueError("changed source configuration")
    head_result = json.loads((readout_path.parent / "result.json").read_text())
    if (head_result["status"] != "complete" or head_result["corpus_manifest_sha256"] != sha256_file(root / "manifest.json")
            or head_result["heads"][0]["head_sha256"] != sha256_file(readout_path)):
        raise ValueError("readout does not match corpus")
    mean, scale, reward_mean = training_statistics(root, manifest)
    config = learner_config(json.loads(checkpoint.read_text())["config"], mean, scale, standardize, posterior_targets)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "training-statistics.npz", mean=mean, scale=scale, reward_mean=reward_mean)
    result = dict(protocol="kindle-rssm-target-standardization-v1", status="running", smoke=smoke,
                  standardize=standardize, posterior_targets=posterior_targets,
                  seed=config["seed"], config=config, new_game_actions=0,
                  corpus=str(root.resolve()), corpus_manifest_sha256=sha256_file(root / "manifest.json"),
                  readout=str(readout_path.resolve()), readout_sha256=sha256_file(readout_path),
                  source_sha256=sha256_file(__file__), native_sha256=sha256_file(_native.__file__),
                  statistics_sha256=sha256_file(output / "training-statistics.npz"), curves=[])

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    core = _native.FeatureCore(config)
    result.update(gpu_device=core.gpu_device, memory=[checked_memory(core)])
    core.save_checkpoint(str(output / "initial"))
    if initial_reference is not None:
        result['initial_reference'] = str(initial_reference.resolve())
        result['initial_tensor_comparison'] = compare_initial(initial_reference, output / 'initial')
        save()
    files = [row for row in manifest["files"] if row["split"] == "train"]
    actions = arrivals = 0
    for row in files[:1] if smoke else files:
        with np.load(root / row["file"]) as archive:
            # No RAM labels are even loaded into the training path.
            data = {key: archive[key] for key in ("features", "current", "following", "actions",
                                                 "rewards", "terminated", "truncated", "episodes")}
        count, inserted = replay_trace(core, data, 512 if smoke else None)
        actions += count
        arrivals += inserted
        del data
    if (core.learner_step != 0 or core.environment_step != actions or core.replay_len != arrivals
            or actions != (512 if smoke else 12288)):
        raise RuntimeError("unexpected replay ingest accounting")
    result.update(recorded_training_actions=actions, replay_arrivals=arrivals,
                  preparation_seconds=time.monotonic() - started)
    updates = 2 if smoke else 2048
    learning_started = time.monotonic()
    for step in range(1, updates + 1):
        report = core.learn()
        if report is None:
            raise RuntimeError("missing production learner update")
        # Serializing the full report rejects NaNs, including unreported stages.
        json.dumps(report, allow_nan=False)
        if core.learner_step != step or core.environment_step != actions:
            raise RuntimeError("learner/recorded-action accounting changed")
        if step % 128 == 0 or step == updates:
            result["curves"].append(dict(update=step, seconds=time.monotonic() - learning_started, report=report))
            result["memory"].append(checked_memory(core))
            save()
    result.update(updates=core.learner_step, learning_seconds=time.monotonic() - learning_started)
    core.save_checkpoint(str(output / "final"))
    print(json.dumps(dict(trained=config["seed"], standardize=standardize, updates=updates,
                          seconds=result["learning_seconds"])), flush=True)
    evaluation_started = time.monotonic()
    result["diagnostics"] = evaluate(core, root, manifest, output, mean, scale, reward_mean, readout_path, smoke)
    core.save_checkpoint(str(output / "after-evaluation"))
    assert_frozen_tensors(output / "final", output / "after-evaluation")
    if core.learner_step != updates:
        raise RuntimeError("frozen diagnostics updated learner")
    result.update(status="complete", frozen_tensor_bytes_unchanged=True, final_learner_step=core.learner_step,
                  recorded_evaluation_actions=core.environment_step - actions,
                  evaluation_seconds=time.monotonic() - evaluation_started, seconds=time.monotonic() - started)
    result["memory"].append(checked_memory(core))
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("readout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--posterior-targets", action="store_true",
                        help="add standardized current-feature prediction from the full posterior (weight .25)")
    parser.add_argument("--initial-reference", type=Path, help="require exact shared initial control tensors before learning")
    args = parser.parse_args()
    run(args.corpus, args.readout, args.output, args.standardize, args.smoke, args.posterior_targets, args.initial_reference)


if __name__ == "__main__":
    main()
