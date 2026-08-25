"""Probe whether Kindle's frozen DINO features retain small Atari objects.

The probe compares three representations of the exact 64x64 frame given to
Dreamer:

* raw RGB pixels (the positive control),
* the fixed 64-channel projection of DINO's 14x14 patch grid, and
* the production 7x7 grid after fixed 2x2 spatial pooling.

Ridge probes predict Pong's ball and paddle positions. Seeds are split by
role so adjacent frames never cross the train/validation/test boundary:
two seeds train, one selects each target's ridge penalty, and one is the final
test.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

from atari import DreamerAtariPreprocessing
from kindle import _native


PONG_BALL = np.array((236, 236, 236), dtype=np.uint8)
PONG_OPPONENT = np.array((213, 130, 74), dtype=np.uint8)
PONG_PLAYER = np.array((92, 186, 92), dtype=np.uint8)
PLAYFIELD_TOP = 34
PLAYFIELD_BOTTOM = 194
TARGETS = ("ball_x", "ball_y", "player_y", "opponent_y")
RIDGE_ALPHAS = (
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
)


def _object_center(
    frames: tuple[np.ndarray, np.ndarray],
    color: np.ndarray,
) -> tuple[float, float] | None:
    points = []
    for frame in frames:
        mask = np.all(frame == color, axis=2)
        mask[:PLAYFIELD_TOP] = False
        mask[PLAYFIELD_BOTTOM:] = False
        y, x = np.nonzero(mask)
        if x.size:
            points.append(np.stack((x, y), axis=1))
    if not points:
        return None
    return tuple(np.concatenate(points).mean(axis=0))


def pong_labels(
    environment: DreamerAtariPreprocessing,
) -> np.ndarray | None:
    """Return normalized object centers aligned with the max-pooled input."""
    if len(environment._frames) != 2:
        return None
    frames = (environment._frames[0], environment._frames[1])
    ball = _object_center(frames, PONG_BALL)
    player = _object_center(frames, PONG_PLAYER)
    opponent = _object_center(frames, PONG_OPPONENT)
    if ball is None or player is None or opponent is None:
        return None
    height, width = frames[0].shape[:2]
    return np.asarray(
        (
            ball[0] / (width - 1),
            ball[1] / (height - 1),
            player[1] / (height - 1),
            opponent[1] / (height - 1),
        ),
        dtype=np.float32,
    )


def collect_split(
    encoder,
    environment_name: str,
    seed: int,
    sample_count: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    environment = gym.make(
        environment_name,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    environment = DreamerAtariPreprocessing(environment)
    frame, _ = environment.reset(seed=seed)
    actions = random.Random(seed ^ 0xD1_30_00_03)
    features = {"rgb64": [], "projected14": [], "pooled7": []}
    labels = []
    attempts = 0
    maximum_attempts = max(10_000, 50 * sample_count)

    try:
        while len(labels) < sample_count and attempts < maximum_attempts:
            attempts += 1
            action = actions.randrange(environment.action_space.n)
            frame, _, terminated, truncated, _ = environment.step(action)
            label = pong_labels(environment)
            if label is not None and not (terminated or truncated):
                projected, pooled = encoder.encode(frame)
                features["rgb64"].append(
                    np.asarray(frame, dtype=np.float32).reshape(-1) / 255.0
                )
                features["projected14"].append(
                    np.asarray(projected, dtype=np.float32)
                )
                features["pooled7"].append(np.asarray(pooled, dtype=np.float32))
                labels.append(label)
                if len(labels) % 64 == 0 or len(labels) == sample_count:
                    print(
                        f"seed {seed}: {len(labels)}/{sample_count} samples "
                        f"after {attempts} decisions",
                        flush=True,
                    )
            if terminated or truncated:
                frame, _ = environment.reset()
    finally:
        environment.close()

    if len(labels) != sample_count:
        raise RuntimeError(
            f"seed {seed} produced only {len(labels)} labeled frames "
            f"in {attempts} decisions"
        )
    return (
        {name: np.stack(values) for name, values in features.items()},
        np.stack(labels),
    )


def _standardize(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    width_scale = np.float32(np.sqrt(train.shape[1]))
    return tuple(
        np.asarray((values - mean) / scale / width_scale, dtype=np.float32)
        for values in (train, *others)
    )


def _predict_dual_ridge_many(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Evaluate several penalties while forming the feature Gram only once."""
    target_mean = train_y.mean(axis=0)
    centered_targets = np.asarray(train_y - target_mean, dtype=np.float64)
    gram = np.asarray(train_x @ train_x.T, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    target_basis = eigenvectors.T @ centered_targets
    test_kernel = np.asarray(test_x @ train_x.T, dtype=np.float64)
    return {
        alpha: np.asarray(
            test_kernel
            @ (
                eigenvectors
                @ (target_basis / (eigenvalues[:, None] + alpha))
            )
            + target_mean
        )
        for alpha in alphas
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    pixel_scales = (159.0, 209.0, 209.0, 209.0)
    errors = np.abs(prediction - target)
    residual = np.square(prediction - target).sum(axis=0)
    total = np.square(target - target.mean(axis=0)).sum(axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    per_target = {}
    for index, name in enumerate(TARGETS):
        per_target[name] = {
            "mae_normalized": float(errors[:, index].mean()),
            "mae_source_pixels": float(
                errors[:, index].mean() * pixel_scales[index]
            ),
            "r2": float(r2[index]),
        }
    return {
        "mean_r2": float(r2.mean()),
        "targets": per_target,
    }


def evaluate_representation(
    feature_splits: list[np.ndarray],
    label_splits: list[np.ndarray],
) -> dict[str, object]:
    selection_x = np.concatenate(feature_splits[:2])
    selection_y = np.concatenate(label_splits[:2])
    selection_x, validation_x = _standardize(selection_x, feature_splits[2])

    validation_scores = {}
    validation_predictions = _predict_dual_ridge_many(
        selection_x,
        selection_y,
        validation_x,
        RIDGE_ALPHAS,
    )
    for alpha, prediction in validation_predictions.items():
        validation_scores[str(alpha)] = {
            name: float(
                np.square(prediction[:, index] - label_splits[2][:, index]).mean()
            )
            for index, name in enumerate(TARGETS)
        }
    selected_alphas = [
        min(
            RIDGE_ALPHAS,
            key=lambda alpha: validation_scores[str(alpha)][name],
        )
        for name in TARGETS
    ]

    refit_x = np.concatenate(feature_splits[:3])
    refit_y = np.concatenate(label_splits[:3])
    refit_x, test_x = _standardize(refit_x, feature_splits[3])
    refit_predictions = _predict_dual_ridge_many(
        refit_x,
        refit_y,
        test_x,
        tuple(sorted(set(selected_alphas))),
    )
    prediction = np.stack(
        [
            refit_predictions[alpha][:, index]
            for index, alpha in enumerate(selected_alphas)
        ],
        axis=1,
    )
    baseline = np.broadcast_to(refit_y.mean(axis=0), label_splits[3].shape)
    return {
        "dimensions": int(feature_splits[0].shape[1]),
        "selected_alpha_by_target": dict(zip(TARGETS, selected_alphas)),
        "validation_mse_by_alpha": validation_scores,
        "test": _metrics(prediction, label_splits[3]),
        "constant_baseline": _metrics(baseline, label_splits[3]),
    }


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--samples-per-seed", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs=4, default=(0, 1, 2, 3))
    parser.add_argument("--dino-plan-cache")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.samples_per_seed <= 0:
        parser.error("--samples-per-seed must be positive")
    if not hasattr(_native, "DinoPerception"):
        parser.error("rebuild the Kindle extension to enable the DINO probe")

    encoder = _native.DinoPerception(
        args.dino_checkpoint,
        dino_plan_cache=args.dino_plan_cache,
    )
    print(
        f"DINO feature shapes: projected={encoder.projected_shape}, "
        f"pooled={encoder.pooled_shape}",
        flush=True,
    )

    splits = [
        collect_split(
            encoder,
            args.environment,
            seed,
            args.samples_per_seed,
        )
        for seed in args.seeds
    ]
    label_splits = [labels for _, labels in splits]
    results = {}
    for representation in ("rgb64", "projected14", "pooled7"):
        print(f"fitting {representation} probe", flush=True)
        results[representation] = evaluate_representation(
            [features[representation] for features, _ in splits],
            label_splits,
        )

    summary = {
        "environment": args.environment,
        "samples_per_seed": args.samples_per_seed,
        "seeds": args.seeds,
        "split_roles": {
            "train": args.seeds[:2],
            "validation": args.seeds[2],
            "test": args.seeds[3],
        },
        "projected_shape": encoder.projected_shape,
        "pooled_shape": encoder.pooled_shape,
        "gpu_device": encoder.gpu_device,
        "results": results,
    }
    encoded = json.dumps(summary, indent=2)
    print(encoded, flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
