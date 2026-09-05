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

from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing, sha256_file
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
    motion: bool = False,
    protocol_name: str = "current",
    agent=None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    protocol = ATARI_PROTOCOLS[protocol_name]
    environment = gym.make(
        environment_name,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=protocol.full_action_space,
    )
    environment = DreamerAtariPreprocessing(environment, noop_max=protocol.noop_max,
                                            max_episode_frames=protocol.max_episode_frames)
    frame, _ = environment.reset(seed=seed)
    actions = random.Random(seed ^ 0xD1_30_00_03)
    features = {"rgb64": [], "projected14": [], "pooled7": []}
    if motion:
        features["pooled7_history"] = []
    if agent is not None:
        if agent.config["action_count"] != environment.action_space.n:
            raise ValueError("checkpoint and probe action vocabularies differ")
        agent.begin_episode(frame)
        features["rssm"] = []
    previous = None
    labels = []
    attempts = 0
    maximum_attempts = max(10_000, 50 * sample_count)

    try:
        while len(labels) < sample_count and attempts < maximum_attempts:
            attempts += 1
            action = actions.randrange(environment.action_space.n)
            if agent is not None:
                mask = [index == action for index in range(environment.action_space.n)]
                if agent.act(action_mask=mask) != action:
                    raise RuntimeError("forced action was not honored")
            frame, reward, terminated, truncated, _ = environment.step(action)
            if agent is not None:
                # Recurrence consumes every arrival, including unlabeled frames.
                # No learner update or privileged object label enters the agent.
                agent.observe(frame, extrinsic_reward=float(reward),
                              terminated=bool(terminated), truncated=bool(truncated))
            label = pong_labels(environment)
            if label is not None and not (terminated or truncated):
                projected, pooled = encoder.encode(frame)
                current = {
                    "rgb64": np.asarray(frame, dtype=np.float32).reshape(-1) / 255.0,
                    "projected14": np.asarray(projected, dtype=np.float32),
                    "pooled7": np.asarray(pooled, dtype=np.float32),
                }
                if agent is not None:
                    current["rssm"] = np.asarray(agent.latent_feature, dtype=np.float32)
                if not motion or previous is not None:
                    for name, values in current.items():
                        features[name].append(values)
                    if motion:
                        previous_pooled, previous_label = previous
                        features["pooled7_history"].append(np.concatenate(
                            (current["pooled7"], current["pooled7"] - previous_pooled)))
                        labels.append(label - previous_label)
                    else:
                        labels.append(label)
                previous = (current["pooled7"], label)
                if len(labels) % 64 == 0 or len(labels) == sample_count:
                    print(
                        f"seed {seed}: {len(labels)}/{sample_count} samples "
                        f"after {attempts} decisions",
                        flush=True,
                    )
            else:
                # No temporal pair may span an unlabeled frame or reset.
                previous = None
            if terminated or truncated:
                frame, _ = environment.reset()
                if agent is not None:
                    agent.begin_episode(frame)
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


def _metrics(prediction: np.ndarray, target: np.ndarray,
             targets: tuple[str, ...] = TARGETS) -> dict[str, object]:
    pixel_scales = (159.0, 209.0, 209.0, 209.0)
    errors = np.abs(prediction - target)
    residual = np.square(prediction - target).sum(axis=0)
    total = np.square(target - target.mean(axis=0)).sum(axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    per_target = {}
    for index, name in enumerate(targets):
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
    targets: tuple[str, ...] = TARGETS,
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
            for index, name in enumerate(targets)
        }
    selected_alphas = [
        min(
            RIDGE_ALPHAS,
            key=lambda alpha: validation_scores[str(alpha)][name],
        )
        for name in targets
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
        "selected_alpha_by_target": dict(zip(targets, selected_alphas)),
        "validation_mse_by_alpha": validation_scores,
        "test": _metrics(prediction, label_splits[3], targets),
        "constant_baseline": _metrics(baseline, label_splits[3], targets),
    }


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--samples-per-seed", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs=4, default=(0, 1, 2, 3))
    parser.add_argument("--dino-plan-cache")
    parser.add_argument("--agent-checkpoint", type=Path,
                        help="also probe a frozen RSSM's policy input; use its original backend")
    parser.add_argument("--output")
    parser.add_argument("--motion", action="store_true",
                        help="probe last-action displacement, adding causal two-frame pooled history")
    parser.add_argument("--atari-protocol", choices=tuple(ATARI_PROTOCOLS), default="current")
    args = parser.parse_args()
    if args.samples_per_seed <= 0:
        parser.error("--samples-per-seed must be positive")
    if len(set(args.seeds)) != 4:
        parser.error("train, validation and test seeds must be distinct")
    if args.environment != "ALE/Pong-v5":
        parser.error("this color/object probe is specific to ALE/Pong-v5")
    if not hasattr(_native, "DinoPerception"):
        parser.error("rebuild the Kindle extension to enable the DINO probe")

    encoder = _native.DinoPerception(
        args.dino_checkpoint,
        dino_plan_cache=args.dino_plan_cache,
    )
    agent_metadata = None
    if args.agent_checkpoint:
        agent_metadata = json.loads((args.agent_checkpoint / "metadata.json").read_text())
    print(
        f"DINO feature shapes: projected={encoder.projected_shape}, "
        f"pooled={encoder.pooled_shape}",
        flush=True,
    )

    splits = []
    for seed in args.seeds:
        # Each independent environment starts from the same checkpoint/RNG.
        # Do not manufacture a terminal flag to reset a live agent between seeds.
        agent = (_native.Agent.restore(str(args.agent_checkpoint), args.dino_checkpoint)
                 if args.agent_checkpoint else None)
        starting_learner_step = agent.learner_step if agent is not None else None
        splits.append(collect_split(
            encoder,
            args.environment,
            seed,
            args.samples_per_seed,
            args.motion,
            args.atari_protocol,
            agent,
        ))
        if agent is not None and agent.learner_step != starting_learner_step:
            raise RuntimeError("a frozen representation probe must not learn")
        del agent
    label_splits = [labels for _, labels in splits]
    results = {}
    targets = tuple(f"delta_{name}" for name in TARGETS) if args.motion else TARGETS
    for representation in splits[0][0]:
        print(f"fitting {representation} probe", flush=True)
        results[representation] = evaluate_representation(
            [features[representation] for features, _ in splits],
            label_splits,
            targets,
        )

    summary = {
        "environment": args.environment,
        "ale_py_version": ale_py.__version__,
        "atari_protocol": args.atari_protocol,
        "dino_checkpoint_sha256": sha256_file(args.dino_checkpoint),
        "native_extension_sha256": sha256_file(_native.__file__),
        "runner_sha256": sha256_file(__file__),
        "target": "displacement_t_minus_1_to_t" if args.motion else "position_t",
        "history": "[u_t, u_t - u_(t-1)]; no future input" if args.motion else None,
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
        "agent_checkpoint": str(args.agent_checkpoint) if args.agent_checkpoint else None,
        "agent_checkpoint_metadata": agent_metadata,
        "agent_world_sha256": (sha256_file(args.agent_checkpoint / "world.safetensors")
                               if args.agent_checkpoint else None),
        "agent_learner_updates": 0 if args.agent_checkpoint else None,
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
