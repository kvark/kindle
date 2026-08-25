"""Probe whether frozen DINO features retain Atari reward events.

The probe collects independent random trajectories for two training seeds, one
validation seed, and one held-out test seed. Every nonzero-reward frame is kept
along with a deterministic sample of zero-reward frames. Ridge probes compare
raw 64x64 RGB, projected 14x14 DINO patches, and Kindle's pooled 7x7 input.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing
from kindle import _native
from kindle._reward_probe import roc_auc
from probe_atari_perception import (
    RIDGE_ALPHAS,
    _predict_dual_ridge_many,
    _standardize,
)


REPRESENTATIONS = ("rgb64", "projected14", "pooled7")


def collect_split(
    encoder,
    environment_name: str,
    atari_protocol: str,
    seed: int,
    decision_count: int,
    zero_keep_probability: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    protocol = ATARI_PROTOCOLS[atari_protocol]
    environment = gym.make(
        environment_name,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=protocol.full_action_space,
    )
    environment = DreamerAtariPreprocessing(
        environment,
        noop_max=protocol.noop_max,
        max_episode_frames=protocol.max_episode_frames,
    )
    frame, _ = environment.reset(seed=seed)
    actions = random.Random(seed ^ 0xD1_30_A7A2)
    zero_selection = random.Random(seed ^ 0xA7A2_D1_30)
    features: dict[str, list[np.ndarray]] = {
        name: [] for name in REPRESENTATIONS
    }
    rewards = []

    try:
        for decision in range(1, decision_count + 1):
            action = actions.randrange(environment.action_space.n)
            frame, reward, terminated, truncated, _ = environment.step(action)
            reward = float(reward)
            if reward != 0.0 or zero_selection.random() < zero_keep_probability:
                projected, pooled = encoder.encode(frame)
                features["rgb64"].append(
                    np.asarray(frame, dtype=np.float32).reshape(-1) / 255.0
                )
                features["projected14"].append(
                    np.asarray(projected, dtype=np.float32)
                )
                features["pooled7"].append(
                    np.asarray(pooled, dtype=np.float32)
                )
                rewards.append(reward)
            if decision % 1_000 == 0 or decision == decision_count:
                reward_count = sum(value != 0.0 for value in rewards)
                print(
                    f"seed {seed}: {decision}/{decision_count} decisions, "
                    f"kept {len(rewards)} frames ({reward_count} rewards)",
                    flush=True,
                )
            if terminated or truncated:
                frame, _ = environment.reset()
    finally:
        environment.close()

    if not any(reward == 0.0 for reward in rewards) or not any(
        reward != 0.0 for reward in rewards
    ):
        raise RuntimeError(f"seed {seed} did not retain both reward classes")
    return (
        {name: np.stack(values) for name, values in features.items()},
        np.asarray(rewards, dtype=np.float32),
    )


def _ranking_metrics(
    rewards: np.ndarray,
    event_prediction: np.ndarray,
    signed_prediction: np.ndarray,
) -> dict[str, object]:
    reward_values = rewards.tolist()
    event_scores = event_prediction.tolist()
    signed_scores = signed_prediction.tolist()
    nonzero_indices = np.flatnonzero(rewards != 0.0)
    return {
        "count": int(rewards.size),
        "positive_count": int(np.count_nonzero(rewards > 0.0)),
        "zero_count": int(np.count_nonzero(rewards == 0.0)),
        "negative_count": int(np.count_nonzero(rewards < 0.0)),
        "reward_event_vs_zero_roc_auc": roc_auc(
            [reward != 0.0 for reward in reward_values], event_scores
        ),
        "positive_vs_rest_roc_auc": roc_auc(
            [reward > 0.0 for reward in reward_values], signed_scores
        ),
        "negative_vs_rest_roc_auc": roc_auc(
            [reward < 0.0 for reward in reward_values],
            [-score for score in signed_scores],
        ),
        "positive_vs_negative_roc_auc": roc_auc(
            [bool(rewards[index] > 0.0) for index in nonzero_indices],
            [float(signed_prediction[index]) for index in nonzero_indices],
        ),
    }


def evaluate_representation(
    feature_splits: list[np.ndarray],
    reward_splits: list[np.ndarray],
) -> dict[str, object]:
    selection_x = np.concatenate(feature_splits[:2])
    selection_rewards = np.concatenate(reward_splits[:2])
    selection_targets = np.stack(
        ((selection_rewards != 0.0).astype(np.float32), selection_rewards),
        axis=1,
    )
    selection_x, validation_x = _standardize(selection_x, feature_splits[2])
    validation_predictions = _predict_dual_ridge_many(
        selection_x,
        selection_targets,
        validation_x,
        RIDGE_ALPHAS,
    )
    validation_events = (reward_splits[2] != 0.0).tolist()
    event_auc_by_alpha = {
        alpha: roc_auc(validation_events, prediction[:, 0].tolist())
        for alpha, prediction in validation_predictions.items()
    }
    event_alpha = max(
        RIDGE_ALPHAS,
        key=lambda alpha: (
            -1.0
            if event_auc_by_alpha[alpha] is None
            else event_auc_by_alpha[alpha]
        ),
    )
    signed_alpha = min(
        RIDGE_ALPHAS,
        key=lambda alpha: float(
            np.square(
                validation_predictions[alpha][:, 1] - reward_splits[2]
            ).mean()
        ),
    )

    refit_x = np.concatenate(feature_splits[:3])
    refit_rewards = np.concatenate(reward_splits[:3])
    refit_targets = np.stack(
        ((refit_rewards != 0.0).astype(np.float32), refit_rewards), axis=1
    )
    refit_x, test_x = _standardize(refit_x, feature_splits[3])
    test_predictions = _predict_dual_ridge_many(
        refit_x,
        refit_targets,
        test_x,
        tuple(sorted({event_alpha, signed_alpha})),
    )
    return {
        "dimensions": int(feature_splits[0].shape[1]),
        "selected_event_alpha": event_alpha,
        "selected_signed_alpha": signed_alpha,
        "validation_event_roc_auc_by_alpha": {
            str(alpha): score for alpha, score in event_auc_by_alpha.items()
        },
        "test": _ranking_metrics(
            reward_splits[3],
            test_predictions[event_alpha][:, 0],
            test_predictions[signed_alpha][:, 1],
        ),
    }


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--decisions-per-seed", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs=4, default=(10, 11, 12, 13))
    parser.add_argument(
        "--atari-protocol", choices=tuple(ATARI_PROTOCOLS), default="published"
    )
    parser.add_argument("--zero-keep-probability", type=float, default=0.05)
    parser.add_argument("--dino-plan-cache")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.decisions_per_seed <= 0:
        parser.error("--decisions-per-seed must be positive")
    if not 0.0 < args.zero_keep_probability <= 1.0:
        parser.error("--zero-keep-probability must be in (0, 1]")
    if not hasattr(_native, "DinoPerception"):
        parser.error("rebuild the Kindle extension to enable the DINO probe")

    encoder = _native.DinoPerception(
        args.dino_checkpoint,
        dino_plan_cache=args.dino_plan_cache,
    )
    splits = [
        collect_split(
            encoder,
            args.environment,
            args.atari_protocol,
            seed,
            args.decisions_per_seed,
            args.zero_keep_probability,
        )
        for seed in args.seeds
    ]
    results = {}
    for representation in REPRESENTATIONS:
        print(f"fitting {representation} reward probe", flush=True)
        results[representation] = evaluate_representation(
            [features[representation] for features, _ in splits],
            [rewards for _, rewards in splits],
        )

    summary = {
        "environment": args.environment,
        "atari_protocol": args.atari_protocol,
        "decisions_per_seed": args.decisions_per_seed,
        "zero_keep_probability": args.zero_keep_probability,
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
