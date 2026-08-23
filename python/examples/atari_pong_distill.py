"""Distill the pixel-only Pong controller into a persisted Kindle policy.

The controller and learner consume the same 64-value ball/paddle motion token.
Training uses Kindle's direct masked supervised cross-entropy path: no synthetic
reward, value baseline, replay admission, or controller action is executed at
evaluation time. A separate load-only mode verifies the frozen checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLER_ACTIONS = 4
_NUM_PONG_ACTIONS = 6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--weights-dir", help="train and save a new checkpoint")
    mode.add_argument(
        "--load-weights-dir",
        help="skip collection/training and evaluate this checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-lanes", type=int, default=8)
    parser.add_argument("--collect-steps", type=int, default=3_000)
    parser.add_argument("--samples-per-action", type=int, default=256)
    parser.add_argument("--train-samples-per-action", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--policy-check-interval", type=int, default=25)
    parser.add_argument("--policy-lr", type=float, default=0.0002)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--repeat-action-probability", type=float, default=0.1)
    parser.add_argument("--eval-lanes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=15_000)
    parser.add_argument("--recent-episodes", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=3_000)
    parser.add_argument("--require-train-accuracy", type=float, default=1.0)
    parser.add_argument("--require-validation-accuracy", type=float, default=0.98)
    parser.add_argument("--require-recent-return", type=float, default=-15.0)
    parser.add_argument("--require-min-episodes", type=int, default=4)
    args = parser.parse_args()
    for name in (
        "collect_lanes",
        "collect_steps",
        "samples_per_action",
        "train_samples_per_action",
        "epochs",
        "policy_check_interval",
        "eval_lanes",
        "eval_steps",
        "recent_episodes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_samples_per_action >= args.samples_per_action:
        parser.error("--train-samples-per-action must be below --samples-per-action")
    if not np.isfinite(args.policy_lr) or args.policy_lr <= 0.0:
        parser.error("--policy-lr must be finite and positive")
    if not 0.0 <= args.repeat_action_probability <= 1.0:
        parser.error("--repeat-action-probability must be in [0, 1]")
    for name in ("require_train_accuracy", "require_validation_accuracy"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.require_min_episodes < 0:
        parser.error("--require-min-episodes must be non-negative")
    return args


def _balanced_split(
    unique_by_action: dict[int, dict[bytes, np.ndarray]],
    samples_per_action: int,
    train_per_action: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, action-balanced train/validation arrays."""
    rng = np.random.default_rng(seed)
    train: list[tuple[np.ndarray, int]] = []
    validation: list[tuple[np.ndarray, int]] = []
    for action in range(_CONTROLLER_ACTIONS):
        values = list(unique_by_action.get(action, {}).values())
        if len(values) < samples_per_action:
            raise ValueError(
                f"action {action} has {len(values)} unique states; "
                f"need {samples_per_action}"
            )
        rng.shuffle(values)
        chosen = values[:samples_per_action]
        train.extend((token, action) for token in chosen[:train_per_action])
        validation.extend((token, action) for token in chosen[train_per_action:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return (
        np.asarray([token for token, _action in train], dtype=np.float32),
        np.asarray([action for _token, action in train], dtype=np.int64),
        np.asarray([token for token, _action in validation], dtype=np.float32),
        np.asarray([action for _token, action in validation], dtype=np.int64),
    )


def _collect_controller_dataset(args: argparse.Namespace) -> tuple:
    import gymnasium as gym
    from atari_batch import (
        FRAME_H,
        _make_env,
        _pong_heuristic_actions,
        _pong_motion_tokens,
    )

    envs = gym.vector.SyncVectorEnv(
        [
            _make_env(
                "Pong",
                args.seed + lane,
                args.frame_skip,
                args.repeat_action_probability,
            )
            for lane in range(args.collect_lanes)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    observations, _info = envs.reset(seed=args.seed)
    last_paddle_y = None
    unique_by_action: dict[int, dict[bytes, np.ndarray]] = {
        action: {} for action in range(_CONTROLLER_ACTIONS)
    }
    action_counts = np.zeros(_NUM_PONG_ACTIONS, dtype=np.int64)
    positive_events = 0
    negative_events = 0
    try:
        for _step in range(args.collect_steps):
            tokens, token_paddle_y = _pong_motion_tokens(
                observations,
                last_paddle_y,
            )
            actions, controller_paddle_y = _pong_heuristic_actions(
                observations,
                last_paddle_y,
                deadband=1.0,
            )
            if not np.allclose(token_paddle_y, controller_paddle_y):
                raise RuntimeError("controller and token paddle state diverged")
            last_paddle_y = controller_paddle_y
            for token, action in zip(tokens, actions):
                action_counts[action] += 1
                key = np.round(token, 4).tobytes()
                unique_by_action[action].setdefault(key, token.copy())
            observations, rewards, terminated, truncated, _infos = envs.step(
                np.asarray(actions, dtype=np.int64)
            )
            positive_events += int(np.count_nonzero(rewards > 0.0))
            negative_events += int(np.count_nonzero(rewards < 0.0))
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                last_paddle_y[done] = FRAME_H / 2.0
    finally:
        envs.close()

    split = _balanced_split(
        unique_by_action,
        args.samples_per_action,
        args.train_samples_per_action,
        args.seed,
    )
    stats = {
        "action_counts": action_counts.tolist(),
        "unique_counts": [
            len(unique_by_action[action]) for action in range(_CONTROLLER_ACTIONS)
        ],
        "positive_events": positive_events,
        "negative_events": negative_events,
    }
    return (*split, stats)


def _common_agent_config(args: argparse.Namespace) -> dict:
    return {
        "obs_dim": 64,
        "num_actions": _NUM_PONG_ACTIONS,
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "learning_rate": 0.0,
        "lr_policy": args.policy_lr,
        "end_to_end_encoder": True,
        "n_step": 2,
        "rollout_length": 2,
        "value_loss_coef": 0.0,
        "recon_loss_coef": 0.0,
        "reward_pred_loss_coef": 0.0,
        "entropy_beta": 0.0,
        "entropy_floor": 0.0,
    }


def _action_masks(kindle, rows: int) -> np.ndarray:
    masks = np.zeros((rows, kindle.MAX_ACTION_DIM), dtype=np.float32)
    masks[:, :_NUM_PONG_ACTIONS] = 1.0
    return masks.reshape(-1)


def _accuracy(predictions, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == labels))


def _train(
    args: argparse.Namespace,
    kindle,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[dict, dict]:
    common = _common_agent_config(args)
    agent = kindle.BatchAgent(
        batch_size=len(train_x),
        env_ids=[0] * len(train_x),
        **common,
    )
    masks = _action_masks(kindle, len(train_x))
    agent.set_action_masks(masks)
    observations = train_x.tolist()
    labels = train_y.tolist()
    initial_accuracy = _accuracy(
        agent.act(observations, deterministic=True),
        train_y,
    )
    training_accuracy = initial_accuracy
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        agent.train_policy_supervised(observations, labels)
        if epoch % args.policy_check_interval == 0:
            agent.set_action_masks(masks)
            training_accuracy = _accuracy(
                agent.act(observations, deterministic=True),
                train_y,
            )
            if training_accuracy >= args.require_train_accuracy:
                break

    weights_dir = Path(args.weights_dir)
    agent.save_weights(str(weights_dir))
    np.savez(
        weights_dir / "pong_validation.npz",
        observations=validation_x,
        labels=validation_y,
    )

    validation_agent = kindle.BatchAgent(
        batch_size=len(validation_x),
        env_ids=[0] * len(validation_x),
        **common,
    )
    validation_agent.set_action_masks(_action_masks(kindle, len(validation_x)))
    validation_agent.load_weights(str(weights_dir))
    validation_accuracy = _accuracy(
        validation_agent.act(validation_x.tolist(), deterministic=True),
        validation_y,
    )
    metrics = {
        "epochs": epochs_run,
        "initial_accuracy": initial_accuracy,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
    }
    return common, metrics


def _load(args: argparse.Namespace, kindle) -> tuple[dict, dict]:
    weights_dir = Path(args.load_weights_dir)
    metadata = json.loads(
        (weights_dir / "pong_distill.json").read_text(encoding="utf-8")
    )
    common = metadata["agent_config"]
    with np.load(weights_dir / "pong_validation.npz", allow_pickle=False) as data:
        validation_x = np.asarray(data["observations"], dtype=np.float32)
        validation_y = np.asarray(data["labels"], dtype=np.int64)
    validation_agent = kindle.BatchAgent(
        batch_size=len(validation_x),
        env_ids=[0] * len(validation_x),
        **common,
    )
    validation_agent.set_action_masks(_action_masks(kindle, len(validation_x)))
    validation_agent.load_weights(str(weights_dir))
    validation_accuracy = _accuracy(
        validation_agent.act(validation_x.tolist(), deterministic=True),
        validation_y,
    )
    return common, {
        "epochs": 0,
        "initial_accuracy": validation_accuracy,
        "training_accuracy": metadata["training"]["training_accuracy"],
        "validation_accuracy": validation_accuracy,
    }


def _evaluate_gameplay(
    args: argparse.Namespace,
    kindle,
    common: dict,
    weights_dir: Path,
) -> dict:
    import gymnasium as gym
    from atari_batch import FRAME_H, _make_env, _pong_motion_tokens

    envs = gym.vector.SyncVectorEnv(
        [
            _make_env(
                "Pong",
                args.seed + lane,
                args.frame_skip,
                args.repeat_action_probability,
            )
            for lane in range(args.eval_lanes)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    observations, _info = envs.reset(seed=args.seed)
    agent = kindle.BatchAgent(
        batch_size=args.eval_lanes,
        env_ids=[0] * args.eval_lanes,
        **common,
    )
    agent.set_action_masks(_action_masks(kindle, args.eval_lanes))
    agent.load_weights(str(weights_dir))
    last_paddle_y = None
    current_returns = np.zeros(args.eval_lanes, dtype=np.float64)
    completed_returns: list[float] = []
    action_counts = np.zeros(_NUM_PONG_ACTIONS, dtype=np.int64)
    positive_events = 0
    negative_events = 0
    start = time.time()
    try:
        for step in range(args.eval_steps):
            tokens, last_paddle_y = _pong_motion_tokens(
                observations,
                last_paddle_y,
            )
            actions = agent.act(tokens.tolist(), deterministic=True)
            action_counts += np.bincount(actions, minlength=_NUM_PONG_ACTIONS)
            observations, rewards, terminated, truncated, _infos = envs.step(
                np.asarray(actions, dtype=np.int64)
            )
            current_returns += rewards
            positive_events += int(np.count_nonzero(rewards > 0.0))
            negative_events += int(np.count_nonzero(rewards < 0.0))
            done = np.logical_or(terminated, truncated)
            for lane in np.flatnonzero(done):
                completed_returns.append(float(current_returns[lane]))
                current_returns[lane] = 0.0
            if np.any(done):
                last_paddle_y[done] = FRAME_H / 2.0
            if args.log_every and step > 0 and step % args.log_every == 0:
                recent = completed_returns[-args.recent_episodes :]
                recent_mean = float(np.mean(recent)) if recent else float("nan")
                print(
                    f"eval_step={step} episodes={len(completed_returns)} "
                    f"events=+{positive_events}/-{negative_events} "
                    f"recent_return={recent_mean:+.2f}",
                    flush=True,
                )
    finally:
        envs.close()
    elapsed = time.time() - start
    recent = completed_returns[-args.recent_episodes :]
    return {
        "decisions": args.eval_steps * args.eval_lanes,
        "episodes": len(completed_returns),
        "mean_return": (
            float(np.mean(completed_returns)) if completed_returns else float("nan")
        ),
        "recent_return": float(np.mean(recent)) if recent else float("nan"),
        "positive_events": positive_events,
        "negative_events": negative_events,
        "action_counts": action_counts.tolist(),
        "seconds": elapsed,
    }


def main() -> int:
    args = _parse_args()
    try:
        import kindle
    except ImportError as exc:
        print(f"missing kindle extension: {exc}", file=sys.stderr)
        return 1

    collection = None
    try:
        if args.weights_dir:
            train_x, train_y, validation_x, validation_y, collection = (
                _collect_controller_dataset(args)
            )
            common, training = _train(
                args,
                kindle,
                train_x,
                train_y,
                validation_x,
                validation_y,
            )
            weights_dir = Path(args.weights_dir)
        else:
            common, training = _load(args, kindle)
            weights_dir = Path(args.load_weights_dir)
        gameplay = _evaluate_gameplay(args, kindle, common, weights_dir)
    except (KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"Pong distillation failed: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "format": "kindle-pong-distill-v1",
        "agent_config": common,
        "collection": collection,
        "training": training,
        "gameplay": gameplay,
        "frame_skip": args.frame_skip,
        "repeat_action_probability": args.repeat_action_probability,
        "seed": args.seed,
    }
    metadata_name = "pong_distill.json" if args.weights_dir else "pong_reload.json"
    (weights_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"training_accuracy={training['training_accuracy']:.1%} "
        f"validation_accuracy={training['validation_accuracy']:.1%} "
        f"epochs={training['epochs']}"
    )
    print(
        f"decisions={gameplay['decisions']} episodes={gameplay['episodes']} "
        f"return={gameplay['recent_return']:+.2f} "
        f"events=+{gameplay['positive_events']}/-{gameplay['negative_events']} "
        f"actions={gameplay['action_counts']}"
    )

    checks = [
        (
            training["training_accuracy"] >= args.require_train_accuracy,
            "training accuracy",
        ),
        (
            training["validation_accuracy"] >= args.require_validation_accuracy,
            "validation accuracy",
        ),
        (
            gameplay["episodes"] >= args.require_min_episodes,
            "completed episodes",
        ),
        (
            np.isfinite(gameplay["recent_return"])
            and gameplay["recent_return"] >= args.require_recent_return,
            "recent return",
        ),
    ]
    failed = [name for passed, name in checks if not passed]
    if failed:
        print(f"gate failed: {', '.join(failed)}", file=sys.stderr)
        return 2
    print("Pong distillation gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
