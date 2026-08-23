"""Learn pixel-token Pong control from a scalar intrinsic tracking reward.

The policy starts from random weights. A model-visible motion token defines a
tracking-consistency reward: +1 when the sampled action agrees with the current
ball-intercept direction and -1 otherwise. Reward magnitudes are balanced over
the four tracking decisions, but no controller action is executed and no label
is passed to Kindle's supervised API. Raw signed Pong points remain active.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_NUM_PONG_ACTIONS = 6
_TRACKING_ACTIONS = 4


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one evaluation seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("evaluation seeds must be unique")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--weights-dir", help="train and save a new checkpoint")
    mode.add_argument(
        "--load-weights-dir",
        help="skip training and evaluate a previously saved checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-lanes", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=5_000)
    parser.add_argument("--eval-lanes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=15_000)
    parser.add_argument("--eval-seeds", type=_parse_seeds, default=[42, 7, 123])
    parser.add_argument("--policy-lr", type=float, default=0.0002)
    parser.add_argument("--tracking-check-interval", type=int, default=100)
    parser.add_argument("--tracking-validation-per-class", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--require-best-tracking", type=float, default=0.99)
    parser.add_argument("--require-active-tracking", type=float, default=0.98)
    parser.add_argument("--require-positive-fraction", type=float, default=0.30)
    parser.add_argument("--require-return", type=float, default=-15.0)
    parser.add_argument("--require-episodes-per-seed", type=int, default=3)
    args = parser.parse_args()
    for name in (
        "train_lanes",
        "train_steps",
        "eval_lanes",
        "eval_steps",
        "tracking_check_interval",
        "tracking_validation_per_class",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    validation_rows = _TRACKING_ACTIONS * args.tracking_validation_per_class
    if validation_rows < args.train_lanes or validation_rows % args.train_lanes:
        parser.error(
            "four times --tracking-validation-per-class must be a positive "
            "multiple of --train-lanes"
        )
    if not np.isfinite(args.policy_lr) or args.policy_lr <= 0.0:
        parser.error("--policy-lr must be finite and positive")
    for name in (
        "require_best_tracking",
        "require_active_tracking",
        "require_positive_fraction",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.require_episodes_per_seed < 0:
        parser.error("--require-episodes-per-seed must be non-negative")
    return args


def _action_masks(kindle, lanes: int) -> np.ndarray:
    masks = np.zeros((lanes, kindle.MAX_ACTION_DIM), dtype=np.float32)
    masks[:, :_TRACKING_ACTIONS] = 1.0
    return masks.reshape(-1)


def _agent_config(args: argparse.Namespace) -> dict:
    return {
        "obs_dim": 64,
        "num_actions": _NUM_PONG_ACTIONS,
        "seed": args.seed,
        "latent_dim": 32,
        "hidden_dim": 64,
        "learning_rate": 0.0,
        "lr_policy": args.policy_lr,
        "end_to_end_encoder": True,
        # E2E requires n_step >= 2, but gamma=0 makes this a contextual
        # one-step objective instead of leaking the following state's target.
        "n_step": 2,
        "gamma": 0.0,
        "rollout_length": 2,
        "extrinsic_reward_alpha": 1.0,
        "advantage_normalize": True,
        "use_grpo": True,
        "use_grpo_episode": False,
        "use_ppo": False,
        "use_kl_ppo": False,
        "use_sil": False,
        "use_adam": True,
        "adam_eps": 1e-4,
        "grad_clip_norm": 0.5,
        "grad_clip_every": 1,
        # Entropy recovery erased the class-balanced reward signal in the
        # matched ablation, so the validated contextual objective leaves it off.
        "entropy_beta": 0.0,
        "entropy_floor": 0.0,
        "value_loss_coef": 0.0,
        "recon_loss_coef": 0.0,
        "reward_pred_loss_coef": 0.0,
    }


def _new_agent(kindle, config: dict, lanes: int):
    agent = kindle.BatchAgent(
        batch_size=lanes,
        env_ids=[0] * lanes,
        **config,
    )
    agent.set_action_masks(_action_masks(kindle, lanes))
    return agent


def _desired_tracking_actions(tokens: np.ndarray) -> np.ndarray:
    """Map current model-visible geometry to NOOP/FIRE/up/down decisions."""
    desired = np.zeros(len(tokens), dtype=np.int64)
    desired[tokens[:, 0] < 0.5] = 1
    desired[(tokens[:, 0] >= 0.5) & (tokens[:, 11] < -0.1)] = 2
    desired[(tokens[:, 0] >= 0.5) & (tokens[:, 11] > 0.1)] = 3
    return desired


def _balanced_tracking_rewards(
    desired: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Return class-balanced signed scalar rewards, never action labels."""
    counts = np.bincount(desired, minlength=_TRACKING_ACTIONS)
    weights = np.zeros(_TRACKING_ACTIONS, dtype=np.float32)
    active = counts > 0
    weights[active] = len(desired) / (_TRACKING_ACTIONS * counts[active])
    signs = np.where(actions == desired, 1.0, -1.0).astype(np.float32)
    return signs * weights[desired]


def _tracking_accuracy(
    agent,
    rows: list[np.ndarray],
    labels: list[int],
) -> float:
    lanes = agent.num_lanes()
    if len(rows) != len(labels) or not rows or len(rows) % lanes:
        raise ValueError("tracking validation rows must be a non-empty lane multiple")
    correct = 0
    for offset in range(0, len(rows), lanes):
        predictions = agent.act(
            [row.tolist() for row in rows[offset : offset + lanes]],
            deterministic=True,
        )
        correct += int(
            np.count_nonzero(np.asarray(predictions) == labels[offset : offset + lanes])
        )
    return correct / len(rows)


def _train(
    args: argparse.Namespace,
    kindle,
    config: dict,
    output_dir: Path,
) -> tuple[dict, np.ndarray, np.ndarray]:
    import gymnasium as gym
    from atari_batch import (
        FRAME_H,
        _make_env,
        _pong_motion_tokens,
        _transition_observations,
    )

    # Sticky actions make an observed paddle displacement belong to a previous
    # command and corrupt the intrinsic causal label. Evaluation restores 0.1.
    envs = gym.vector.SyncVectorEnv(
        [
            _make_env(
                "Pong", args.seed + lane, frame_skip=1, repeat_action_probability=0.0
            )
            for lane in range(args.train_lanes)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    observations, _info = envs.reset(seed=args.seed)
    agent = _new_agent(kindle, config, args.train_lanes)
    last_paddle_y = None
    positive_events = 0
    negative_events = 0
    action_counts = np.zeros(_NUM_PONG_ACTIONS, dtype=np.int64)
    desired_counts = np.zeros(_TRACKING_ACTIONS, dtype=np.int64)
    tracking_matches = 0
    tracking_rows = 0
    validation_by_class: dict[int, dict[bytes, np.ndarray]] = {
        action: {} for action in range(_TRACKING_ACTIONS)
    }
    validation_rows: list[np.ndarray] | None = None
    validation_labels: list[int] | None = None
    best_accuracy = -1.0
    best_step = 0
    best_params = None
    start = time.time()
    try:
        for step in range(args.train_steps):
            tokens, last_paddle_y = _pong_motion_tokens(observations, last_paddle_y)
            desired = _desired_tracking_actions(tokens)
            actions = np.asarray(agent.act(tokens.tolist()), dtype=np.int64)
            action_counts += np.bincount(actions, minlength=_NUM_PONG_ACTIONS)
            desired_counts += np.bincount(desired, minlength=_TRACKING_ACTIONS)
            tracking_matches += int(np.count_nonzero(actions == desired))
            tracking_rows += args.train_lanes

            if validation_rows is None:
                for token, target in zip(tokens, desired):
                    bucket = validation_by_class[int(target)]
                    if len(bucket) < args.tracking_validation_per_class:
                        bucket.setdefault(np.round(token, 4).tobytes(), token.copy())
                if all(
                    len(bucket) >= args.tracking_validation_per_class
                    for bucket in validation_by_class.values()
                ):
                    validation_rows = []
                    validation_labels = []
                    for target in range(_TRACKING_ACTIONS):
                        rows = list(validation_by_class[target].values())[
                            : args.tracking_validation_per_class
                        ]
                        validation_rows.extend(rows)
                        validation_labels.extend([target] * len(rows))

            next_observations, rewards, terminated, truncated, infos = envs.step(
                actions
            )
            positive_events += int(np.count_nonzero(rewards > 0.0))
            negative_events += int(np.count_nonzero(rewards < 0.0))
            done = np.logical_or(terminated, truncated)
            transition_observations = _transition_observations(
                next_observations,
                done,
                infos,
            )
            transition_tokens, _ = _pong_motion_tokens(
                transition_observations,
                last_paddle_y,
            )
            agent.set_extrinsic_reward(rewards.astype(np.float32, copy=False))
            agent.set_intrinsic_progress(_balanced_tracking_rewards(desired, actions))
            agent.observe(
                transition_tokens.tolist(),
                actions.tolist(),
                homeostatic=[[] for _ in range(args.train_lanes)],
            )
            for lane in np.flatnonzero(done):
                agent.mark_boundary(int(lane))
            observations = next_observations
            if np.any(done):
                last_paddle_y[done] = FRAME_H / 2.0

            if validation_rows is not None and step % args.tracking_check_interval == 0:
                accuracy = _tracking_accuracy(
                    agent,
                    validation_rows,
                    validation_labels,
                )
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_step = step
                    best_params = agent.dump_policy_params()
            if args.log_every and step > 0 and step % args.log_every == 0:
                print(
                    f"train_step={step} points=+{positive_events}/-{negative_events} "
                    f"sampled_tracking={tracking_matches / tracking_rows:.1%} "
                    f"best_tracking={best_accuracy:.1%}@{best_step}",
                    flush=True,
                )
    finally:
        envs.close()

    if best_params is None or validation_rows is None or validation_labels is None:
        raise RuntimeError("training never filled the balanced tracking validation set")
    loaded = agent.load_policy_params(best_params)
    if loaded != len(best_params):
        raise RuntimeError(
            f"restored {loaded}/{len(best_params)} best policy parameters"
        )
    agent.save_weights(str(output_dir))
    validation_x = np.asarray(validation_rows, dtype=np.float32)
    validation_y = np.asarray(validation_labels, dtype=np.int64)
    np.savez(
        output_dir / "pong_intrinsic_validation.npz",
        observations=validation_x,
        labels=validation_y,
    )
    metrics = {
        "steps": args.train_steps,
        "decisions": args.train_steps * args.train_lanes,
        "positive_events": positive_events,
        "negative_events": negative_events,
        "action_counts": action_counts.tolist(),
        "desired_counts": desired_counts.tolist(),
        "sampled_tracking_accuracy": tracking_matches / tracking_rows,
        "best_tracking_accuracy": best_accuracy,
        "best_tracking_step": best_step,
        "seconds": time.time() - start,
    }
    return metrics, validation_x, validation_y


def _evaluate_seed(
    args: argparse.Namespace,
    kindle,
    config: dict,
    weights_dir: Path,
    seed: int,
) -> dict:
    import gymnasium as gym
    from atari_batch import FRAME_H, _make_env, _pong_motion_tokens

    envs = gym.vector.SyncVectorEnv(
        [
            _make_env("Pong", seed + lane, frame_skip=1, repeat_action_probability=0.1)
            for lane in range(args.eval_lanes)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    observations, _info = envs.reset(seed=seed)
    agent = _new_agent(kindle, config, args.eval_lanes)
    agent.load_weights(str(weights_dir))
    last_paddle_y = None
    current_returns = np.zeros(args.eval_lanes, dtype=np.float64)
    completed_returns: list[float] = []
    positive_events = 0
    negative_events = 0
    action_counts = np.zeros(_NUM_PONG_ACTIONS, dtype=np.int64)
    tracking_matches = 0
    active_matches = 0
    active_rows = 0
    start = time.time()
    try:
        for step in range(args.eval_steps):
            tokens, last_paddle_y = _pong_motion_tokens(observations, last_paddle_y)
            desired = _desired_tracking_actions(tokens)
            actions = np.asarray(
                agent.act(tokens.tolist(), deterministic=True),
                dtype=np.int64,
            )
            action_counts += np.bincount(actions, minlength=_NUM_PONG_ACTIONS)
            tracking_matches += int(np.count_nonzero(actions == desired))
            active = tokens[:, 0] >= 0.5
            active_matches += int(np.count_nonzero((actions == desired) & active))
            active_rows += int(np.count_nonzero(active))
            observations, rewards, terminated, truncated, _infos = envs.step(actions)
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
                fraction = positive_events / max(1, positive_events + negative_events)
                print(
                    f"eval_seed={seed} step={step} episodes={len(completed_returns)} "
                    f"points=+{positive_events}/-{negative_events} "
                    f"positive_fraction={fraction:.1%}",
                    flush=True,
                )
    finally:
        envs.close()
    total_events = positive_events + negative_events
    return {
        "seed": seed,
        "decisions": args.eval_steps * args.eval_lanes,
        "episodes": len(completed_returns),
        "completed_return_sum": float(sum(completed_returns)),
        "mean_return": (
            float(np.mean(completed_returns)) if completed_returns else float("nan")
        ),
        "positive_events": positive_events,
        "negative_events": negative_events,
        "positive_fraction": positive_events / total_events if total_events else 0.0,
        "tracking_accuracy": tracking_matches / (args.eval_steps * args.eval_lanes),
        "active_tracking_accuracy": active_matches / max(1, active_rows),
        "action_counts": action_counts.tolist(),
        "seconds": time.time() - start,
    }


def _evaluate(
    args: argparse.Namespace,
    kindle,
    config: dict,
    weights_dir: Path,
) -> list[dict]:
    return [
        _evaluate_seed(args, kindle, config, weights_dir, seed)
        for seed in args.eval_seeds
    ]


def _aggregate(results: list[dict]) -> dict:
    episodes = sum(result["episodes"] for result in results)
    return_sum = sum(result["completed_return_sum"] for result in results)
    positive = sum(result["positive_events"] for result in results)
    negative = sum(result["negative_events"] for result in results)
    total_events = positive + negative
    return {
        "decisions": sum(result["decisions"] for result in results),
        "episodes": episodes,
        "mean_return": return_sum / episodes if episodes else float("nan"),
        "positive_events": positive,
        "negative_events": negative,
        "positive_fraction": positive / total_events if total_events else 0.0,
    }


def _gate(
    args: argparse.Namespace,
    training: dict,
    results: list[dict],
) -> list[tuple[bool, str]]:
    return [
        (
            training["best_tracking_accuracy"] >= args.require_best_tracking,
            f"balanced validation tracking >= {args.require_best_tracking:.1%}",
        ),
        (
            all(
                result["active_tracking_accuracy"] >= args.require_active_tracking
                for result in results
            ),
            f"every seed active tracking >= {args.require_active_tracking:.1%}",
        ),
        (
            all(
                result["positive_fraction"] >= args.require_positive_fraction
                for result in results
            ),
            f"every seed positive-point fraction >= {args.require_positive_fraction:.1%}",
        ),
        (
            all(
                result["episodes"] >= args.require_episodes_per_seed
                for result in results
            ),
            f"at least {args.require_episodes_per_seed} completed episodes per seed",
        ),
        (
            all(
                np.isfinite(result["mean_return"])
                and result["mean_return"] >= args.require_return
                for result in results
            ),
            f"every seed mean return >= {args.require_return:+.2f}",
        ),
        (
            all(result["action_counts"][4:] == [0, 0] for result in results),
            "redundant FIRE+motion actions remain masked",
        ),
    ]


def _print_results(results: list[dict]) -> None:
    for result in results:
        print(
            f"seed={result['seed']} episodes={result['episodes']} "
            f"return={result['mean_return']:+.2f} "
            f"points=+{result['positive_events']}/-{result['negative_events']} "
            f"positive_fraction={result['positive_fraction']:.1%} "
            f"active_tracking={result['active_tracking_accuracy']:.1%}"
        )
    aggregate = _aggregate(results)
    print(
        f"aggregate decisions={aggregate['decisions']} episodes={aggregate['episodes']} "
        f"return={aggregate['mean_return']:+.2f} "
        f"points=+{aggregate['positive_events']}/-{aggregate['negative_events']} "
        f"positive_fraction={aggregate['positive_fraction']:.1%}"
    )


def main() -> int:
    args = _parse_args()
    try:
        import kindle
    except ImportError as exc:
        print(f"missing kindle extension: {exc}", file=sys.stderr)
        return 1

    try:
        if args.weights_dir:
            weights_dir = Path(args.weights_dir)
            config = _agent_config(args)
            training, _validation_x, _validation_y = _train(
                args,
                kindle,
                config,
                weights_dir,
            )
        else:
            weights_dir = Path(args.load_weights_dir)
            metadata = json.loads(
                (weights_dir / "pong_intrinsic.json").read_text(encoding="utf-8")
            )
            expected_evaluation = metadata["evaluation_config"]
            actual_evaluation = {
                "lanes": args.eval_lanes,
                "steps": args.eval_steps,
                "seeds": args.eval_seeds,
                "repeat_action_probability": 0.1,
            }
            if actual_evaluation != expected_evaluation:
                raise ValueError(
                    "load-only evaluation settings must match pong_intrinsic.json: "
                    f"expected {expected_evaluation}, got {actual_evaluation}"
                )
            config = metadata["agent_config"]
            training = metadata["training"]
        results = _evaluate(args, kindle, config, weights_dir)
    except (KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"Pong intrinsic training failed: {exc}", file=sys.stderr)
        return 1

    _print_results(results)
    checks = _gate(args, training, results)
    metadata = {
        "format": "kindle-pong-intrinsic-v1",
        "agent_config": config,
        "training": training,
        "evaluation": results,
        "evaluation_config": {
            "lanes": args.eval_lanes,
            "steps": args.eval_steps,
            "seeds": args.eval_seeds,
            "repeat_action_probability": 0.1,
        },
    }
    metadata_name = (
        "pong_intrinsic.json" if args.weights_dir else "pong_intrinsic_reload.json"
    )
    (weights_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = [description for passed, description in checks if not passed]
    if failed:
        print(f"Pong intrinsic gate: FAIL ({'; '.join(failed)})", file=sys.stderr)
        return 2
    print("Pong intrinsic gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
