"""Continue a distilled Pong policy using only online extrinsic rewards.

Training loads the supervised checkpoint produced by ``atari_pong_distill.py``,
freezes its motion encoder, and updates the policy head with same-task GRPO.
No controller actions or labels are used after initialization. Evaluation uses
fresh, deterministic, zero-update agents on matched source/fine-tuned seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_NUM_PONG_ACTIONS = 6


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
    mode.add_argument("--weights-dir", help="train and save a fine-tuned checkpoint")
    mode.add_argument(
        "--load-weights-dir",
        help="skip training and evaluate a previously fine-tuned checkpoint",
    )
    parser.add_argument(
        "--source-weights-dir",
        help="distilled checkpoint to continue; required with --weights-dir",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-lanes", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=5_000)
    parser.add_argument("--eval-lanes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=15_000)
    parser.add_argument("--eval-seeds", type=_parse_seeds, default=[42, 7, 123])
    parser.add_argument("--policy-lr", type=float, default=2e-5)
    parser.add_argument("--n-step", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=8)
    parser.add_argument("--encoder-lr-mul", type=float, default=0.0)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--repeat-action-probability", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=3_000)
    parser.add_argument("--require-deficit-improvement", type=float, default=0.2)
    parser.add_argument("--require-event-retention", type=float, default=0.75)
    parser.add_argument("--require-return", type=float, default=-15.0)
    parser.add_argument("--require-episodes-per-seed", type=int, default=1)
    args = parser.parse_args()
    if args.weights_dir and not args.source_weights_dir:
        parser.error("--source-weights-dir is required with --weights-dir")
    if args.load_weights_dir and args.source_weights_dir:
        parser.error(
            "--source-weights-dir is read from fine-tune metadata in load mode"
        )
    if (
        args.weights_dir
        and Path(args.weights_dir).resolve() == Path(args.source_weights_dir).resolve()
    ):
        parser.error("--weights-dir must not overwrite --source-weights-dir")
    for name in (
        "train_lanes",
        "train_steps",
        "eval_lanes",
        "eval_steps",
        "n_step",
        "rollout_length",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.policy_lr) or args.policy_lr <= 0.0:
        parser.error("--policy-lr must be finite and positive")
    if not np.isfinite(args.encoder_lr_mul) or args.encoder_lr_mul < 0.0:
        parser.error("--encoder-lr-mul must be finite and non-negative")
    if not 0.0 <= args.repeat_action_probability <= 1.0:
        parser.error("--repeat-action-probability must be in [0, 1]")
    if not 0.0 <= args.require_deficit_improvement <= 1.0:
        parser.error("--require-deficit-improvement must be in [0, 1]")
    if not 0.0 <= args.require_event_retention <= 1.0:
        parser.error("--require-event-retention must be in [0, 1]")
    if args.require_episodes_per_seed < 0:
        parser.error("--require-episodes-per-seed must be non-negative")
    return args


def _action_masks(kindle, lanes: int) -> np.ndarray:
    masks = np.zeros((lanes, kindle.MAX_ACTION_DIM), dtype=np.float32)
    masks[:, :_NUM_PONG_ACTIONS] = 1.0
    return masks.reshape(-1)


def _training_config(source_config: dict, args: argparse.Namespace) -> dict:
    config = dict(source_config)
    config.update(
        seed=args.seed,
        lr_policy=args.policy_lr,
        n_step=args.n_step,
        rollout_length=args.rollout_length,
        extrinsic_reward_alpha=1.0,
        advantage_normalize=True,
        use_grpo=True,
        use_grpo_episode=False,
        use_ppo=False,
        use_kl_ppo=False,
        use_sil=False,
        use_adam=True,
        adam_eps=1e-4,
        grad_clip_norm=0.5,
        grad_clip_every=1,
        entropy_beta=0.0,
        entropy_floor=0.0,
        value_loss_coef=0.0,
        recon_loss_coef=0.0,
        reward_pred_loss_coef=0.0,
    )
    return config


def _new_agent(kindle, config: dict, lanes: int):
    agent = kindle.BatchAgent(
        batch_size=lanes,
        env_ids=[0] * lanes,
        **config,
    )
    agent.set_action_masks(_action_masks(kindle, lanes))
    return agent


def _parameter_change(before: dict[str, list[float]], after: dict[str, list[float]]):
    changes: dict[str, dict[str, float]] = {}
    for prefix in ("policy_encoder.", "policy."):
        total = 0.0
        maximum = 0.0
        count = 0
        for name, old_values in before.items():
            if not name.startswith(prefix):
                continue
            new_values = np.asarray(after[name], dtype=np.float32)
            delta = np.abs(new_values - np.asarray(old_values, dtype=np.float32))
            total += float(delta.sum(dtype=np.float64))
            maximum = max(maximum, float(delta.max(initial=0.0)))
            count += int(delta.size)
        changes[prefix] = {"l1": total, "max": maximum, "parameters": count}
    return changes


def _train_reward_only(
    args: argparse.Namespace,
    kindle,
    config: dict,
    source_dir: Path,
    output_dir: Path,
) -> tuple[dict, dict]:
    import gymnasium as gym
    from atari_batch import (
        FRAME_H,
        _make_env,
        _pong_motion_tokens,
        _transition_observations,
    )

    envs = gym.vector.SyncVectorEnv(
        [
            _make_env(
                "Pong",
                args.seed + lane,
                args.frame_skip,
                args.repeat_action_probability,
            )
            for lane in range(args.train_lanes)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    observations, _info = envs.reset(seed=args.seed)
    agent = _new_agent(kindle, config, args.train_lanes)
    agent.set_policy_lr_multiplier("policy_encoder.", args.encoder_lr_mul)
    agent.load_weights(str(source_dir))
    before = dict(agent.dump_policy_params())
    last_paddle_y = None
    current_returns = np.zeros(args.train_lanes, dtype=np.float64)
    completed_returns: list[float] = []
    action_counts = np.zeros(_NUM_PONG_ACTIONS, dtype=np.int64)
    positive_events = 0
    negative_events = 0
    start = time.time()
    try:
        for step in range(args.train_steps):
            tokens, last_paddle_y = _pong_motion_tokens(
                observations,
                last_paddle_y,
            )
            actions = agent.act(tokens.tolist())
            action_counts += np.bincount(actions, minlength=_NUM_PONG_ACTIONS)
            next_observations, rewards, terminated, truncated, infos = envs.step(
                np.asarray(actions, dtype=np.int64)
            )
            current_returns += rewards
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
            agent.observe(
                transition_tokens.tolist(),
                [int(action) for action in actions],
                homeostatic=[[] for _ in range(args.train_lanes)],
            )
            for lane in np.flatnonzero(done):
                completed_returns.append(float(current_returns[lane]))
                current_returns[lane] = 0.0
                agent.mark_boundary(int(lane))
            observations = next_observations
            if np.any(done):
                last_paddle_y[done] = FRAME_H / 2.0
            if args.log_every and step > 0 and step % args.log_every == 0:
                print(
                    f"train_step={step} episodes={len(completed_returns)} "
                    f"events=+{positive_events}/-{negative_events}",
                    flush=True,
                )
    finally:
        envs.close()
    after = dict(agent.dump_policy_params())
    changes = _parameter_change(before, after)
    agent.save_weights(str(output_dir))
    metrics = {
        "steps": args.train_steps,
        "decisions": args.train_steps * args.train_lanes,
        "episodes": len(completed_returns),
        "completed_return_sum": float(sum(completed_returns)),
        "positive_events": positive_events,
        "negative_events": negative_events,
        "action_counts": action_counts.tolist(),
        "seconds": time.time() - start,
    }
    return metrics, changes


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
            _make_env(
                "Pong",
                seed + lane,
                args.frame_skip,
                args.repeat_action_probability,
            )
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
                print(
                    f"eval_seed={seed} step={step} "
                    f"episodes={len(completed_returns)} "
                    f"events=+{positive_events}/-{negative_events}",
                    flush=True,
                )
    finally:
        envs.close()
    total_events = positive_events + negative_events
    return {
        "seed": seed,
        "steps": args.eval_steps,
        "decisions": args.eval_steps * args.eval_lanes,
        "episodes": len(completed_returns),
        "completed_return_sum": float(sum(completed_returns)),
        "mean_return": (
            float(np.mean(completed_returns)) if completed_returns else float("nan")
        ),
        "positive_events": positive_events,
        "negative_events": negative_events,
        "point_deficit": negative_events - positive_events,
        "positive_point_fraction": (
            positive_events / total_events if total_events else float("nan")
        ),
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
        "seeds": len(results),
        "decisions": sum(result["decisions"] for result in results),
        "episodes": episodes,
        "mean_return": return_sum / episodes if episodes else float("nan"),
        "positive_events": positive,
        "negative_events": negative,
        "point_deficit": negative - positive,
        "positive_point_fraction": positive / total_events
        if total_events
        else float("nan"),
    }


def _gate(
    args: argparse.Namespace,
    source_results: list[dict],
    fine_results: list[dict],
    changes: dict,
) -> tuple[list[tuple[bool, str]], float]:
    source_by_seed = {result["seed"]: result for result in source_results}
    fine_by_seed = {result["seed"]: result for result in fine_results}
    matching_seeds = source_by_seed.keys() == fine_by_seed.keys()
    source_aggregate = _aggregate(source_results)
    fine_aggregate = _aggregate(fine_results)
    source_deficit = source_aggregate["point_deficit"]
    fine_deficit = fine_aggregate["point_deficit"]
    source_events = (
        source_aggregate["positive_events"] + source_aggregate["negative_events"]
    )
    fine_events = fine_aggregate["positive_events"] + fine_aggregate["negative_events"]
    event_retention = fine_events / source_events if source_events else 0.0
    improvement = (
        (source_deficit - fine_deficit) / source_deficit
        if source_deficit > 0
        else float("-inf")
    )
    checks: list[tuple[bool, str]] = [
        (
            improvement >= args.require_deficit_improvement,
            (
                f"aggregate point-deficit improvement {improvement:.1%} "
                f">= {args.require_deficit_improvement:.1%}"
            ),
        ),
        (
            matching_seeds
            and all(
                fine_by_seed[seed]["point_deficit"]
                < source_by_seed[seed]["point_deficit"]
                for seed in source_by_seed
            ),
            "point deficit improves on every evaluation seed",
        ),
        (
            matching_seeds
            and all(
                fine_by_seed[seed]["positive_point_fraction"]
                > source_by_seed[seed]["positive_point_fraction"]
                for seed in source_by_seed
            ),
            "positive point fraction improves on every evaluation seed",
        ),
        (
            event_retention >= args.require_event_retention,
            (
                f"event count retention {event_retention:.1%} "
                f">= {args.require_event_retention:.1%}"
            ),
        ),
        (
            all(
                result["episodes"] >= args.require_episodes_per_seed
                for result in fine_results
            ),
            f"at least {args.require_episodes_per_seed} completed episode(s) per seed",
        ),
        (
            all(
                np.isfinite(result["mean_return"])
                and result["mean_return"] >= args.require_return
                for result in fine_results
            ),
            f"every seed mean return >= {args.require_return:+.2f}",
        ),
        (
            changes["policy_encoder."]["max"] <= 1e-7,
            "frozen policy encoder is unchanged",
        ),
        (
            changes["policy."]["l1"] > 0.0,
            "policy head changed under extrinsic reward",
        ),
    ]
    return checks, improvement


def _print_evaluation(label: str, results: list[dict]) -> None:
    for result in results:
        print(
            f"{label} seed={result['seed']} episodes={result['episodes']} "
            f"return={result['mean_return']:+.2f} "
            f"events=+{result['positive_events']}/-{result['negative_events']} "
            f"deficit={result['point_deficit']}"
        )
    aggregate = _aggregate(results)
    print(
        f"{label} aggregate decisions={aggregate['decisions']} "
        f"episodes={aggregate['episodes']} return={aggregate['mean_return']:+.2f} "
        f"events=+{aggregate['positive_events']}/-{aggregate['negative_events']} "
        f"deficit={aggregate['point_deficit']}"
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
            output_dir = Path(args.weights_dir)
            source_dir = Path(args.source_weights_dir)
            source_metadata = json.loads(
                (source_dir / "pong_distill.json").read_text(encoding="utf-8")
            )
            config = _training_config(source_metadata["agent_config"], args)
            print("evaluating distilled source checkpoint", flush=True)
            source_results = _evaluate(args, kindle, config, source_dir)
            print("running reward-only fine-tuning", flush=True)
            training, changes = _train_reward_only(
                args,
                kindle,
                config,
                source_dir,
                output_dir,
            )
            fine_results = _evaluate(args, kindle, config, output_dir)
        else:
            output_dir = Path(args.load_weights_dir)
            metadata = json.loads(
                (output_dir / "pong_finetune.json").read_text(encoding="utf-8")
            )
            expected_evaluation = metadata["evaluation_config"]
            actual_evaluation = {
                "lanes": args.eval_lanes,
                "steps": args.eval_steps,
                "seeds": args.eval_seeds,
                "frame_skip": args.frame_skip,
                "repeat_action_probability": args.repeat_action_probability,
            }
            if actual_evaluation != expected_evaluation:
                raise ValueError(
                    "load-only evaluation settings must match pong_finetune.json: "
                    f"expected {expected_evaluation}, got {actual_evaluation}"
                )
            source_dir = Path(metadata["source_weights_dir"])
            config = metadata["agent_config"]
            source_results = metadata["source_evaluation"]
            training = metadata["training"]
            changes = metadata["parameter_change"]
            fine_results = _evaluate(args, kindle, config, output_dir)
    except (KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"Pong fine-tuning failed: {exc}", file=sys.stderr)
        return 1

    checks, improvement = _gate(
        args,
        source_results,
        fine_results,
        changes,
    )
    _print_evaluation("source", source_results)
    _print_evaluation("fine", fine_results)
    print(
        f"parameter_change encoder_max={changes['policy_encoder.']['max']:.3g} "
        f"policy_l1={changes['policy.']['l1']:.6f} "
        f"deficit_improvement={improvement:.1%}"
    )
    metadata = {
        "format": "kindle-pong-finetune-v1",
        "source_weights_dir": str(source_dir),
        "agent_config": config,
        "training": training,
        "parameter_change": changes,
        "source_evaluation": source_results,
        "fine_evaluation": fine_results,
        "evaluation_config": {
            "lanes": args.eval_lanes,
            "steps": args.eval_steps,
            "seeds": args.eval_seeds,
            "frame_skip": args.frame_skip,
            "repeat_action_probability": args.repeat_action_probability,
        },
        "deficit_improvement": improvement,
    }
    metadata_name = (
        "pong_finetune.json" if args.weights_dir else "pong_finetune_reload.json"
    )
    (output_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = [description for passed, description in checks if not passed]
    if failed:
        print(f"Pong fine-tuning gate: FAIL ({'; '.join(failed)})", file=sys.stderr)
        return 2
    print("Pong fine-tuning gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
