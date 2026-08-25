"""Atari-100k-style training loop for Kindle.

Usage:
    python examples/atari.py /path/to/model.safetensors [ALE/Pong-v5]
"""

import argparse
import json
import random
import sys
import time

import ale_py
import gymnasium as gym

import kindle


MODEL_SIZES = ("1m", "12m", "25m", "50m", "100m", "200m")
MEAN_DINO_RECONSTRUCTION_SCALE = 1.0 / (7 * 7 * 64)


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="12m")
    parser.add_argument(
        "--observation-decoder-depth",
        type=int,
        help="per-patch DINO decoder width (default: 64; 0 restores legacy preset width)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=256.0,
        help="replayed samples per environment step (D3 Atari-100k: 256)",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--actor-learning-starts",
        type=int,
        help="delay actor parameter updates for this many learner updates; critic training is unchanged",
    )
    parser.add_argument("--learning-rate-warmup", type=int)
    parser.add_argument("--free-nats", type=float)
    parser.add_argument(
        "--dynamics-free-nats",
        type=float,
        help="override only the prior/dynamics KL floor",
    )
    parser.add_argument(
        "--dynamics-loss-scale",
        type=float,
        help="override the prior/dynamics KL loss coefficient",
    )
    parser.add_argument(
        "--reconstruction-loss-scale",
        type=float,
        default=MEAN_DINO_RECONSTRUCTION_SCALE,
        help="frozen-DINO candidate uses a per-feature mean (1 / 3136)",
    )
    parser.add_argument(
        "--replay-value-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let replay-value loss shape the RSSM (D3 default; disabled by the frozen-DINO candidate)",
    )
    parser.add_argument(
        "--random-action-steps",
        type=int,
        default=0,
        help="force uniformly random actions for the first N interactions",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="run a deterministic random control without constructing Dreamer",
    )
    parser.add_argument("--restore", help="restore a Dreamer checkpoint")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run the restored policy greedily without learning",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", help="write JSONL metrics to this file")
    parser.add_argument("--report-every", type=int, default=1_000)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    if args.random_action_steps < 0:
        parser.error("--random-action-steps must be non-negative")
    if args.evaluate and not args.restore:
        parser.error("--evaluate requires --restore")
    if args.evaluate and args.random_action_steps:
        parser.error("--evaluate cannot be combined with --random-action-steps")
    if args.random_policy and (args.restore or args.evaluate or args.checkpoint):
        parser.error("--random-policy cannot use Dreamer checkpoints or evaluation")

    # Match D3's Atari-100k interaction protocol: minimal action set, four
    # repeated frames, max-pooling over the last two, no sticky actions, and
    # up to 30 random no-op frames after reset. Gymnasium's wrapper uses its
    # own OpenCV resize rather than D3's Pillow resize.
    environment = gym.make(
        args.environment,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    environment = gym.wrappers.AtariPreprocessing(
        environment,
        noop_max=30,
        frame_skip=4,
        screen_size=64,
        terminal_on_life_loss=False,
        grayscale_obs=False,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    frame, _ = environment.reset(seed=args.seed)
    if getattr(frame, "ndim", 0) != 3:
        raise ValueError("the environment must return H×W×3 RGB observations")
    action_count = int(environment.action_space.n)
    if args.random_policy:
        agent = None
    elif args.restore:
        agent = kindle.Agent.restore(args.restore, args.dino_checkpoint)
        restored_actions = int(agent.config["action_count"])
        if restored_actions != action_count:
            raise ValueError(
                f"checkpoint has {restored_actions} actions, environment has {action_count}"
            )
    else:
        agent = kindle.Agent(
            args.dino_checkpoint,
            action_count,
            model_size=args.model_size,
            observation_decoder_depth=args.observation_decoder_depth,
            seed=args.seed,
            train_ratio=args.train_ratio,
            learning_rate=args.learning_rate,
            actor_learning_starts=args.actor_learning_starts,
            learning_rate_warmup=args.learning_rate_warmup,
            free_nats=args.free_nats,
            dynamics_free_nats=args.dynamics_free_nats,
            dynamics_loss_scale=args.dynamics_loss_scale,
            reconstruction_loss_scale=args.reconstruction_loss_scale,
            replay_value_gradient=args.replay_value_gradient,
        )
    if agent is not None:
        agent.begin_episode(frame)
    random_actions = random.Random(args.seed ^ 0xA7A2_1000)
    run_mode = (
        "random" if args.random_policy else "evaluate" if args.evaluate else "train"
    )

    output = (
        open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    )

    def emit(event: dict[str, object]) -> None:
        print(json.dumps(event, separators=(",", ":")), file=output, flush=True)

    started = time.perf_counter()
    total_reward = 0.0
    episode_return = 0.0
    episode_length = 0
    completed_return_sum = 0.0
    episodes = 0
    interval_reward = 0.0
    interval_episodes = 0
    interval_updates = 0
    action_counts = [0] * action_count
    interval_action_counts = [0] * action_count
    emit(
        {
            "event": "run_start",
            "environment": args.environment,
            "steps": args.steps,
            "seed": args.seed,
            "actions": action_count,
            "mode": run_mode,
            "config": agent.config if agent is not None else None,
        }
    )

    try:
        for run_step in range(1, args.steps + 1):
            if args.random_policy:
                action = random_actions.randrange(action_count)
            elif run_step <= args.random_action_steps:
                forced_action = random_actions.randrange(action_count)
                action_mask = [False] * action_count
                action_mask[forced_action] = True
                assert agent is not None
                action = agent.act(action_mask=action_mask)
                assert action == forced_action
            else:
                assert agent is not None
                action = agent.act(greedy=args.evaluate)
            frame, reward, terminated, truncated, _ = environment.step(action)
            reward = float(reward)
            terminated = bool(terminated)
            truncated = bool(truncated)
            if agent is not None:
                agent.observe(
                    frame,
                    extrinsic_reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            reports = [] if agent is None or args.evaluate else agent.learn_scheduled()
            interval_updates += len(reports)
            for report in reports:
                emit(
                    {
                        "event": "learner",
                        "run_step": run_step,
                        "elapsed_seconds": time.perf_counter() - started,
                        "report": report,
                    }
                )

            total_reward += reward
            interval_reward += reward
            episode_return += reward
            episode_length += 1
            action_counts[action] += 1
            interval_action_counts[action] += 1
            if terminated or truncated:
                episodes += 1
                interval_episodes += 1
                completed_return_sum += episode_return
                emit(
                    {
                        "event": "episode",
                        "run_step": run_step,
                        "return": episode_return,
                        "length": episode_length,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                frame, _ = environment.reset()
                if agent is not None:
                    agent.begin_episode(frame)
                episode_return = 0.0
                episode_length = 0

            if run_step % args.report_every == 0:
                emit(
                    {
                        "event": "interval",
                        "run_step": run_step,
                        "interval_steps": args.report_every,
                        "reward": interval_reward,
                        "completed_episodes": interval_episodes,
                        "learner_updates": interval_updates,
                        "action_counts": interval_action_counts,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                interval_reward = 0.0
                interval_episodes = 0
                interval_updates = 0
                interval_action_counts = [0] * action_count

        if args.checkpoint and agent is not None:
            agent.save_checkpoint(args.checkpoint)
            emit(
                {
                    "event": "checkpoint",
                    "path": args.checkpoint,
                    "environment_step": agent.environment_step,
                    "learner_step": agent.learner_step,
                }
            )
        elapsed = time.perf_counter() - started
        emit(
            {
                "event": "run_end",
                "environment": args.environment,
                "steps": args.steps,
                "seed": args.seed,
                "mode": run_mode,
                "total_reward": total_reward,
                "completed_episodes": episodes,
                "mean_completed_return": (
                    completed_return_sum / episodes if episodes else 0.0
                ),
                "partial_episode_return": episode_return,
                "partial_episode_length": episode_length,
                "action_counts": action_counts,
                "learner_updates": agent.learner_step if agent is not None else 0,
                "elapsed_seconds": elapsed,
                "environment_steps_per_second": args.steps / elapsed,
            }
        )
    finally:
        environment.close()
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
