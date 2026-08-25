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
import numpy as np
from PIL import Image

import kindle


MODEL_SIZES = ("1m", "12m", "25m", "50m", "100m", "200m")
MEAN_DINO_RECONSTRUCTION_SCALE = 1.0 / (7 * 7 * 64)
ATARI_ACTION_REPEAT = 4
ATARI_NOOP_MAX = 30
ATARI_SCREEN_SIZE = 64
ATARI_MAX_EPISODE_FRAMES = 108_000


class DreamerAtariPreprocessing(gym.Wrapper):
    """DreamerV3's Atari-100k image and interaction protocol.

    Gymnasium's AtariPreprocessing differs in three consequential details: it
    samples 1--30 rather than 0--30 reset no-ops, resizes with OpenCV, and
    checks termination before capturing a frame for the two-frame max pool.
    """

    def __init__(self, environment: gym.Env):
        super().__init__(environment)
        action_meanings = environment.unwrapped.get_action_meanings()
        if not action_meanings or action_meanings[0] != "NOOP":
            raise ValueError("Atari action 0 must be NOOP")
        raw_space = environment.observation_space
        if raw_space.shape is None or len(raw_space.shape) != 3:
            raise ValueError("the Atari environment must return RGB images")
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(ATARI_SCREEN_SIZE, ATARI_SCREEN_SIZE, 3),
            dtype=np.uint8,
        )
        self._frames: list[np.ndarray] = []
        self._episode_frames = 0

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        noops = int(
            self.env.unwrapped.np_random.integers(ATARI_NOOP_MAX + 1)
        )
        for _ in range(noops):
            observation, _, terminated, truncated, step_info = self.env.step(0)
            info.update(step_info)
            if terminated or truncated:
                observation, reset_info = self.env.reset(options=options)
                info.update(reset_info)
        frame = np.asarray(observation, dtype=np.uint8)
        self._frames = [frame.copy(), frame.copy()]
        self._episode_frames = 0
        info["noop_count"] = noops
        return self._observation(), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for repeat in range(ATARI_ACTION_REPEAT):
            observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            total_reward += float(reward)
            self._episode_frames += 1

            # D3 captures before checking game-over, so a terminal frame in
            # either of the last two repeats remains visible to the agent.
            if repeat >= ATARI_ACTION_REPEAT - 2:
                self._frames.append(
                    np.asarray(observation, dtype=np.uint8).copy()
                )
                self._frames = self._frames[-2:]

            if self._episode_frames >= ATARI_MAX_EPISODE_FRAMES:
                truncated = True
            if terminated or truncated:
                break

        return (
            self._observation(),
            total_reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    def _observation(self) -> np.ndarray:
        image = np.maximum(self._frames[0], self._frames[1])
        image = Image.fromarray(image).resize(
            (ATARI_SCREEN_SIZE, ATARI_SCREEN_SIZE), Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.uint8)


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
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-length", type=int)
    parser.add_argument(
        "--world-backprop-length",
        type=int,
        help="truncated world-model BPTT length (D3-equivalent: 64)",
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
        help="run the restored sampled policy without learning, matching D3",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="use actor argmax during --evaluate instead of D3's sampling",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save --checkpoint every N runner steps in addition to the final save",
    )
    parser.add_argument("--output", help="write JSONL metrics to this file")
    parser.add_argument("--report-every", type=int, default=1_000)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    if args.random_action_steps < 0:
        parser.error("--random-action-steps must be non-negative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.checkpoint_every and not args.checkpoint:
        parser.error("--checkpoint-every requires --checkpoint")
    if args.evaluate and not args.restore:
        parser.error("--evaluate requires --restore")
    if args.evaluate and args.random_action_steps:
        parser.error("--evaluate cannot be combined with --random-action-steps")
    if args.evaluate and args.checkpoint:
        parser.error("--evaluate does not write training checkpoints")
    if args.greedy and not args.evaluate:
        parser.error("--greedy requires --evaluate")
    if args.random_policy and (args.restore or args.evaluate or args.checkpoint):
        parser.error("--random-policy cannot use Dreamer checkpoints or evaluation")

    # Match D3's Atari-100k protocol: minimal actions, repeat 4, max-pool 2,
    # non-sticky actions, 0--30 reset no-ops, and Pillow bilinear RGB resize.
    environment = gym.make(
        args.environment,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    environment = DreamerAtariPreprocessing(environment)
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
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            batch_length=args.batch_length,
            world_backprop_length=args.world_backprop_length,
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
    if args.random_policy:
        run_mode = "random"
    elif args.evaluate:
        run_mode = "evaluate_greedy" if args.greedy else "evaluate_sample"
    else:
        run_mode = "train"
    starting_environment_step = agent.environment_step if agent is not None else 0
    starting_learner_step = agent.learner_step if agent is not None else 0

    def absolute_environment_step(run_step: int) -> int:
        return starting_environment_step + run_step

    def absolute_environment_frames(run_step: int) -> int:
        return absolute_environment_step(run_step) * ATARI_ACTION_REPEAT

    output = (
        open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    )

    def emit(event: dict[str, object]) -> None:
        print(json.dumps(event, separators=(",", ":")), file=output, flush=True)

    def save_checkpoint(run_step: int) -> None:
        assert args.checkpoint is not None
        assert agent is not None
        agent.save_checkpoint(args.checkpoint)
        emit(
            {
                "event": "checkpoint",
                "path": args.checkpoint,
                "run_step": run_step,
                "environment_step": agent.environment_step,
                "environment_frames": (
                    agent.environment_step * ATARI_ACTION_REPEAT
                ),
                "learner_step": agent.learner_step,
            }
        )

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
            "action_repeat": ATARI_ACTION_REPEAT,
            "environment_frame_budget": args.steps * ATARI_ACTION_REPEAT,
            "seed": args.seed,
            "actions": action_count,
            "mode": run_mode,
            "starting_environment_step": starting_environment_step,
            "starting_learner_step": starting_learner_step,
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
                action = agent.act(greedy=args.evaluate and args.greedy)
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
                        "environment_step": absolute_environment_step(run_step),
                        "environment_frames": absolute_environment_frames(run_step),
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
                        "environment_step": absolute_environment_step(run_step),
                        "environment_frames": absolute_environment_frames(run_step),
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
                        "environment_step": absolute_environment_step(run_step),
                        "environment_frames": absolute_environment_frames(run_step),
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

            if (
                args.checkpoint
                and args.checkpoint_every
                and run_step % args.checkpoint_every == 0
            ):
                save_checkpoint(run_step)

        if args.checkpoint and (
            not args.checkpoint_every
            or args.steps % args.checkpoint_every != 0
        ):
            save_checkpoint(args.steps)
        elapsed = time.perf_counter() - started
        environment_step_delta = (
            agent.environment_step - starting_environment_step
            if agent is not None
            else args.steps
        )
        learner_step_delta = (
            agent.learner_step - starting_learner_step
            if agent is not None
            else 0
        )
        emit(
            {
                "event": "run_end",
                "environment": args.environment,
                "steps": args.steps,
                "environment_step": absolute_environment_step(args.steps),
                "environment_frames": absolute_environment_frames(args.steps),
                "environment_frame_delta": args.steps * ATARI_ACTION_REPEAT,
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
                "environment_step_delta": environment_step_delta,
                "learner_step_delta": learner_step_delta,
                "learner_updates": learner_step_delta,
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
