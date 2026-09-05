"""Atari-100k-style training loop for Kindle.

Usage:
    python examples/atari.py /path/to/model.safetensors [ALE/Pong-v5]
"""

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

import kindle
from kindle._reward_probe import RewardProbe


MODEL_SIZES = ("1m", "12m", "25m", "50m", "100m", "200m")
DINO_RECONSTRUCTION_SCALE = 0.25
ATARI_ACTION_REPEAT = 4
ATARI_NOOP_MAX = 30
ATARI_SCREEN_SIZE = 64
ATARI_MAX_EPISODE_FRAMES = 108_000
ATARI_SCORE_WINDOW_START_FRAMES = 350_000
ATARI_SCORE_WINDOW_END_FRAMES = 400_000


@dataclass(frozen=True)
class AtariProtocol:
    full_action_space: bool
    noop_max: int
    max_episode_frames: int


ATARI_PROTOCOLS = {
    # Pinned Dreamer V3's current Atari-100k wrapper defaults.
    "current": AtariProtocol(False, ATARI_NOOP_MAX, ATARI_MAX_EPISODE_FRAMES),
    # Wrapper settings used to produce scores/atari100k-dreamerv3.json.gz at
    # upstream commit 2411f7d1.
    "published": AtariProtocol(True, 0, 100_000),
    # One-variable action-alias ablation: retain every published wrapper
    # setting while exposing only each game's minimal legal action set.
    "published-minimal": AtariProtocol(False, 0, 100_000),
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DreamerAtariPreprocessing(gym.Wrapper):
    """DreamerV3's Atari-100k image and interaction protocol.

    Gymnasium's AtariPreprocessing differs in three consequential details: it
    samples 1--30 rather than 0--30 reset no-ops, resizes with OpenCV, and
    checks termination before capturing a frame for the two-frame max pool.
    """

    def __init__(
        self,
        environment: gym.Env,
        *,
        noop_max: int = ATARI_NOOP_MAX,
        max_episode_frames: int = ATARI_MAX_EPISODE_FRAMES,
    ):
        super().__init__(environment)
        if noop_max < 0:
            raise ValueError("noop_max must be non-negative")
        if max_episode_frames <= 0:
            raise ValueError("max_episode_frames must be positive")
        action_meanings = environment.unwrapped.get_action_meanings()
        if not action_meanings or action_meanings[0] != "NOOP":
            raise ValueError("Atari action 0 must be NOOP")
        self.action_meanings = tuple(str(action) for action in action_meanings)
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
        # Exact emulator step() calls, separate from the conventional action×4
        # score coordinates. Early terminals need not execute all four repeats.
        self.executed_action_frames = 0
        self.reset_noop_frames = 0
        self.emulator_resets = 0
        self._noop_max = noop_max
        self._max_episode_frames = max_episode_frames

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.emulator_resets += 1
        info = dict(info)
        noops = int(self.env.unwrapped.np_random.integers(self._noop_max + 1))
        for _ in range(noops):
            observation, _, terminated, truncated, step_info = self.env.step(0)
            self.reset_noop_frames += 1
            info.update(step_info)
            if terminated or truncated:
                observation, reset_info = self.env.reset(options=options)
                self.emulator_resets += 1
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
            self.executed_action_frames += 1

            # D3 captures before checking game-over, so a terminal frame in
            # either of the last two repeats remains visible to the agent.
            if repeat >= ATARI_ACTION_REPEAT - 2:
                self._frames.append(
                    np.asarray(observation, dtype=np.uint8).copy()
                )
                self._frames = self._frames[-2:]

            if self._episode_frames >= self._max_episode_frames:
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
    parser.add_argument(
        "--atari-protocol",
        choices=tuple(ATARI_PROTOCOLS),
        default="current",
        help=(
            "Atari wrapper profile: current follows pinned D3 source; "
            "published matches its released score artifact; published-minimal "
            "changes only the published action vocabulary"
        ),
    )
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
    parser.add_argument(
        "--world-microbatch-size",
        type=int,
        help=(
            "world-model rows per gradient-accumulation pass; defaults to the "
            "full effective batch"
        ),
    )
    parser.add_argument(
        "--skip-full-optimize",
        action="store_true",
        help=(
            "skip Meganeura's post-autodiff rewrite pass; preserves graph "
            "semantics but may reduce training throughput"
        ),
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--agc", type=float, help="adaptive gradient clipping ratio; zero disables it")
    parser.add_argument("--visitation-bonus", action="store_true",
                        help="bounded novelty in frozen visual features")
    parser.add_argument("--intrinsic-reward-scale", type=float, default=0.0)
    parser.add_argument("--extrinsic-reward-scale", type=float, default=1.0)
    parser.add_argument("--future-prediction-loss-scale", type=float, default=0.0,
                        help="predict the next frozen features before observing them")
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
        default=DINO_RECONSTRUCTION_SCALE,
        help="frozen-DINO scale calibrated to D3's summed-pixel loss magnitude",
    )
    parser.add_argument(
        "--replay-value-gradient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let replay-value loss shape the RSSM (D3 default; use --no-replay-value-gradient for the isolated ablation)",
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
    parser.add_argument(
        "--reward-probe",
        action="store_true",
        help="during --evaluate, report prior/posterior reward calibration by target sign",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save --checkpoint every N runner steps in addition to the final save",
    )
    parser.add_argument("--output", help="write JSONL metrics to this file")
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="append to --output when continuing with --restore",
    )
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
    if args.evaluate and args.checkpoint:
        parser.error("--evaluate does not write training checkpoints")
    if args.greedy and not args.evaluate:
        parser.error("--greedy requires --evaluate")
    if args.reward_probe and not args.evaluate:
        parser.error("--reward-probe requires --evaluate")
    if args.append_output and not args.output:
        parser.error("--append-output requires --output")
    if args.append_output and not args.restore:
        parser.error("--append-output requires --restore")
    if args.random_policy and (args.restore or args.evaluate or args.checkpoint):
        parser.error("--random-policy cannot use Dreamer checkpoints or evaluation")
    training_flags = {
        "--model-size", "--observation-decoder-depth", "--train-ratio",
        "--replay-capacity", "--batch-size", "--batch-length",
        "--world-backprop-length", "--world-microbatch-size", "--skip-full-optimize",
        "--learning-rate", "--learning-rate-warmup", "--actor-learning-starts",
        "--free-nats", "--dynamics-free-nats", "--dynamics-loss-scale",
        "--reconstruction-loss-scale", "--future-prediction-loss-scale", "--agc",
        "--replay-value-gradient", "--no-replay-value-gradient", "--visitation-bonus",
        "--intrinsic-reward-scale", "--extrinsic-reward-scale",
    }
    explicit_training = training_flags.intersection(arg.split("=", 1)[0] for arg in sys.argv[1:])
    if (args.restore or args.random_policy) and explicit_training:
        parser.error("training overrides require a fresh Dreamer run: " + ", ".join(sorted(explicit_training)))

    protocol = ATARI_PROTOCOLS[args.atari_protocol]
    # Both D3 Atari-100k profiles use repeat 4, max-pool 2, non-sticky
    # actions, and Pillow bilinear RGB resize. The released score artifact
    # predates changes to the action set, reset no-ops, and episode cap.
    environment = gym.make(
        args.environment,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=protocol.full_action_space,
    )
    environment = DreamerAtariPreprocessing(
        environment,
        noop_max=protocol.noop_max,
        max_episode_frames=protocol.max_episode_frames,
    )
    frame, _ = environment.reset(seed=args.seed)
    if getattr(frame, "ndim", 0) != 3:
        raise ValueError("the environment must return H×W×3 RGB observations")
    action_count = int(environment.action_space.n)
    action_meanings = list(environment.action_meanings)
    if len(action_meanings) != action_count:
        raise ValueError(
            f"environment exposes {action_count} actions but "
            f"{len(action_meanings)} action meanings"
        )
    dino_checkpoint_sha256 = (
        None if args.random_policy else sha256_file(args.dino_checkpoint)
    )
    construction_started = time.perf_counter()
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
            world_microbatch_size=args.world_microbatch_size,
            train_ratio=args.train_ratio,
            learning_rate=args.learning_rate,
            actor_learning_starts=args.actor_learning_starts,
            learning_rate_warmup=args.learning_rate_warmup,
            free_nats=args.free_nats,
            dynamics_free_nats=args.dynamics_free_nats,
            dynamics_loss_scale=args.dynamics_loss_scale,
            reconstruction_loss_scale=args.reconstruction_loss_scale,
            future_prediction_loss_scale=args.future_prediction_loss_scale,
            agc=args.agc,
            visitation_bonus=args.visitation_bonus,
            intrinsic_reward_scale=args.intrinsic_reward_scale,
            extrinsic_reward_scale=args.extrinsic_reward_scale,
            replay_value_gradient=args.replay_value_gradient,
            skip_full_optimize=args.skip_full_optimize,
        )
    agent_construction_seconds = (
        time.perf_counter() - construction_started if agent is not None else None
    )
    agent_config = agent.config if agent is not None else None
    if agent is not None:
        agent.begin_episode(frame)
    random_actions = random.Random(args.seed ^ 0xA7A2_1000)
    if args.random_policy:
        run_mode = "random"
    elif args.evaluate:
        evaluation_policy = "greedy" if args.greedy else "sample"
        if args.random_action_steps >= args.steps:
            run_mode = "evaluate_forced_random"
        elif args.random_action_steps:
            run_mode = f"evaluate_forced_random_then_{evaluation_policy}"
        else:
            run_mode = f"evaluate_{evaluation_policy}"
    else:
        run_mode = "train"
    starting_environment_step = agent.environment_step if agent is not None else 0
    starting_learner_step = agent.learner_step if agent is not None else 0
    parameter_counts = (
        getattr(agent, "trainable_parameter_counts", None)
        if agent is not None
        else None
    )
    if parameter_counts is None:
        trainable_parameters = None
    else:
        world_parameters, behavior_parameters = parameter_counts
        trainable_parameters = {
            "world": world_parameters,
            "behavior": behavior_parameters,
            "total": world_parameters + behavior_parameters,
        }

    def absolute_environment_step(run_step: int) -> int:
        return starting_environment_step + run_step

    def absolute_environment_frames(run_step: int) -> int:
        return absolute_environment_step(run_step) * ATARI_ACTION_REPEAT

    output = (
        open(
            args.output,
            "a" if args.append_output else "w",
            encoding="utf-8",
        )
        if args.output
        else sys.stdout
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
    total_intrinsic_reward = 0.0
    interval_intrinsic_reward = 0.0
    positive_reward_events = 0
    negative_reward_events = 0
    interval_positive_reward_events = 0
    interval_negative_reward_events = 0
    episode_return = 0.0
    episode_length = 0
    completed_return_sum = 0.0
    episodes = 0
    score_window_return_sum = 0.0
    score_window_episodes = 0
    interval_reward = 0.0
    interval_episodes = 0
    interval_updates = 0
    action_counts = [0] * action_count
    interval_action_counts = [0] * action_count
    reward_probe = RewardProbe() if args.reward_probe else None
    emit(
        {
            "event": "run_start",
            "environment": args.environment,
            "ale_py_version": ale_py.__version__,
            "steps": args.steps,
            "atari_protocol": args.atari_protocol,
            "action_repeat": ATARI_ACTION_REPEAT,
            "full_action_space": protocol.full_action_space,
            "noop_max": protocol.noop_max,
            "max_episode_frames": protocol.max_episode_frames,
            "environment_frame_budget": args.steps * ATARI_ACTION_REPEAT,
            "score_window_frames": [
                ATARI_SCORE_WINDOW_START_FRAMES,
                ATARI_SCORE_WINDOW_END_FRAMES,
            ],
            "seed": args.seed,
            "actions": action_count,
            "action_meanings": action_meanings,
            "mode": run_mode,
            "reward_probe": args.reward_probe,
            "random_action_steps": args.random_action_steps,
            "output_appended": args.append_output,
            "dino_model_id": (
                kindle.DINO_MODEL_ID if agent is not None else None
            ),
            "dino_checkpoint_revision": (
                kindle.DINO_CHECKPOINT_REVISION if agent is not None else None
            ),
            "dino_checkpoint_sha256": dino_checkpoint_sha256,
            "starting_environment_step": starting_environment_step,
            "starting_learner_step": starting_learner_step,
            "agent_construction_seconds": agent_construction_seconds,
            "gpu_device": agent.gpu_device if agent is not None else None,
            "trainable_parameters": trainable_parameters,
            "config": agent_config,
            "model_provenance": agent.provenance if agent is not None else None,
            "native_extension_sha256": (
                sha256_file(kindle._native.__file__) if agent is not None else None
            ),
            "runner_sha256": sha256_file(__file__),
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
            one_step_prior_reward = (
                agent.prior_reward_prediction(action)
                if reward_probe is not None and agent is not None
                else None
            )
            frame, reward, terminated, truncated, _ = environment.step(action)
            reward = float(reward)
            terminated = bool(terminated)
            truncated = bool(truncated)
            intrinsic_reward = 0.0
            if agent is not None:
                _, intrinsic_reward = agent.observe(
                    frame,
                    extrinsic_reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            if reward_probe is not None:
                assert agent is not None
                assert one_step_prior_reward is not None
                reward_probe.record(
                    (agent_config["extrinsic_reward_scale"] * reward
                     + agent_config["intrinsic_reward_scale"] * intrinsic_reward),
                    one_step_prior_reward,
                    agent.posterior_reward_prediction(),
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
            total_intrinsic_reward += intrinsic_reward
            interval_intrinsic_reward += intrinsic_reward
            positive_reward_events += int(reward > 0.0)
            negative_reward_events += int(reward < 0.0)
            interval_positive_reward_events += int(reward > 0.0)
            interval_negative_reward_events += int(reward < 0.0)
            episode_return += reward
            episode_length += 1
            action_counts[action] += 1
            interval_action_counts[action] += 1
            if terminated or truncated:
                episode_environment_frames = absolute_environment_frames(run_step)
                episodes += 1
                interval_episodes += 1
                completed_return_sum += episode_return
                if (
                    ATARI_SCORE_WINDOW_START_FRAMES
                    <= episode_environment_frames
                    <= ATARI_SCORE_WINDOW_END_FRAMES
                ):
                    score_window_return_sum += episode_return
                    score_window_episodes += 1
                emit(
                    {
                        "event": "episode",
                        "episode": episodes,
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
                        "executed_action_frames": environment.executed_action_frames,
                        "reset_noop_frames": environment.reset_noop_frames,
                        "emulator_resets": environment.emulator_resets,
                        "reward": interval_reward,
                        "intrinsic_reward": interval_intrinsic_reward,
                        "positive_reward_events": interval_positive_reward_events,
                        "negative_reward_events": interval_negative_reward_events,
                        "completed_episodes": interval_episodes,
                        "learner_updates": interval_updates,
                        "action_counts": interval_action_counts,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                interval_reward = 0.0
                interval_intrinsic_reward = 0.0
                interval_positive_reward_events = 0
                interval_negative_reward_events = 0
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
        if reward_probe is not None:
            emit(
                {
                    "event": "reward_probe",
                    "target": "scaled_extrinsic_plus_intrinsic",
                    "run_step": args.steps,
                    "environment_step": absolute_environment_step(args.steps),
                    "environment_frames": absolute_environment_frames(args.steps),
                    **reward_probe.summary(),
                }
            )
        starting_environment_frames = starting_environment_step * ATARI_ACTION_REPEAT
        ending_environment_frames = absolute_environment_frames(args.steps)
        emit(
            {
                "event": "run_end",
                "environment": args.environment,
                "steps": args.steps,
                "environment_step": absolute_environment_step(args.steps),
                "environment_frames": ending_environment_frames,
                "environment_frame_delta": args.steps * ATARI_ACTION_REPEAT,
                "executed_action_frames": environment.executed_action_frames,
                "reset_noop_frames": environment.reset_noop_frames,
                "emulator_resets": environment.emulator_resets,
                "seed": args.seed,
                "mode": run_mode,
                "total_reward": total_reward,
                "total_intrinsic_reward": total_intrinsic_reward,
                "positive_reward_events": positive_reward_events,
                "negative_reward_events": negative_reward_events,
                "completed_episodes": episodes,
                "mean_completed_return": (
                    completed_return_sum / episodes if episodes else 0.0
                ),
                "score_window": {
                    "start_environment_frames": ATARI_SCORE_WINDOW_START_FRAMES,
                    "end_environment_frames": ATARI_SCORE_WINDOW_END_FRAMES,
                    "complete": (
                        starting_environment_frames
                        <= ATARI_SCORE_WINDOW_START_FRAMES
                        and ending_environment_frames
                        >= ATARI_SCORE_WINDOW_END_FRAMES
                    ),
                    "completed_episodes": score_window_episodes,
                    "mean_completed_return": (
                        score_window_return_sum / score_window_episodes
                        if score_window_episodes
                        else None
                    ),
                },
                "partial_episode_return": episode_return,
                "partial_episode_length": episode_length,
                "action_counts": action_counts,
                "environment_step_delta": environment_step_delta,
                "learner_step_delta": learner_step_delta,
                "learner_updates": learner_step_delta,
                "elapsed_seconds": elapsed,
                "environment_steps_per_second": args.steps / elapsed,
                "environment_frames_per_second": (
                    args.steps * ATARI_ACTION_REPEAT / elapsed
                ),
                "learner_updates_per_second": learner_step_delta / elapsed,
            }
        )
    finally:
        environment.close()
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
