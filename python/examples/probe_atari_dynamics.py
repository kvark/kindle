"""Measure open-loop Atari dynamics in frozen visual feature space.

The probe restores a checkpoint, follows a deterministic forced-action
trajectory, and compares decoded prior predictions with the observations that
actually follow. A persistence predictor using the start observation is
reported beside the model at every horizon.
"""

import argparse
import json
import random

import ale_py
import gymnasium as gym
import numpy as np

import kindle
from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing


ACTION_SEED_XOR = 0xA7A2_1000
CONTROL_ACTION_SEED_XOR = 0xD1A6_4057


def checked_mean(total, count):
    return total / count if count else None


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder_checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument(
        "--atari-protocol", choices=tuple(ATARI_PROTOCOLS), default="published"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.horizon <= 0 or args.horizon > args.steps:
        parser.error("--horizon must be in [1, steps]")
    if args.stride <= 0:
        parser.error("--stride must be positive")

    protocol = ATARI_PROTOCOLS[args.atari_protocol]
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
    action_count = int(environment.action_space.n)
    agent = kindle.Agent.restore(args.checkpoint, args.encoder_checkpoint)
    if agent.config["intrinsic_reward_scale"] != 0 or agent.config["extrinsic_reward_scale"] != 1:
        raise ValueError("this probe's reward labels require unscaled extrinsic-only training")
    if int(agent.config["action_count"]) != action_count:
        raise ValueError("checkpoint and environment action counts differ")
    agent.begin_episode(frame)
    starting_environment_step = agent.environment_step

    action_rng = random.Random(args.seed ^ ACTION_SEED_XOR)
    actions = [action_rng.randrange(action_count) for _ in range(args.steps)]
    control_action_rng = random.Random(args.seed ^ CONTROL_ACTION_SEED_XOR)
    control_actions = [
        control_action_rng.randrange(action_count) for _ in range(args.steps)
    ]
    horizon = args.horizon
    model_mse_sum = [0.0] * horizon
    control_mse_sum = [0.0] * horizon
    persistence_mse_sum = [0.0] * horizon
    reward_mae_sum = [0.0] * horizon
    control_reward_mae_sum = [0.0] * horizon
    sample_count = [0] * horizon
    posterior_mse_sum = 0.0
    posterior_count = 0
    rollout_starts = 0
    episodes = 0
    pending = {}
    is_first = True
    forecasts = agent.config["loss_scales"]["future_prediction"] > 0

    for step, action in enumerate(actions):
        if step % args.stride == 0 and step + horizon <= args.steps:
            start_observation = np.asarray(agent.visual_observation, dtype=np.float64)
            posterior = np.asarray(
                agent.observation_prediction(), dtype=np.float64
            )
            if not is_first or not forecasts:
                posterior_mse_sum += float(
                    np.mean(np.square(posterior - start_observation))
                )
                posterior_count += 1
            predicted_rewards, predicted_observations = (
                agent.prior_diagnostic_rollout(actions[step : step + horizon])
            )
            control_rewards, control_observations = agent.prior_diagnostic_rollout(
                control_actions[step : step + horizon]
            )
            if (
                len(predicted_rewards) != horizon
                or len(predicted_observations) != horizon
            ):
                raise RuntimeError("prior rollout returned the wrong horizon")
            if (
                len(control_rewards) != horizon
                or len(control_observations) != horizon
            ):
                raise RuntimeError("control rollout returned the wrong horizon")
            predictions = zip(
                predicted_rewards,
                predicted_observations,
                control_rewards,
                control_observations,
            )
            for offset, prediction in enumerate(predictions):
                reward, observation, control_reward, control_observation = prediction
                target_step = step + offset
                pending.setdefault(target_step, []).append(
                    (
                        offset,
                        float(reward),
                        np.asarray(observation, dtype=np.float64),
                        float(control_reward),
                        np.asarray(control_observation, dtype=np.float64),
                        start_observation,
                    )
                )
            rollout_starts += 1

        action_mask = [False] * action_count
        action_mask[action] = True
        if agent.act(action_mask=action_mask) != action:
            raise RuntimeError("forced action was not honored")
        frame, reward, terminated, truncated, _ = environment.step(action)
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        agent.observe(
            frame,
            extrinsic_reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        target_observation = np.asarray(agent.visual_observation, dtype=np.float64)
        for (
            offset,
            predicted_reward,
            predicted_observation,
            control_reward,
            control_observation,
            start_observation,
        ) in pending.pop(step, []):
            model_mse_sum[offset] += float(
                np.mean(np.square(predicted_observation - target_observation))
            )
            control_mse_sum[offset] += float(
                np.mean(np.square(control_observation - target_observation))
            )
            persistence_mse_sum[offset] += float(
                np.mean(np.square(start_observation - target_observation))
            )
            reward_mae_sum[offset] += abs(predicted_reward - reward)
            control_reward_mae_sum[offset] += abs(control_reward - reward)
            sample_count[offset] += 1

        if terminated or truncated:
            episodes += 1
            pending.clear()
            frame, _ = environment.reset()
            agent.begin_episode(frame)
        is_first = terminated or truncated
    environment.close()

    model_mse = [
        checked_mean(total, count) for total, count in zip(model_mse_sum, sample_count)
    ]
    control_mse = [
        checked_mean(total, count)
        for total, count in zip(control_mse_sum, sample_count)
    ]
    persistence_mse = [
        checked_mean(total, count)
        for total, count in zip(persistence_mse_sum, sample_count)
    ]
    result = {
        "environment": args.environment,
        "atari_protocol": args.atari_protocol,
        "checkpoint": args.checkpoint,
        "starting_environment_step": starting_environment_step,
        "steps": args.steps,
        "seed": args.seed,
        "horizon": horizon,
        "stride": args.stride,
        "rollout_starts": rollout_starts,
        "completed_episodes": episodes,
        "gpu_device": agent.gpu_device,
        "observation_prediction_source": (
            "deterministic_forecast" if forecasts else "posterior_reconstruction"
        ),
        "observed_state_prediction_count": posterior_count,
        "observed_state_prediction_mse": checked_mean(
            posterior_mse_sum, posterior_count
        ),
        "sample_count_by_horizon": sample_count,
        "prior_observation_mse_by_horizon": model_mse,
        "unrelated_action_mse_by_horizon": control_mse,
        "prior_over_unrelated_action_by_horizon": [
            model / control
            if model is not None and control is not None and control > 0.0
            else None
            for model, control in zip(model_mse, control_mse)
        ],
        "persistence_mse_by_horizon": persistence_mse,
        "prior_over_persistence_by_horizon": [
            model / baseline
            if model is not None and baseline is not None and baseline > 0.0
            else None
            for model, baseline in zip(model_mse, persistence_mse)
        ],
        "prior_reward_mae_by_horizon": [
            checked_mean(total, count)
            for total, count in zip(reward_mae_sum, sample_count)
        ],
        "unrelated_action_reward_mae_by_horizon": [
            checked_mean(total, count)
            for total, count in zip(control_reward_mae_sum, sample_count)
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(encoded)
            output.write("\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
