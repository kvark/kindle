"""Measure critic calibration on a frozen Atari policy trajectory."""

import argparse
import json

import ale_py
import gymnasium as gym
import numpy as np

import kindle
from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing


def checked_mean(values):
    return float(np.mean(values)) if len(values) else None


def checked_correlation(left, right):
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--atari-protocol", choices=tuple(ATARI_PROTOCOLS), default="published"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

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
    action_meanings = list(environment.unwrapped.get_action_meanings())
    agent = kindle.Agent.restore(args.checkpoint, args.dino_checkpoint)
    if int(agent.config["action_count"]) != action_count:
        raise ValueError("checkpoint and environment action counts differ")
    agent.begin_episode(frame)
    starting_environment_step = agent.environment_step
    gamma = 1.0 - 1.0 / float(agent.config["horizon"])

    values = []
    rewards = []
    policy_entropies = []
    action_counts = [0] * action_count
    episode_boundaries = []
    episode_start = 0
    episode_reward = 0.0
    episodes = []

    for step in range(args.steps):
        probabilities = np.asarray(
            agent.posterior_action_probabilities(), dtype=np.float64
        )
        if probabilities.shape != (action_count,):
            raise RuntimeError("policy returned the wrong action count")
        if not np.isclose(float(np.sum(probabilities)), 1.0):
            raise RuntimeError("policy probabilities do not sum to one")
        values.append(float(agent.posterior_value_prediction()))
        policy_entropies.append(
            -float(
                np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)))
            )
        )
        action = int(agent.act(greedy=args.greedy))
        action_counts[action] += 1
        frame, reward, terminated, truncated, _ = environment.step(action)
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        rewards.append(reward)
        episode_reward += reward
        agent.observe(
            frame,
            extrinsic_reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        if terminated or truncated:
            end = step + 1
            episode_boundaries.append((episode_start, end, terminated, truncated))
            episodes.append(
                {
                    "return": episode_reward,
                    "length": end - episode_start,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            episode_start = end
            episode_reward = 0.0
            frame, _ = environment.reset()
            agent.begin_episode(frame)
    environment.close()

    monte_carlo_returns = [None] * args.steps
    for start, end, terminated, truncated in episode_boundaries:
        if not terminated or truncated:
            continue
        running = 0.0
        for index in range(end - 1, start - 1, -1):
            running = rewards[index] + gamma * running
            monte_carlo_returns[index] = running
    evaluated_indices = [
        index for index, target in enumerate(monte_carlo_returns) if target is not None
    ]
    if not evaluated_indices:
        raise RuntimeError("no terminated episode completed within the probe")
    predictions = np.asarray([values[index] for index in evaluated_indices])
    targets = np.asarray([monte_carlo_returns[index] for index in evaluated_indices])
    errors = predictions - targets
    prediction_variance = float(np.var(predictions))
    calibration_slope = (
        float(np.cov(predictions, targets, ddof=0)[0, 1] / prediction_variance)
        if prediction_variance > 0.0
        else None
    )
    calibration_intercept = (
        float(np.mean(targets) - calibration_slope * np.mean(predictions))
        if calibration_slope is not None
        else None
    )
    result = {
        "environment": args.environment,
        "atari_protocol": args.atari_protocol,
        "checkpoint": args.checkpoint,
        "starting_environment_step": starting_environment_step,
        "steps": args.steps,
        "seed": args.seed,
        "mode": "greedy" if args.greedy else "sample",
        "gamma": gamma,
        "action_meanings": action_meanings,
        "action_counts": action_counts,
        "gpu_device": agent.gpu_device,
        "completed_episodes": len(episodes),
        "episodes": episodes,
        "partial_episode_return": episode_reward,
        "partial_episode_length": args.steps - episode_start,
        "evaluated_states": len(evaluated_indices),
        "excluded_states": args.steps - len(evaluated_indices),
        "mean_policy_entropy": checked_mean(policy_entropies),
        "mean_predicted_value": float(np.mean(predictions)),
        "predicted_value_std": float(np.std(predictions)),
        "mean_realized_discounted_return": float(np.mean(targets)),
        "realized_discounted_return_std": float(np.std(targets)),
        "value_bias": float(np.mean(errors)),
        "value_error_std": float(np.std(errors)),
        "value_mae": float(np.mean(np.abs(errors))),
        "value_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "value_return_correlation": checked_correlation(predictions, targets),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
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
