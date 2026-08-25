"""Compare a frozen Atari policy with its one-step model values.

The probe restores a checkpoint, follows a deterministic forced-action
trajectory, and evaluates every candidate action from sampled posterior states.
Each candidate reuses identical categorical random draws. The reported
one-step target is reward(next) + continuation(next) * value(next), which helps
separate an actor-alignment failure from an uninformative or misleading critic.
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


def checked_mean(values):
    return float(np.mean(values)) if values else None


def checked_correlation(left, right):
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def pairwise_ranking_accuracy(probabilities, targets):
    correct = 0.0
    compared = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            target_order = float(targets[left] - targets[right])
            if abs(target_order) <= 1e-9:
                continue
            probability_order = float(probabilities[left] - probabilities[right])
            if probability_order == 0.0:
                correct += 0.5
            elif probability_order * target_order > 0.0:
                correct += 1.0
            compared += 1
    return correct / compared if compared else None


def main() -> None:
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument(
        "--atari-protocol", choices=tuple(ATARI_PROTOCOLS), default="published"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
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
    action_meanings = list(environment.unwrapped.get_action_meanings())
    agent = kindle.Agent.restore(args.checkpoint, args.dino_checkpoint)
    if int(agent.config["action_count"]) != action_count:
        raise ValueError("checkpoint and environment action counts differ")
    agent.begin_episode(frame)
    starting_environment_step = agent.environment_step

    action_rng = random.Random(args.seed ^ ACTION_SEED_XOR)
    actions = [action_rng.randrange(action_count) for _ in range(args.steps)]
    policy_entropy = []
    posterior_values = []
    reward_spans = []
    next_value_spans = []
    one_step_spans = []
    uniform_one_step_values = []
    policy_one_step_values = []
    greedy_one_step_values = []
    best_one_step_values = []
    best_action_probabilities = []
    action_correlations = []
    ranking_accuracies = []
    forced_reward_predictions = []
    forced_one_step_values = []
    realized_rewards = []
    greedy_action_counts = [0] * action_count
    best_action_counts = [0] * action_count
    policy_probability_sums = np.zeros(action_count, dtype=np.float64)
    reward_prediction_sums = np.zeros(action_count, dtype=np.float64)
    continuation_prediction_sums = np.zeros(action_count, dtype=np.float64)
    next_value_sums = np.zeros(action_count, dtype=np.float64)
    one_step_value_sums = np.zeros(action_count, dtype=np.float64)
    top_action_agreements = 0
    state_samples = 0
    episodes = 0
    pending_prediction = None

    for step, action in enumerate(actions):
        if step % args.stride == 0:
            probabilities = np.asarray(
                agent.posterior_action_probabilities(), dtype=np.float64
            )
            if probabilities.shape != (action_count,):
                raise RuntimeError("policy returned the wrong action count")
            if not np.isclose(float(np.sum(probabilities)), 1.0):
                raise RuntimeError("policy probabilities do not sum to one")
            rewards = np.empty(action_count, dtype=np.float64)
            continuations = np.empty(action_count, dtype=np.float64)
            values = np.empty(action_count, dtype=np.float64)
            for candidate in range(action_count):
                predicted = agent.prior_behavior_rollout([candidate])
                if any(len(channel) != 1 for channel in predicted):
                    raise RuntimeError(
                        "one-step behavior rollout returned the wrong shape"
                    )
                rewards[candidate] = predicted[0][0]
                continuations[candidate] = predicted[1][0]
                values[candidate] = predicted[2][0]
            one_step = rewards + continuations * values
            greedy = int(np.argmax(probabilities))
            best = int(np.argmax(one_step))
            entropy = -float(
                np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)))
            )
            correlation = checked_correlation(probabilities, one_step)
            ranking = pairwise_ranking_accuracy(probabilities, one_step)

            policy_entropy.append(entropy)
            posterior_values.append(float(agent.posterior_value_prediction()))
            reward_spans.append(float(np.max(rewards) - np.min(rewards)))
            next_value_spans.append(float(np.max(values) - np.min(values)))
            one_step_spans.append(float(np.max(one_step) - np.min(one_step)))
            uniform_one_step_values.append(float(np.mean(one_step)))
            policy_one_step_values.append(float(np.dot(probabilities, one_step)))
            greedy_one_step_values.append(float(one_step[greedy]))
            best_one_step_values.append(float(one_step[best]))
            best_action_probabilities.append(float(probabilities[best]))
            if correlation is not None:
                action_correlations.append(correlation)
            if ranking is not None:
                ranking_accuracies.append(ranking)
            greedy_action_counts[greedy] += 1
            best_action_counts[best] += 1
            policy_probability_sums += probabilities
            reward_prediction_sums += rewards
            continuation_prediction_sums += continuations
            next_value_sums += values
            one_step_value_sums += one_step
            top_action_agreements += int(greedy == best)
            forced_reward_predictions.append(float(rewards[action]))
            forced_one_step_values.append(float(one_step[action]))
            pending_prediction = len(forced_reward_predictions) - 1
            state_samples += 1

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
        if pending_prediction is not None:
            if pending_prediction != len(realized_rewards):
                raise RuntimeError("sampled reward alignment was lost")
            realized_rewards.append(reward)
            pending_prediction = None
        if terminated or truncated:
            episodes += 1
            frame, _ = environment.reset()
            agent.begin_episode(frame)
    environment.close()

    policy_array = np.asarray(policy_one_step_values)
    uniform_array = np.asarray(uniform_one_step_values)
    greedy_array = np.asarray(greedy_one_step_values)
    best_array = np.asarray(best_one_step_values)
    result = {
        "environment": args.environment,
        "atari_protocol": args.atari_protocol,
        "checkpoint": args.checkpoint,
        "starting_environment_step": starting_environment_step,
        "steps": args.steps,
        "seed": args.seed,
        "stride": args.stride,
        "state_samples": state_samples,
        "completed_episodes": episodes,
        "action_meanings": action_meanings,
        "gpu_device": agent.gpu_device,
        "one_step_target": "reward_next + continuation_next * value_next",
        "paired_categorical_draws": True,
        "mean_policy_entropy": checked_mean(policy_entropy),
        "mean_posterior_value": checked_mean(posterior_values),
        "mean_reward_action_span": checked_mean(reward_spans),
        "mean_next_value_action_span": checked_mean(next_value_spans),
        "mean_one_step_action_span": checked_mean(one_step_spans),
        "mean_uniform_one_step_value": checked_mean(uniform_one_step_values),
        "mean_policy_one_step_value": checked_mean(policy_one_step_values),
        "mean_greedy_one_step_value": checked_mean(greedy_one_step_values),
        "mean_best_one_step_value": checked_mean(best_one_step_values),
        "mean_policy_gain_over_uniform": float(np.mean(policy_array - uniform_array)),
        "mean_greedy_gap_to_best": float(np.mean(best_array - greedy_array)),
        "mean_probability_of_best_action": checked_mean(best_action_probabilities),
        "greedy_best_action_agreement": top_action_agreements / state_samples,
        "mean_policy_q_correlation": checked_mean(action_correlations),
        "mean_pairwise_ranking_accuracy": checked_mean(ranking_accuracies),
        "greedy_action_counts": greedy_action_counts,
        "best_one_step_action_counts": best_action_counts,
        "mean_policy_probability_by_action": (
            policy_probability_sums / state_samples
        ).tolist(),
        "mean_reward_prediction_by_action": (
            reward_prediction_sums / state_samples
        ).tolist(),
        "mean_continuation_prediction_by_action": (
            continuation_prediction_sums / state_samples
        ).tolist(),
        "mean_next_value_by_action": (next_value_sums / state_samples).tolist(),
        "mean_one_step_value_by_action": (
            one_step_value_sums / state_samples
        ).tolist(),
        "forced_action_reward_prediction_mae": float(
            np.mean(
                np.abs(
                    np.asarray(forced_reward_predictions)
                    - np.asarray(realized_rewards)
                )
            )
        ),
        "mean_forced_action_one_step_value": checked_mean(forced_one_step_values),
        "sampled_realized_reward_mean": checked_mean(realized_rewards),
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
