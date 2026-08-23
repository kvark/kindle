"""Minimal Gymnasium Atari loop for Kindle.

Usage:
    python examples/atari.py /path/to/model.safetensors [ALE/Pong-v5]
"""

import argparse

import gymnasium as gym

import kindle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_checkpoint")
    parser.add_argument("environment", nargs="?", default="ALE/Pong-v5")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    environment = gym.make(args.environment)
    frame, _ = environment.reset()
    if getattr(frame, "ndim", 0) != 3:
        raise ValueError("the environment must return H×W×3 RGB observations")
    agent = kindle.Agent(
        args.dino_checkpoint,
        environment.action_space.n,
        model_size="12m",
    )
    agent.begin_episode(frame)

    for _ in range(args.steps):
        action = agent.act()
        frame, reward, terminated, truncated, _ = environment.step(action)
        agent.observe(
            frame,
            extrinsic_reward=float(reward),
            terminated=terminated,
            truncated=truncated,
        )
        for report in agent.learn_scheduled():
            print(report)
        if terminated or truncated:
            frame, _ = environment.reset()
            agent.begin_episode(frame)

    if args.checkpoint:
        agent.save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
