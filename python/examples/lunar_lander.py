"""Train a kindle agent on gymnasium's LunarLander.

kindle is an intrinsic-reward agent — it does *not* consume the env's
extrinsic reward during training. Instead, we give it domain knowledge
through a `homeo_fn(obs) -> list[{value, target, tolerance}]` that
encodes homeostatic targets (low altitude, level attitude, low speed).
The agent's frozen reward circuit turns those into a penalty signal.

The env's extrinsic reward is collected and printed as a pure observer
metric — it never feeds the learner.

Usage:
    pip install "gymnasium[box2d]"
    python python/examples/lunar_lander.py --steps 50000

LunarLander observation layout:
    [0] x          horizontal position
    [1] y          altitude
    [2] vx         horizontal velocity
    [3] vy         vertical velocity
    [4] angle      body angle (radians)
    [5] ang_vel    angular velocity
    [6] leg1       left leg contact (0/1)
    [7] leg2       right leg contact (0/1)
"""

from __future__ import annotations

import argparse
import math
import sys


def lunar_lander_homeo(obs):
    """Homeostatic targets: land level, land slow, and — above all — don't
    crash into the ground.

    The first term is a severe crash-risk signal. It multiplies downward
    speed by proximity to the ground, so it fires only when the lander is
    both falling *and* near the surface — the precondition of a crash.
    The tolerance is zero and the value is pre-scaled 10× so the penalty
    dwarfs the other homeostats whenever the agent gets close to a
    ground impact. The other three terms are mild shaping signals for
    steady descent.
    """
    altitude = float(obs[1])
    vy = float(obs[3])
    speed = math.sqrt(obs[2] * obs[2] + obs[3] * obs[3])

    # Downward speed only — upward motion isn't a crash risk.
    descent = max(0.0, -vy)
    # Smooth proximity-to-ground weight: ~1 at the pad, decays aloft.
    proximity = math.exp(-max(0.0, altitude) * 2.0)
    crash_risk = descent * proximity

    return [
        # Severe: any crash risk triggers a large penalty immediately.
        {"value": crash_risk * 10.0, "target": 0.0, "tolerance": 0.0},
        # Mild shaping: level body.
        {"value": float(obs[4]), "target": 0.0, "tolerance": 0.1},
        # Mild shaping: keep low.
        {"value": altitude, "target": 0.0, "tolerance": 0.2},
        # Mild shaping: low overall speed.
        {"value": float(speed), "target": 0.0, "tolerance": 0.3},
    ]


def main() -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium isn't installed. Try: pip install 'gymnasium[box2d]'", file=sys.stderr)
        return 1

    import kindle

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=2_000)
    args = parser.parse_args()

    env = gym.make(args.env)
    obs, _info = env.reset(seed=args.seed)
    obs_dim = len(obs)
    num_actions = int(env.action_space.n)

    print(
        f"env={args.env} obs_dim={obs_dim} num_actions={num_actions} "
        f"steps={args.steps} seed={args.seed}"
    )

    agent = kindle.Agent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        env_id=1,
        seed=args.seed,
    )

    returns: list[float] = []
    episode_return = 0.0
    obs_list = [float(x) for x in obs]

    for step in range(args.steps):
        action = agent.act(obs_list)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)

        next_list = [float(x) for x in next_obs]
        agent.observe(next_list, action, homeostatic=lunar_lander_homeo(next_list))

        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            obs_list = [float(x) for x in obs]
            agent.mark_boundary()
        else:
            obs_list = next_list

        if args.log_every and step > 0 and step % args.log_every == 0:
            d = agent.diagnostics()
            recent = returns[-20:]
            avg = sum(recent) / max(1, len(recent))
            print(
                f"step={step:>7} episodes={len(returns):>4} "
                f"avg_return(last20)={avg:+8.1f} "
                f"surprise={d['reward_surprise']:+6.3f} "
                f"novelty={d['reward_novelty']:+6.3f} "
                f"homeo={d['reward_homeo']:+7.3f} "
                f"order={d['reward_order']:+6.3f} "
                f"entropy={d['policy_entropy']:.2f}"
            )

    env.close()
    if returns:
        recent = returns[-20:]
        print(
            f"\nFinished {len(returns)} episodes. "
            f"avg_return(last20)={sum(recent) / len(recent):.1f}, "
            f"best={max(returns):.1f}, worst={min(returns):.1f}"
        )
    else:
        print(f"\nNo episodes completed in {args.steps} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
