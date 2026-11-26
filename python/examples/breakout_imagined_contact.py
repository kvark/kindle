"""Compare a frozen Breakout agent's imagined life losses with the emulator's, for the same action sequences.

Evaluation only. One lane plays itself under the published protocol (sticky .25,
18 actions) with its belief restarted at each lost life. Whenever the ball is
visible, descending and at or below --min-ball-y, the world model is rolled
open loop from the live posterior for each fixed action sequence
(`Agent.prior_behavior_rollout`: reward, continuation probability and critic
value per imagined step), and the emulator is cloned and stepped through the
same sequence --clones times with fresh sticky draws to record the action on
which a life is lost. Every action also gets a one-step rollout of the action
actually chosen, which checks the alignment of state and label against the real
outcome. Labels never enter the agent; nothing is fitted here.

    python breakout_imagined_contact.py CHECKPOINT OUTPUT.npz [--actions N] [--seed S]
"""
import argparse
import json
from pathlib import Path
import time

import ale_py
import gymnasium as gym
import numpy as np

import kindle
from atari import DreamerAtariPreprocessing, checkpoint_identity

HORIZON = 8
STAY, RIGHT, LEFT = 0, 3, 4
SEQUENCES = {
    "stay": [STAY] * HORIZON,
    "right": [RIGHT] * HORIZON,
    "left": [LEFT] * HORIZON,
    "tap-right": [RIGHT] + [STAY] * (HORIZON - 1),
    "tap-left": [LEFT] + [STAY] * (HORIZON - 1),
    "right2": [RIGHT] * 2 + [STAY] * (HORIZON - 2),
    "left2": [LEFT] * 2 + [STAY] * (HORIZON - 2),
}


def labels(ale):
    ram = ale.getRAM()
    ball = (float(ram[99]) - 48, float(ram[101]) + 9) if 0 < ram[101] <= 187 else (np.nan, np.nan)
    row = ale.getScreen()[190, 8:152]
    values, counts = np.unique(row, return_counts=True)
    counts = counts[values != 0]
    return ball[0], ball[1], float(ram[72]) - 39, int(counts.max()) if len(counts) else 0


def emulator_loss_step(ale, action_set, sequence):
    """Action index (1-based) on which a life is lost under the sequence, 0 if none. Restores the emulator."""
    saved, lives, lost = ale.cloneState(), ale.lives(), 0
    for step, action in enumerate(sequence, 1):
        for _ in range(4):
            ale.act(action_set[action])
            if ale.lives() < lives or ale.game_over():
                lost = step
                break
        if lost:
            break
    ale.restoreState(saved)
    return lost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actions", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=6660101)
    parser.add_argument("--clones", type=int, default=2)
    parser.add_argument("--min-ball-y", type=float, default=100.0)
    args = parser.parse_args()
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=.25, full_action_space=True)
    env = DreamerAtariPreprocessing(env, noop_max=0, max_episode_frames=100_000, screen_size=None)
    frame = env.reset(seed=args.seed)[0]
    ale, action_set = env.unwrapped.ale, env.unwrapped._action_set
    agent = kindle.Agent.restore_rgb(str(args.checkpoint))
    agent.begin_episode(frame)
    lives, score, returns, previous_y = ale.lives(), 0.0, [], np.nan
    step_rows = dict(action=[], ball_x=[], ball_y=[], paddle_x=[], paddle_width=[], lives=[], score=[], reward=[],
                     lost=[], ended=[], value=[], own_continuation=[], own_value=[], probed=[])
    probe_rows = dict(step=[], continuation=[], reward=[], value=[], emulator_lost=[])
    names, started = list(SEQUENCES), time.perf_counter()
    for index in range(args.actions):
        action = int(agent.act(greedy=False))
        ball_x, ball_y, paddle_x, width = labels(ale)
        _, own_continuation, own_value = agent.prior_behavior_rollout([action])
        probed = bool(np.isfinite(ball_y) and np.isfinite(previous_y) and ball_y > previous_y and ball_y >= args.min_ball_y)
        if probed:
            rollouts = [agent.prior_behavior_rollout(SEQUENCES[name]) for name in names]
            probe_rows["step"].append(index)
            probe_rows["reward"].append(np.asarray([r[0] for r in rollouts], np.float32))
            probe_rows["continuation"].append(np.asarray([r[1] for r in rollouts], np.float32))
            probe_rows["value"].append(np.asarray([r[2] for r in rollouts], np.float32))
            probe_rows["emulator_lost"].append(np.asarray(
                [[emulator_loss_step(ale, action_set, SEQUENCES[name]) for _ in range(args.clones)] for name in names], np.int8))
        for name, value in (("action", action), ("ball_x", ball_x), ("ball_y", ball_y), ("paddle_x", paddle_x),
                            ("paddle_width", width), ("lives", lives), ("score", score),
                            ("value", float(agent.posterior_value_prediction())),
                            ("own_continuation", float(own_continuation[0])), ("own_value", float(own_value[0])),
                            ("probed", probed)):
            step_rows[name].append(value)
        previous_y = ball_y
        frame, reward, terminated, truncated, _ = env.step(action)
        score += float(reward)
        lost = ale.lives() < lives and not (terminated or truncated)
        lives = ale.lives()
        agent.observe(frame, float(reward), 0.0, bool(terminated or lost), bool(truncated))
        step_rows["reward"].append(float(reward))
        step_rows["lost"].append(bool(lost or terminated or truncated))
        step_rows["ended"].append(bool(terminated or truncated))
        if terminated or truncated:
            returns.append(score)
            score, frame = 0.0, env.reset()[0]
            lives = ale.lives()
        if lost or terminated or truncated:
            agent.begin_episode(frame)
            previous_y = np.nan
    meta = dict(protocol="kindle-breakout-imagined-contact-v1", checkpoint=checkpoint_identity(args.checkpoint),
                seed=args.seed, actions=args.actions, horizon=HORIZON, clones=args.clones, min_ball_y=args.min_ball_y,
                sequences=SEQUENCES, completed_returns=returns, collect_seconds=time.perf_counter() - started,
                layout="step arrays are [action]; labels are read before the action executes; lost marks the action on "
                       "which a life ended; probe arrays are [probe, sequence, imagined step] and emulator_lost is "
                       "[probe, sequence, clone] holding the 1-based action on which a life was lost, 0 if none")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(rows) for name, rows in step_rows.items()}
    arrays.update({"probe_" + name: np.stack(rows) if rows else np.zeros(0) for name, rows in probe_rows.items()})
    np.savez_compressed(args.output, meta=json.dumps(meta), **arrays)
    print(json.dumps(dict(actions=args.actions, probes=len(probe_rows["step"]), games=len(returns),
                          mean_return=float(np.mean(returns)) if returns else None,
                          lives_lost=int(np.sum(step_rows["lost"])), collect_seconds=meta["collect_seconds"])))


if __name__ == "__main__":
    main()
