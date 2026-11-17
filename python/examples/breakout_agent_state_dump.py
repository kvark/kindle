"""Dump a frozen Breakout agent's own-play model state with emulator labels, for offline readouts.

Evaluation only. The agent plays eight lanes itself under the published protocol
(sticky .25, 18 actions unless --minimal-actions) with its belief restarted at
each lost life (--life-loss-reset semantics, always on here). After every action
choice the actor's input (the posterior model state), logits and the critic's
value are recorded with
RAM labels read before the action executes: ball centre x and top y, paddle
centre x, lives, and the paddle's on-screen width. Labels never enter the agent.
Nothing is fitted here; the output is one compressed .npz for CPU analysis.

    python breakout_agent_state_dump.py CHECKPOINT OUTPUT.npz [--actions N] [--greedy] [--seed S]
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

LANES = 8


def labels(ale):
    ram = ale.getRAM()
    ball = (float(ram[99]) - 48, float(ram[101]) + 9) if 0 < ram[101] <= 187 else (np.nan, np.nan)
    row = ale.getScreen()[190, 8:152]
    values, counts = np.unique(row, return_counts=True)
    counts = counts[values != 0]
    return ball[0], ball[1], float(ram[72]) - 39, int(counts.max()) if len(counts) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actions", type=int, default=6000, help="actions per lane")
    parser.add_argument("--seed", type=int, default=6660001)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--minimal-actions", action="store_true")
    args = parser.parse_args()
    gym.register_envs(ale_py)
    environments, frames = [], []
    for lane in range(LANES):
        env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=.25,
                       full_action_space=not args.minimal_actions)
        env = DreamerAtariPreprocessing(env, noop_max=0, max_episode_frames=100_000, screen_size=None)
        environments.append(env)
        frames.append(env.reset(seed=args.seed + lane)[0])
    ales = [env.unwrapped.ale for env in environments]
    agent = kindle.VectorAgent.restore_rgb(str(args.checkpoint), LANES)
    ids = list(range(LANES))
    agent.begin_episodes(ids, frames)
    lives = [ale.lives() for ale in ales]
    columns = dict(state=[], logits=[], value=[], action=[], ball_x=[], ball_y=[], paddle_x=[], paddle_width=[], lives=[],
                   reward=[], lost=[], ended=[], score=[])
    scores, returns, started = [0.0] * LANES, [], time.perf_counter()
    for _ in range(args.actions):
        selected = agent.act(greedy=args.greedy)
        inputs, logits = agent.policy_inputs_and_logits()
        values, _ = agent.behavior_predictions()
        before = [labels(ale) for ale in ales]
        columns["state"].append(np.asarray(inputs, np.float32).reshape(LANES, -1).astype(np.float16))
        columns["logits"].append(np.asarray(logits, np.float32).reshape(LANES, -1))
        columns["action"].append(np.asarray(selected, np.int16))
        columns["value"].append(np.asarray(values, np.float32).reshape(LANES, -1))
        for index, name in enumerate(("ball_x", "ball_y", "paddle_x", "paddle_width")):
            columns[name].append(np.asarray([row[index] for row in before], np.float32))
        columns["lives"].append(np.asarray(lives, np.int16))
        columns["score"].append(np.asarray(scores, np.float32))
        rewards, terminated, truncated = [], [], []
        for lane, (env, action) in enumerate(zip(environments, selected)):
            frame, reward, term, trunc, _ = env.step(action)
            frames[lane] = frame
            rewards.append(float(reward)); terminated.append(bool(term)); truncated.append(bool(trunc))
            scores[lane] += float(reward)
        lost = [lane for lane in ids if ales[lane].lives() < lives[lane] and not (terminated[lane] or truncated[lane])]
        lives = [ale.lives() for ale in ales]
        agent.observe(ids, list(frames), rewards, [terminated[lane] or lane in lost for lane in ids], truncated)
        if lost:
            agent.begin_episodes(lost, [frames[lane] for lane in lost])
        resets = [lane for lane in ids if terminated[lane] or truncated[lane]]
        columns["reward"].append(np.asarray(rewards, np.float32))
        columns["lost"].append(np.asarray([lane in lost or lane in resets for lane in ids], bool))
        columns["ended"].append(np.asarray([lane in resets for lane in ids], bool))
        if resets:
            for lane in resets:
                returns.append(scores[lane]); scores[lane] = 0.0
                frames[lane] = environments[lane].reset()[0]
                lives[lane] = ales[lane].lives()
            agent.begin_episodes(resets, [frames[lane] for lane in resets])
    meta = dict(protocol="kindle-breakout-agent-state-dump-v1", checkpoint=checkpoint_identity(args.checkpoint),
                seed=args.seed, greedy=args.greedy, minimal_actions=args.minimal_actions, actions_per_lane=args.actions,
                lanes=LANES, completed_returns=returns, collect_seconds=time.perf_counter() - started,
                layout="arrays are [action, lane]; labels are read before the action executes; lost marks the "
                       "action on which a life ended; ended marks game over")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=json.dumps(meta), **{name: np.stack(rows) for name, rows in columns.items()})
    print(json.dumps(dict(rows=args.actions * LANES, feature_width=int(columns["state"][0].shape[1]),
                          games=len(returns), mean_return=float(np.mean(returns)) if returns else None,
                          collect_seconds=meta["collect_seconds"])))


if __name__ == "__main__":
    main()
