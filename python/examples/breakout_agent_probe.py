"""Linear ball and paddle readout from a frozen agent's state on held-out teacher play.

Evaluation only. The edge-correcting pixel teacher plays every action on fixed
held-out seeds under the published protocol (sticky .25, full action space), so
every checkpoint observes identical trajectories; the agent only observes and
its action choices are overridden. RAM positions label the state after each
observed frame and never enter the agent. Ridge regression from the actor's
input (the posterior model state) is fitted on lanes 0-3, its penalty chosen on
lanes 4-5 and scored on lanes 6-7, in native source pixels. Velocities are
one-action RAM differences within an episode.

    python breakout_agent_probe.py CHECKPOINT OUTPUT.json [--actions 1536] [--seed 5550001]
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
from kindle._breakout_guidance import PixelPolicy
from kindle._representation_probe import positions, regression_metrics, ridge_probe

LANES = 8
TARGETS = ("ball_x", "ball_y", "paddle_x", "ball_dx", "ball_dy")
TRAIN, VALIDATION, TEST = (0, 1, 2, 3), (4, 5), (6, 7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actions", type=int, default=1536, help="actions per lane")
    parser.add_argument("--seed", type=int, default=5550001)
    args = parser.parse_args()
    gym.register_envs(ale_py)
    environments, frames = [], []
    for lane in range(LANES):
        env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=.25, full_action_space=True)
        env = DreamerAtariPreprocessing(env, noop_max=0, max_episode_frames=100_000, screen_size=None)
        environments.append(env)
        frames.append(env.reset(seed=args.seed + lane)[0])
    ales = [env.unwrapped.ale for env in environments]
    teachers = [PixelPolicy(edge_reset=True) for _ in range(LANES)]
    agent = kindle.VectorAgent.restore_rgb(str(args.checkpoint), LANES)
    ids = list(range(LANES))
    agent.begin_episodes(ids, frames)
    labels = [np.array(positions("Breakout", ale.getRAM())[:3], np.float64) for ale in ales]
    previous = [None] * LANES
    features, targets = [[] for _ in ids], [[] for _ in ids]
    started, episodes, returns = time.perf_counter(), [0] * LANES, [0.0] * LANES
    for _ in range(args.actions):
        overrides = [teacher.action(frame) for teacher, frame in zip(teachers, frames)]
        selected = agent.act(False, overrides)
        if selected != overrides:
            raise RuntimeError("teacher override was not honored")
        inputs, _ = agent.policy_inputs_and_logits()
        state = np.asarray(inputs, np.float32).reshape(LANES, -1)
        for lane in ids:
            velocity = (labels[lane][:2] - previous[lane][:2]) if previous[lane] is not None else np.full(2, np.nan)
            features[lane].append(state[lane])
            targets[lane].append(np.concatenate([labels[lane], velocity]))
        rewards, terminated, truncated = [], [], []
        for lane, (env, action) in enumerate(zip(environments, overrides)):
            frame, reward, term, trunc, _ = env.step(action)
            frames[lane] = frame
            rewards.append(float(reward)); terminated.append(bool(term)); truncated.append(bool(trunc))
            returns[lane] += float(reward)
        agent.observe(ids, list(frames), rewards, terminated, truncated)
        resets = [lane for lane in ids if terminated[lane] or truncated[lane]]
        for lane in ids:
            previous[lane] = None if lane in resets else labels[lane]
            labels[lane] = np.array(positions("Breakout", ales[lane].getRAM())[:3], np.float64)
        if resets:
            for lane in resets:
                frames[lane] = environments[lane].reset()[0]
                teachers[lane] = PixelPolicy(edge_reset=True)
                labels[lane] = np.array(positions("Breakout", ales[lane].getRAM())[:3], np.float64)
                episodes[lane] += 1
            agent.begin_episodes(resets, [frames[lane] for lane in resets])
    collected = time.perf_counter() - started
    x = {lane: np.stack(features[lane]) for lane in ids}
    y = {lane: np.stack(targets[lane]) for lane in ids}
    stack = lambda lanes, d: np.concatenate([d[lane] for lane in lanes])
    prediction, selections = ridge_probe(stack(TRAIN, x), stack(TRAIN, y), stack(VALIDATION, x),
                                         stack(VALIDATION, y), stack(TEST, x))
    truth = stack(TEST, y)
    metrics = dict(zip(TARGETS, regression_metrics(prediction, truth)))
    ball = np.isfinite(truth[:, 0]) & np.isfinite(truth[:, 1])
    distance = np.hypot(prediction[ball, 0] - truth[ball, 0], prediction[ball, 1] - truth[ball, 1])
    summary = dict(ball_within_4px=float(np.mean(distance <= 4)), ball_within_8px=float(np.mean(distance <= 8)),
                   ball_rows=int(ball.sum()), test_rows=len(truth))
    result = dict(protocol="kindle-breakout-agent-probe-v1", checkpoint=checkpoint_identity(args.checkpoint),
                  seed=args.seed, actions_per_lane=args.actions, feature_width=int(x[0].shape[1]),
                  episodes_completed=episodes, teacher_returns=returns, collect_seconds=collected,
                  metrics=metrics, summary=summary, selections=dict(zip(TARGETS, selections)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(dict(summary=summary, **{name: dict(mae=m["mae"], r2=m["r2"]) for name, m in metrics.items()})))


if __name__ == "__main__":
    main()
