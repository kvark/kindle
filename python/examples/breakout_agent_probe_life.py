"""Linear ball and paddle readout from a frozen agent's state on held-out teacher play.

Evaluation only. The edge-correcting pixel teacher plays every action on fixed
held-out seeds under the published protocol (sticky .25), so every checkpoint
observes identical trajectories; the agent only observes and its action choices
are overridden. This variant of breakout_agent_probe_actions.py (--minimal-actions
for agents trained on Breakout's four-action set, same trajectories) adds
--life-loss-reset: the agent's belief restarts from the same frame when the
teacher loses a life, as life-loss-terminal training and --life-loss-reset
evaluation do. Without it the belief runs through lost lives, which an agent
trained with the life terminal never saw. The trajectories do not depend on
the flag; lives lost per lane are recorded either way. RAM positions label the
state after each observed frame and never enter the agent. Ridge regression from the actor's
input (the posterior model state) is fitted on lanes 0-3, its penalty chosen on
lanes 4-5 and scored on lanes 6-7, in native source pixels. Velocities are
one-action RAM differences within an episode.

    python breakout_agent_probe_life.py CHECKPOINT OUTPUT.json [--life-loss-reset] [--minimal-actions]
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
from kindle._breakout_guidance import MINIMAL_TO_FULL, PixelPolicy
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
    parser.add_argument("--minimal-actions", action="store_true",
                        help="the checkpoint acts on Breakout's four-action vocabulary")
    parser.add_argument("--life-loss-reset", action="store_true",
                        help="restart the agent's belief from the same frame when a life is lost")
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
    teachers = [PixelPolicy(edge_reset=True) for _ in range(LANES)]
    agent = kindle.VectorAgent.restore_rgb(str(args.checkpoint), LANES)
    ids = list(range(LANES))
    agent.begin_episodes(ids, frames)
    labels = [np.array(positions("Breakout", ale.getRAM())[:3], np.float64) for ale in ales]
    previous = [None] * LANES
    features, targets = [[] for _ in ids], [[] for _ in ids]
    started, episodes, returns = time.perf_counter(), [0] * LANES, [0.0] * LANES
    lives, lives_lost = [ale.lives() for ale in ales], [0] * LANES
    for _ in range(args.actions):
        overrides = [teacher.action(frame) for teacher, frame in zip(teachers, frames)]
        if args.minimal_actions:
            overrides = [MINIMAL_TO_FULL.index(action) for action in overrides]
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
        lost = [lane for lane in ids if ales[lane].lives() < lives[lane] and not (terminated[lane] or truncated[lane])]
        lives = [ale.lives() for ale in ales]
        for lane in lost:
            lives_lost[lane] += 1
        restarts = lost if args.life_loss_reset else []
        agent.observe(ids, list(frames), rewards, [terminated[lane] or lane in restarts for lane in ids], truncated)
        if restarts:
            agent.begin_episodes(restarts, [frames[lane] for lane in restarts])
        resets = [lane for lane in ids if terminated[lane] or truncated[lane]]
        for lane in ids:
            previous[lane] = None if lane in resets else labels[lane]
            labels[lane] = np.array(positions("Breakout", ales[lane].getRAM())[:3], np.float64)
        if resets:
            for lane in resets:
                frames[lane] = environments[lane].reset()[0]
                teachers[lane] = PixelPolicy(edge_reset=True)
                labels[lane] = np.array(positions("Breakout", ales[lane].getRAM())[:3], np.float64)
                lives[lane] = ales[lane].lives()
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
    result = dict(protocol="kindle-breakout-agent-probe-v1", minimal_actions=args.minimal_actions,
                  life_loss_reset=args.life_loss_reset, lives_lost=lives_lost,
                  checkpoint=checkpoint_identity(args.checkpoint),
                  seed=args.seed, actions_per_lane=args.actions, feature_width=int(x[0].shape[1]),
                  episodes_completed=episodes, teacher_returns=returns, collect_seconds=collected,
                  metrics=metrics, summary=summary, selections=dict(zip(TARGETS, selections)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(dict(summary=summary, **{name: dict(mae=m["mae"], r2=m["r2"]) for name, m in metrics.items()})))


if __name__ == "__main__":
    main()
