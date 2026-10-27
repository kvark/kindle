"""CPU-only reconstruction of every action and pixel-rule override in a guided training log."""

import argparse
from contextlib import ExitStack, closing
import json
from pathlib import Path
import time

import ale_py
import gymnasium as gym

import atari
from audit_atari import require, sha256
from check_atari_adapter import rom_identity
from kindle._qbert_guidance import GUIDANCE_KINDS, QbertGuidance
from kindle._breakout_guidance import GUIDANCE_KINDS as BREAKOUT_GUIDANCE_KINDS, BreakoutGuidance
from kindle._vector_audit import audit
from replay_atari import TaskMonitor, verify_replay_identity


def read_guided_run(path):
    accounting = audit(path)
    require(accounting['budget_complete'], "guidance replay requires a completed action budget")
    with Path(path).open() as source:
        header = json.loads(next(source))
        end = None
        for line in source:
            end = json.loads(line)
    require(header['mode'] == 'train' and header.get('exploration', {}).get('kind') in (*GUIDANCE_KINDS, *BREAKOUT_GUIDANCE_KINDS),
            "requires declared pixel guidance training")
    return dict(start=header, end=end, accounting=accounting, sha256=sha256(path))


def replay_rows(source, environments, monitors, frames, guidance, deadline):
    episodes = []
    overrides = None
    for line in source:
        require(time.monotonic() < deadline, "guidance replay exceeded wall cap")
        row = json.loads(line)
        if row['event'] == 'transition':
            overrides = guidance.actions(frames)
            require(overrides == row['action_overrides'], "pixel guidance action differs")
            require(guidance.frame_sha256 == row['guidance_frame_sha256'], "guidance input pixels differ")
            require(all(forced is None or forced == action for forced, action in zip(overrides, row['actions'])),
                    "guidance override was not executed")
            results = [environment.step(action) for environment, action in zip(environments, row['actions'])]
            frames = [result[0] for result in results]
            expected = dict(rewards=[result[1] for result in results],
                            terminated=[result[2] for result in results], truncated=[result[3] for result in results],
                            executed_action_frames=[env.executed_action_frames for env in environments])
            for key, value in expected.items():
                require(row[key] == value, f'guidance replay diverged at action {row["run_step"]}: {key}')
        elif row['event'] == 'episode':
            stream = row['stream']
            result = monitors[stream].task.result(row['terminated'], row['truncated'])
            require(result['episode_score'] == row['episode_return'], "guidance replay episode score differs")
            episodes.append(dict(**row, guided=guidance.selected_episodes[stream], task_outcome=result))
        elif row['event'] == 'reset':
            for stream in row['streams']:
                frames[stream], _ = environments[stream].reset()
            guidance.reset(row['streams'])
        elif row['event'] in ('progress', 'run_end'):
            require(guidance.overridden_actions == row['overridden_actions'], "guidance action totals differ")
            require(guidance.episode_counts == row['guidance_episodes'], "guidance episode totals differ")
    partial = [dict(stream=lane, **monitor.task.result(False, False)) for lane, monitor in enumerate(monitors)]
    return episodes, partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wall-cap', type=int, default=300)
    args = parser.parse_args()
    require(args.wall_cap > 0, "wall cap must be positive")
    require(not args.output.exists() and args.output.parent.is_dir(), "output must be a fresh path")
    started = time.monotonic()
    run = read_guided_run(args.log)
    header = run['start']
    manifest = json.loads(args.source_manifest.read_text())
    gym.register_envs(ale_py)
    rom = rom_identity(header['environment'])
    verify_replay_identity(header, manifest, rom)
    cls = BreakoutGuidance if header['exploration']['kind'] in BREAKOUT_GUIDANCE_KINDS else QbertGuidance
    guidance = cls(header['exploration'], header['num_envs'], header['config']['action_count'])
    environments, monitors, frames = [], [], []
    with ExitStack() as stack:
        for seed in header['environment_seeds']:
            raw = stack.enter_context(closing(gym.make(header['environment'], frameskip=1,
                                                       repeat_action_probability=header['sticky_actions'],
                                                       full_action_space=True)))
            monitor = TaskMonitor(raw, header['environment'], rom['sha256'])
            env = atari.DreamerAtariPreprocessing(monitor, noop_max=0, max_episode_frames=100000, screen_size=None)
            require(list(env.action_meanings) == header['action_meanings'], "changed guidance action vocabulary")
            frame, _ = env.reset(seed=seed)
            environments.append(env)
            monitors.append(monitor)
            frames.append(frame)
        with args.log.open() as source:
            require(json.loads(next(source)) == header, "guidance source changed")
            episodes, partial = replay_rows(source, environments, monitors, frames, guidance, started + args.wall_cap)
        require([env.executed_action_frames for env in environments] == run['end']['executed_action_frames'],
                "guidance replay total frames differ")
        require([env.emulator_resets for env in environments] == run['end']['emulator_resets'],
                "guidance replay reset counts differ")
    require(sha256(args.log) == run['sha256'], "guidance source changed during replay")
    result = dict(protocol=('kindle-breakout-guidance-replay-v1' if cls is BreakoutGuidance else 'kindle-qbert-guidance-replay-v1'), source_log=str(args.log.resolve()),
                  source_log_sha256=run['sha256'], source_header=header, source_accounting=run['accounting'],
                  source_manifest_sha256=sha256(args.source_manifest), replay_script_sha256=sha256(__file__),
                  elapsed_seconds=time.monotonic() - started, agent_constructed=False, learner_updates=0,
                  full_trajectory_replayed=True, exploration_pixel_actions_verified=True,
                  overridden_actions=guidance.overridden_actions, guidance_episodes=guidance.episode_counts,
                  actual_overridden_action_fraction=sum(guidance.overridden_actions) / run['accounting']['actions'],
                  episodes=episodes, partial=partial, reliability_assessed=False,
                  limits='Training trajectories include hand-coded aid; these are not frozen unaided task scores.')
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(output=str(args.output), completed_episodes=len(episodes),
                         overridden_actions=sum(guidance.overridden_actions), pixel_actions_verified=True)))


if __name__ == '__main__':
    main()
