"""Record independent random-play RGB/RAM clips for offline representation probes.

No GPU, learner, privileged policy input, training checkpoints or data selection
by label/score. Whole seeds belong to exactly one split. Only complete causal
clips are used; discarded episode tails and all actual interactions are counted.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import time

import ale_py
import gymnasium as gym
import numpy as np

from atari import DreamerAtariPreprocessing, sha256_file
from kindle._representation_probe import CLIP_LENGTH, GAMES, LABEL_SOURCE, PROTOCOL, SPLITS, clip_targets, target_names


def collect(environment, game, seed, count):
    environment.reset(seed=seed)
    rng = random.Random(seed ^ 0x2A27_0927)
    clips, rams, actions, rewards, frame_counts, episodes = [], [], [], [], [], []
    pending = []
    episode = decisions = discarded = 0
    maximum = count * CLIP_LENGTH * 4
    while len(clips) < count and decisions < maximum:
        action = rng.randrange(environment.action_space.n)
        before = environment.executed_action_frames
        frame, reward, terminal, truncated, _ = environment.step(action)
        elapsed = environment.executed_action_frames - before
        ram = np.asarray(environment.unwrapped.ale.getRAM(), dtype=np.uint8).copy()
        decisions += 1
        pending.append((frame.copy(), ram, action, float(reward), elapsed))
        # Do not use terminal/fake-cutoff frames whose last-two-frame pool can
        # refer to an earlier repeat. Keep their counts, never cross a reset.
        if terminal or truncated:
            discarded += len(pending)
            pending.clear()
            episode += 1
            environment.reset()
        elif len(pending) == CLIP_LENGTH:
            pixels, memory, controls, scores, elapsed = zip(*pending)
            clips.append(np.stack(pixels))
            rams.append(np.stack(memory))
            actions.append(controls)
            rewards.append(scores)
            frame_counts.append(elapsed)
            episodes.append(episode)
            pending.clear()
    if len(clips) != count:
        raise RuntimeError(f"{game}/{seed}: only {len(clips)} complete clips in {decisions} actions")
    targets = np.stack([clip_targets(game, r, f) for r, f in zip(rams, frame_counts)])
    return dict(rgb=np.stack(clips), ram=np.stack(rams), actions=np.asarray(actions, dtype=np.uint8),
                rewards=np.asarray(rewards, dtype=np.float32), executed_frames=np.asarray(frame_counts, dtype=np.uint8),
                episodes=np.asarray(episodes, dtype=np.int32), targets=targets), dict(
                    actions=decisions, executed_action_frames=environment.executed_action_frames,
                    completed_episodes=episode, discarded_tail_actions=discarded,
                    target_counts=np.isfinite(targets).sum(axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clips-per-seed", type=int, default=256)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=GAMES)
    args = parser.parse_args()
    if args.clips_per_seed <= 0:
        parser.error("--clips-per-seed must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    gym.register_envs(ale_py)
    started = time.monotonic()
    manifest = dict(protocol=PROTOCOL, status="running", clips_per_seed=args.clips_per_seed,
                    clip_length=CLIP_LENGTH, splits=SPLITS, games=list(args.games),
                    observation="native last-two-frame max-pooled RGB; no resizing",
                    action_repeat=4, sticky_probability=0.25, full_actions=True, reset_noops=0,
                    max_episode_frames=100_000, policy="uniform random over all 18 actions",
                    label_source=LABEL_SOURCE, label_use="offline probes only; never agent input",
                    packages={name: importlib.metadata.version(name) for name in ("ale-py", "gymnasium", "numpy")},
                    files=[])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for game in args.games:
        for split, seeds in SPLITS.items():
            for seed in seeds:
                base = gym.make(f"ALE/{game}-v5", frameskip=1, repeat_action_probability=0.25, full_action_space=True)
                environment = DreamerAtariPreprocessing(base, noop_max=0, max_episode_frames=100_000, screen_size=None)
                try:
                    arrays, stats = collect(environment, game, seed, args.clips_per_seed)
                finally:
                    environment.close()
                path = args.output / f"{game.lower()}-{split}-{seed}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                row = dict(game=game, split=split, seed=seed, file=path.name, sha256=sha256_file(path),
                           targets=target_names(game), **stats)
                manifest["files"].append(row)
                print(json.dumps(row), flush=True)
    manifest.update(status="complete", seconds=time.monotonic() - started)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
