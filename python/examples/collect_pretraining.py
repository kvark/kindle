"""Collect fresh random-policy Atari pixels without loading a native learner.

Uses the same published RGB observation wrapper as the active Large campaign.
No rewards, actions or privileged game state are inputs to JEPA. This is
separately counted offline experience, not an untrained gameplay control.
"""

import argparse
import json
from pathlib import Path
import time

import ale_py
import gymnasium as gym
import numpy as np
import PIL

from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing
from kindle._video_pretrain import sha256_file


GAMES = ("Boxing", "Pong", "Freeway", "Breakout", "Qbert")


def collect_recording(environment, directory, name, seed, observations, split):
    if observations < 31:
        raise ValueError("a recording needs at least one 16-frame stride-two clip")
    frame, _ = environment.reset(seed=seed)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("expected uint8 RGB observations")
    frame_path, episode_path = directory / f"{name}-frames.npy", directory / f"{name}-episodes.npy"
    if frame_path.exists() or episode_path.exists():
        raise FileExistsError("recording output already exists")
    frames = np.lib.format.open_memmap(frame_path, mode="w+", dtype=np.uint8,
                                     shape=(observations, *frame.shape))
    episodes = np.lib.format.open_memmap(episode_path, mode="w+", dtype=np.uint32,
                                       shape=(observations,))
    rng = np.random.default_rng(seed)
    episode, terminated_count, truncated_count = 0, 0, 0
    start = time.monotonic()
    for index in range(observations):
        action = int(rng.integers(environment.action_space.n))
        frame, _reward, terminated, truncated, _info = environment.step(action)
        frames[index], episodes[index] = frame, episode
        if terminated or truncated:
            terminated_count += int(terminated)
            truncated_count += int(truncated)
            episode += 1
            if index + 1 < observations:
                environment.reset()
    frames.flush()
    episodes.flush()
    shape = list(frames.shape)
    del frames, episodes
    return {
        "name": name, "split": split, "seed": seed, "observations": observations,
        "shape": shape, "random_actions": observations,
        "executed_emulator_frames": environment.executed_action_frames,
        "reset_noop_frames": environment.reset_noop_frames,
        "emulator_resets": environment.emulator_resets,
        "terminated_episodes": terminated_count, "truncated_episodes": truncated_count,
        "elapsed_seconds": time.monotonic() - start,
        "frames": {"file": frame_path.name, "sha256": sha256_file(frame_path)},
        "episodes": {"file": episode_path.name, "sha256": sha256_file(episode_path)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new corpus directory; refuses overwrite")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-recordings", type=int, required=True)
    parser.add_argument("--train-observations", type=int, required=True)
    parser.add_argument("--validation-observations", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1000 or not 1 <= args.train_recordings < 100:
        parser.error("invalid disjoint recording seed range")
    if min(args.train_observations, args.validation_observations) < 31:
        parser.error("recordings require at least 31 observations")
    args.output.mkdir(parents=True, exist_ok=False)
    gym.register_envs(ale_py)
    protocol = ATARI_PROTOCOLS["published"]
    recordings = []
    for game_index, game in enumerate(GAMES):
        for recording in range(args.train_recordings + 1):
            split = "train" if recording < args.train_recordings else "validation"
            seed = args.seed + game_index * 100 + recording
            count = args.train_observations if split == "train" else args.validation_observations
            environment = DreamerAtariPreprocessing(gym.make(
                f"ALE/{game}-v5", frameskip=1, repeat_action_probability=0.0,
                full_action_space=protocol.full_action_space), noop_max=protocol.noop_max,
                max_episode_frames=protocol.max_episode_frames)
            try:
                record = collect_recording(environment, args.output,
                                           f"{game.lower()}-{seed}", seed, count, split)
                record["environment"] = f"ALE/{game}-v5"
                record["action_meanings"] = list(environment.action_meanings)
                recordings.append(record)
                (args.output / (record["name"] + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                print(json.dumps(record), flush=True)
            finally:
                environment.close()
    declaration = {
        "format": 1, "policy": "uniform-full-action-random", "atari_protocol": "published",
        "observation": "repeat4-maxpool2-Pillow-bilinear-RGB64",
        "recordings": recordings,
        "total_observations": sum(r["observations"] for r in recordings),
        "total_executed_emulator_frames": sum(r["executed_emulator_frames"] for r in recordings),
        "versions": {"numpy": np.__version__, "pillow": PIL.__version__,
                     "gymnasium": gym.__version__, "ale_py": ale_py.__version__},
        "sources": {"collector": sha256_file(__file__),
                    "atari_wrapper": sha256_file(Path(__file__).with_name("atari.py"))},
    }
    (args.output / "corpus.json").write_text(json.dumps(declaration, indent=2) + "\n")


if __name__ == "__main__":
    main()
