"""Fixed-trajectory diagnostics of trained Tiny features; never policy inputs.

Collection restores an actor, forces seeded random actions and never learns.
RAM positions are evaluation labels only. Complete trajectories, including
terminal targets and every reset arrival, are retained with explicit indices.
Run every native-bearing invocation under gpu_host_guard.py.
"""

import argparse
import json
from pathlib import Path
import random
import time

import ale_py
import gymnasium as gym
import numpy as np

import kindle
from atari import DreamerAtariPreprocessing, checkpoint_identity, sha256_file
from atari_vector import require_gpu_budget, require_gpu_device
from kindle._representation_probe import LABEL_SOURCE, positions
from probe_atari_dynamics import ACTION_SEED_XOR


SPLITS = {"train": (9101, 9109, 9127), "validation": (10103, 10111),
          "test": (11113, 11117, 11131)}
PROTOCOL = "kindle-fixed-latent-trajectories-v1"


def collect_trace(agent, environment, seed, steps, check_memory):
    frame, _ = environment.reset(seed=seed)
    agent.begin_episode(frame)
    rng = random.Random(seed ^ ACTION_SEED_XOR)
    features, labels, phase = [], [], []
    current, following, actions, rewards, terminated, truncated, episodes = [], [], [], [], [], [], []
    episode, chunk_phase = 0, 0

    def arrival():
        features.append(np.asarray(agent.visual_observation, dtype=np.float32))
        labels.append(positions("Seaquest", environment.unwrapped.ale.getRAM()))
        phase.append(chunk_phase)
        return len(features) - 1

    index = arrival()
    start_updates, start_actions = agent.learner_step, agent.environment_step
    check_memory()
    for step in range(steps):
        action = rng.randrange(environment.action_space.n)
        mask = [i == action for i in range(environment.action_space.n)]
        if agent.act(action_mask=mask) != action:
            raise RuntimeError("forced action not honored")
        frame, reward, terminal, cutoff, _ = environment.step(action)
        agent.observe(frame, extrinsic_reward=float(reward), terminated=terminal, truncated=cutoff)
        chunk_phase = (chunk_phase + 1) % 16
        next_index = arrival()
        current.append(index)
        following.append(next_index)
        actions.append(action)
        rewards.append(reward)
        terminated.append(terminal)
        truncated.append(cutoff)
        episodes.append(episode)
        index = next_index
        if (step + 1) % 128 == 0:
            check_memory()
        if (terminal or cutoff) and step + 1 < steps:
            episode += 1
            frame, _ = environment.reset()
            agent.begin_episode(frame)
            chunk_phase = 0
            index = arrival()
    if agent.learner_step != start_updates or agent.environment_step - start_actions != steps:
        raise RuntimeError("collection changed learner or action accounting")
    values = np.stack(features)
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite fixed features")
    check_memory()
    return dict(features=values, positions=np.stack(labels), phase=np.asarray(phase, np.uint8),
                current=np.asarray(current, np.int32), following=np.asarray(following, np.int32),
                actions=np.asarray(actions, np.int32), rewards=np.asarray(rewards, np.float32),
                terminated=np.asarray(terminated, bool), truncated=np.asarray(truncated, bool),
                episodes=np.asarray(episodes, np.int32))


def collect(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity = checkpoint_identity(args.checkpoint)
    agent = kindle.Agent.restore(str(args.checkpoint), str(args.encoder_checkpoint))
    if agent.config["video_encoder"] != "joint" or not agent.config["actor_critic_gradient"]:
        raise ValueError("requires the declared direct-policy Tiny checkpoint")
    if agent.config["action_count"] != 18 or agent.config["intrinsic_reward_scale"] != 0:
        raise ValueError("requires unassisted full-action Seaquest actor")
    require_gpu_device(agent.gpu_device, "NVIDIA GeForce RTX 5080")
    memory = dict(samples=0, minimum_headroom_bytes=None, maximum_usage_bytes=0)

    def check_memory():
        value = agent.gpu_memory_budget
        require_gpu_budget(value, 2 << 30)
        headroom = value["budget_bytes"] - value["usage_bytes"]
        memory["minimum_headroom_bytes"] = min(headroom, memory["minimum_headroom_bytes"] or headroom)
        memory["maximum_usage_bytes"] = max(value["usage_bytes"], memory["maximum_usage_bytes"])
        memory["samples"] += 1

    check_memory()
    before_steps = agent.learner_step
    manifest = dict(protocol=PROTOCOL, status="running", splits=SPLITS, steps_per_trajectory=args.steps,
                    checkpoint=identity, native_sha256=sha256_file(kindle._native.__file__),
                    source_sha256=sha256_file(__file__), wrapper_sha256=sha256_file(Path(__file__).with_name("atari.py")),
                    ale_py_version=ale_py.__version__, model_provenance=agent.provenance,
                    gpu_device=agent.gpu_device, observation_size="native", sticky_actions=.25,
                    full_action_space=True, action_repeat=4, action_seed_xor=ACTION_SEED_XOR,
                    label_source=LABEL_SOURCE, labels="RAM positions: diagnostic targets only, never actor inputs",
                    files=[])
    gym.register_envs(ale_py)
    environment = DreamerAtariPreprocessing(gym.make("ALE/Seaquest-v5", frameskip=1,
        repeat_action_probability=.25, full_action_space=True), noop_max=0,
        max_episode_frames=100000, screen_size=None)
    try:
        for split, seeds in SPLITS.items():
            for seed in seeds:
                data = collect_trace(agent, environment, seed, args.steps, check_memory)
                if data["features"].shape[1] != 7 * 7 * 64:
                    raise ValueError("not the production pooled Tiny latent")
                path = args.output / f"{split}-{seed}.npz"
                with path.open("xb") as stream:
                    np.savez(stream, **data)
                row = dict(file=path.name, split=split, seed=seed, sha256=sha256_file(path),
                           actions=args.steps, arrivals=len(data["features"]),
                           positive_rewards=int(np.count_nonzero(data["rewards"] > 0)),
                           terminals=int(data["terminated"].sum()), truncations=int(data["truncated"].sum()))
                manifest["files"].append(row)
                print(json.dumps(row), flush=True)
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        environment.close()
    after = args.output / "frozen-after"
    agent.save_checkpoint(str(after))
    if checkpoint_identity(after)["tensor_sha256"] != identity["tensor_sha256"]:
        raise RuntimeError("frozen collection changed checkpoint tensors")
    if agent.learner_step != before_steps:
        raise RuntimeError("collection updated the learner")
    manifest.update(status="complete", learner_updates=0, tensor_hashes_unchanged=True,
                    sampled_gpu_budget=memory, seconds=time.monotonic() - started)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="complete", learner_updates=0, trajectories=len(manifest["files"]))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("encoder_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=4096)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    collect(args)


if __name__ == "__main__":
    main()
