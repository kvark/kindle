"""One bounded, sparse-reward MinAtar screening seed; no frozen competence claim.

Default recipe: eight streams, Dreamer1M/B8/T16/full BPTT/H15/R32, 32,768
aggregate actions. Run seeds 1009/2017/3019 separately with the host GPU guard.
The CPU environment is the strategy's temporary fallback; inference and the
learned observation encoder, world model, policy and replay stay on the GPU.
"""

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import kindle
from kindle._screening import PROTOCOL, pack_minatar
from minatar import Environment
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--game", choices=("breakout", "freeway", "asterix", "seaquest", "space_invaders"), default="breakout")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--report-every", type=int, default=2048)
    parser.add_argument("--model-size", default="1m", choices=("tiny", "1m"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.steps % args.num_envs or args.report_every < 1 or args.report_every % args.num_envs:
        parser.error("steps and report-every must be positive multiples of num-envs")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must fit u32")

    with args.output.open("x") as output:
        def emit(record):
            print(json.dumps(record, allow_nan=False, separators=(",", ":")), file=output, flush=True)

        started = time.perf_counter()
        env_seeds = [(args.seed + i * 1_000_003) % 2**32 for i in range(args.num_envs)]
        environments = [Environment(args.game, sticky_action_prob=.1, difficulty_ramping=True)
                        for _ in env_seeds]
        for env, seed in zip(environments, env_seeds):
            env.seed(seed)
            env.reset()
        config = kindle.default_config(6, args.model_size)
        config.update(seed=args.seed, batch_size=8, batch_length=16, world_backprop_length=16,
                      world_microbatch_size=8, train_ratio=32.0, replay_capacity=16384)
        config["loss_scales"].update(reconstruction=0.0, future_prediction=.25)
        agent = kindle.FeatureVectorAgent(args.num_envs, config)
        device = agent.gpu_device
        expected = os.environ.get("KINDLE_EXPECT_DEVICE_NAME")
        if expected and (device["device_name"] != expected or device["is_software_emulated"]):
            raise RuntimeError(f"unexpected device: {device}")

        def check_memory():
            memory = agent.gpu_memory_budget
            if memory["budget_bytes"] - memory["usage_bytes"] < 2 << 30:
                raise RuntimeError(f"less than 2 GiB Vulkan estimated budget headroom: {memory}")
            return memory

        emit(dict(event="start", protocol=PROTOCOL, game=args.game, seed=args.seed,
                  environment_seeds=env_seeds, config=agent.config, num_envs=args.num_envs,
                  steps=args.steps, report_every=args.report_every,
                  observation="lossless 10x10xC space-to-depth, zero-padded to 7x7x64",
                  observation_shape=environments[0].state_shape(),
                  rewards="unmodified game reward", action_count=6, action_repeat=1,
                  sticky_action_probability=.1, difficulty_ramping=True,
                  exploration_override=None, pretraining=None, intrinsic_reward=None,
                  environment_device="CPU", learner_device=device,
                  provenance=agent.provenance, trainable_parameters=agent.trainable_parameter_counts,
                  native_sha256=hashlib.sha256(Path(kindle._native.__file__).read_bytes()).hexdigest(),
                  minatar_version=importlib.metadata.version("minatar"),
                  construction_seconds=time.perf_counter() - started, memory=check_memory()))
        ids = list(range(args.num_envs))
        agent.begin_episodes(ids, [pack_minatar(env.state()).tolist() for env in environments])
        episode_returns = np.zeros(args.num_envs)
        episode_lengths = np.zeros(args.num_envs, dtype=int)
        counts = np.zeros(args.num_envs, dtype=int)
        histogram = np.zeros(6, dtype=int)
        recent = deque(maxlen=50)
        metrics = []
        curve = []
        for actions_done in range(args.num_envs, args.steps + 1, args.num_envs):
            actions = agent.act()
            np.add.at(histogram, actions, 1)
            results = [env.act(action) for env, action in zip(environments, actions)]
            rewards, terminal = zip(*results)
            episode_returns += rewards
            episode_lengths += 1
            # Consume terminal observation and reward before each stream reset.
            agent.observe(ids, [pack_minatar(env.state()).tolist() for env in environments],
                          list(rewards), list(terminal), [False] * args.num_envs)
            resets = []
            for i, done in enumerate(terminal):
                if done:
                    recent.append(float(episode_returns[i]))
                    emit(dict(event="episode", stream=i, episode=int(counts[i]),
                              actions=actions_done, seconds=time.perf_counter() - started,
                              score=float(episode_returns[i]), length=int(episode_lengths[i])))
                    counts[i] += 1
                    episode_returns[i] = 0
                    episode_lengths[i] = 0
                    environments[i].reset()
                    resets.append(i)
            if resets:
                agent.begin_episodes(resets, [pack_minatar(environments[i].state()).tolist() for i in resets])
            metrics.extend(agent.learn_scheduled())
            if actions_done % args.report_every == 0 or actions_done == args.steps:
                point = dict(actions=actions_done, seconds=time.perf_counter() - started,
                             learner_steps=agent.learner_step, training_debt=agent.training_debt,
                             score=float(np.mean(recent)) if recent else None,
                             completed_episodes=int(counts.sum()), memory=check_memory(),
                             action_histogram=histogram.tolist(), interval_updates=len(metrics),
                             metrics={section: {key: float(np.mean([m[section][key] for m in metrics]))
                                               for key in metrics[0][section]}
                                      for section in ("world", "behavior", "timing")} if metrics else {})
                curve.append(point)
                emit(dict(event="curve", **point))
                metrics.clear()
        assert agent.environment_step == args.steps and agent.learner_step > 0
        assert agent.training_debt < 1.0
        if args.checkpoint:
            agent.save_checkpoint(str(args.checkpoint))
        emit(dict(event="complete", seed=args.seed, actions=agent.environment_step,
                  learner_steps=agent.learner_step, seconds=time.perf_counter() - started,
                  curve=curve, unfinished_returns=episode_returns.tolist(),
                  unfinished_lengths=episode_lengths.tolist(), completed_by_stream=counts.tolist(),
                  action_histogram=histogram.tolist(), memory=check_memory()))


if __name__ == "__main__":
    main()
