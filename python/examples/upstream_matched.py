"""Protocol alignment for the pinned upstream control; model math is unchanged.

The common Gym/ALE wrapper matches Kindle's seeds, repeats, pooling, sticky
actions and artificial time-limit bootstrapping. This is an experiment adapter,
not a second agent implementation. Imports never initialize CUDA or Vulkan.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np


class UpdateSchedule:
    """Kindle/D3 warmup: discard prefill debt, one update on first eligibility."""

    def __init__(self, ratio=256, batch_steps=1024):
        self.ratio = ratio / batch_steps
        self.started = False
        self.credit = 0.0

    def observe(self, actions, ready):
        if self.started:
            self.credit += actions * self.ratio
        if not ready:
            return 0
        if not self.started:
            self.started, self.credit = True, 1.0
        count = int(self.credit)
        self.credit -= count
        return count


def observation(result, *, first=False):
    if first:
        image, _ = result
        reward, terminated, truncated = 0.0, False, False
    else:
        image, reward, terminated, truncated, _ = result
    return dict(image=image, reward=np.float32(reward), is_first=np.bool_(first),
                is_last=np.bool_(terminated or truncated), is_terminal=np.bool_(terminated))


def parameter_fingerprints(parameters):
    result = {}
    for name, values in parameters.items():
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite upstream state: {name}")
        result[name] = hashlib.sha256(values.tobytes()).hexdigest()
    return result


def map_carry(function, *trees):
    """Pinned JAX agent carry: dict/tuple structure with per-stream LIST leaves.

    Keep device arrays opaque; selecting/resetting streams must not read their
    recurrent state back to the CPU or confuse tuple components with streams.
    """
    first = trees[0]
    if isinstance(first, dict):
        if any(not isinstance(t, dict) or t.keys() != first.keys() for t in trees):
            raise ValueError("policy carry structure changed")
        return {key: map_carry(function, *(t[key] for t in trees)) for key in first}
    if isinstance(first, tuple):
        if any(not isinstance(t, tuple) or len(t) != len(first) for t in trees):
            raise ValueError("policy carry structure changed")
        return tuple(map_carry(function, *(t[i] for t in trees)) for i in range(len(first)))
    if not all(isinstance(t, list) for t in trees):
        raise ValueError("expected per-stream list leaves from upstream Agent._split")
    return function(*trees)


def make_environments(game, seed, streams):
    import ale_py
    import gymnasium as gym
    from atari import DreamerAtariPreprocessing

    gym.register_envs(ale_py)
    environments, initial = [], []
    for stream in range(streams):
        env = DreamerAtariPreprocessing(gym.make(
            f"ALE/{game.capitalize()}-v5", frameskip=1, repeat_action_probability=.25,
            full_action_space=True), noop_max=0, max_episode_frames=100000, screen_size=64)
        environments.append(env)
        initial.append(observation(env.reset(seed=(seed + stream * 1000003) % 2**32), first=True))
    return environments, initial


class GpuBudget:
    """Sample Vulkan estimated budget, never NVML or physical-free claims."""

    def __init__(self, output):
        import vulkan as vk
        self.vk, self.output = vk, output
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            pApplicationInfo=vk.VkApplicationInfo(pApplicationName="kindle_upstream_comparison",
                                                   apiVersion=vk.VK_MAKE_VERSION(1, 1, 0))), None)
        devices = [d for d in vk.vkEnumeratePhysicalDevices(self.instance)
                   if vk.vkGetPhysicalDeviceProperties(d).deviceID == 0x2c02]
        if len(devices) != 1:
            raise RuntimeError("expected one RTX 5080 Vulkan device")
        self.physical = devices[0]
        props = vk.vkGetPhysicalDeviceProperties(self.physical)
        if props.vendorID != 0x10de or props.deviceName != "NVIDIA GeForce RTX 5080":
            raise RuntimeError(f"unexpected Vulkan GPU: {props.deviceName}")
        self.minimum_headroom = 1 << 64

    def check(self, stage, agent=None):
        vk = self.vk
        budget = vk.VkPhysicalDeviceMemoryBudgetPropertiesEXT()
        props = vk.VkPhysicalDeviceMemoryProperties2(pNext=budget)
        vk.vkGetPhysicalDeviceMemoryProperties2(self.physical, props)
        heaps = [i for i in range(props.memoryProperties.memoryHeapCount)
                 if props.memoryProperties.memoryHeaps[i].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT]
        row = dict(stage=stage, usage_bytes=sum(int(budget.heapUsage[i]) for i in heaps),
                   budget_bytes=sum(int(budget.heapBudget[i]) for i in heaps))
        headroom = row["budget_bytes"] - row["usage_bytes"]
        self.minimum_headroom = min(self.minimum_headroom, headroom)
        if headroom < 2 << 30:
            raise RuntimeError(f"less than 2GiB Vulkan estimated headroom: {row}")
        if agent is not None:
            import jax
            devices = jax.devices()
            if (len(devices) != 1 or devices[0].platform != "gpu" or
                    devices[0].device_kind != "NVIDIA GeForce RTX 5080" or
                    agent.policy_devices != devices or agent.train_devices != devices):
                raise RuntimeError(f"unexpected JAX devices: {devices}")
            row["cuda_allocator"] = devices[0].memory_stats()
            if row["cuda_allocator"]["bytes_limit"] - row["cuda_allocator"]["bytes_in_use"] < 2 << 30:
                raise RuntimeError("less than 2GiB CUDA allocator headroom")
        with self.output.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def close(self):
        self.vk.vkDestroyInstance(self.instance, None)


def train(make_agent, make_replay, make_env, make_stream, make_logger, args, *, game, seed, model_size="12m"):
    import embodied

    if args.replay_context != 1 or args.consec_train != 1:
        raise ValueError("matched comparison requires context1/consec1")
    if min(args.batch_size, args.batch_length, args.train_ratio, args.envs) <= 0:
        raise ValueError("matched recipe dimensions and ratio must be positive")
    batch_steps = args.batch_size * args.batch_length
    capacity = 100000 - args.envs * args.batch_length
    if capacity < batch_steps:
        raise ValueError("replay capacity must hold a complete batch of eligible starts")
    if args.steps <= 0 or args.steps % args.envs:
        raise ValueError("exact full-batch action budget required")
    steps = int(args.steps)
    logdir = Path(str(args.logdir))
    budget = GpuBudget(logdir / "gpu-memory.jsonl")
    environments = []
    output = (logdir / "comparison.jsonl").open("x")

    def emit(row):
        row["unix_time"] = time.time()
        output.write(json.dumps(row, allow_nan=False) + "\n")
        output.flush()

    try:
        budget.check("before_agent")
        construction = time.monotonic()
        agent = make_agent()
        agent.jaxcfg.profiler = False
        initial_parameters = parameter_fingerprints(agent.save()["params"])
        budget.check("constructed", agent)
        construction = time.monotonic() - construction
        environments, initial = make_environments(game, seed, args.envs)
        # Upstream capacity counts eligible sequence starts, Kindle counts
        # stored arrivals. Each stream retains batch_length context/tail rows.
        replay = embodied.replay.Replay(length=args.batch_length + 1, capacity=capacity, seed=seed)
        training = None
        train_carry = agent.init_train(args.batch_size)
        carries = agent.init_policy(args.envs)
        schedule = UpdateSchedule(args.train_ratio, batch_steps)
        actual, updates = 0, 0
        returns, lengths, counts = [0.0] * args.envs, [0] * args.envs, [0] * args.envs
        total_rewards = [0.0] * args.envs
        completed = []
        first_training_action = None
        ids = list(range(args.envs))
        started = time.monotonic()

        def policy_and_store(streams, observations):
            obs = {key: np.stack([o[key] for o in observations]) for key in observations[0]}
            selected_carry = map_carry(lambda rows: [rows[i] for i in streams], carries)
            carry, actions, extras = agent.policy(selected_carry, obs, mode="train")

            def commit(rows, updated):
                if len(updated) != len(streams):
                    raise ValueError("upstream policy returned the wrong stream count")
                for stream, value in zip(streams, updated):
                    rows[stream] = value
                return rows

            map_carry(commit, carries, carry)
            selected = actions["action"].copy()
            selected[obs["is_last"]] = 0
            for row, stream in enumerate(streams):
                replay.add(dict(**observations[row], action=selected[row],
                                **{k: v[row] for k, v in extras.items()}), stream)
            return selected

        selected = policy_and_store(ids, initial)
        budget.check("initialized", agent)
        emit(dict(event="run_start", protocol="replication-matched-actions-v2", game=game,
                  steps=args.steps, num_envs=args.envs, seed=seed,
                  environment_seeds=[(seed+i*1000003) % 2**32 for i in ids],
                  full_action_space=True, sticky_actions=.25, action_repeat=4, noop_max=0,
                  max_episode_frames=100000, observation_size=64, compute_dtype="float32",
                  model_size=model_size, batch_size=args.batch_size, batch_length=args.batch_length,
                  replay_context=1, train_ratio=args.train_ratio,
                  replay_arrival_capacity=100000, replay_sequence_capacity=capacity,
                  agent_construction_seconds=construction, reward_action_aids="none",
                  limits=["pinned upstream agent/math; custom matched collector",
                          "native upstream policy synchronization lag is unchanged",
                          "replay capacity matched; sampling/eviction implementations differ"]))

        def progress(event):
            elapsed = time.monotonic() - started
            return dict(event=event, run_step=actual, learner_step=updates, elapsed_seconds=elapsed,
                        actions_per_second=actual/elapsed, total_rewards=total_rewards.copy(),
                        episode_counts=counts.copy(), partial_returns=returns.copy(), partial_lengths=lengths.copy(),
                        executed_action_frames=[e.executed_action_frames for e in environments],
                        reset_noop_frames=[e.reset_noop_frames for e in environments],
                        emulator_resets=[e.emulator_resets for e in environments],
                        replay_sequences=len(replay), training_debt=schedule.credit)

        for tick in range(1, steps // args.envs + 1):
            results = [env.step(int(action)) for env, action in zip(environments, selected)]
            obs = [observation(result) for result in results]
            emit(dict(event="transition", run_step=actual+args.envs, actions=selected.tolist(),
                      rewards=[float(o["reward"]) for o in obs], terminated=[bool(o["is_terminal"]) for o in obs],
                      truncated=[bool(r[3]) for r in results]))
            selected = policy_and_store(ids, obs)
            actual += args.envs
            for _ in range(schedule.observe(args.envs, len(replay) >= batch_steps)):
                if first_training_action is None:
                    first_training_action = actual
                    # Starting upstream prefetch before warmup could sample its
                    # first batch from only one early sequence, unlike Kindle.
                    training = iter(agent.stream(make_stream(replay, "train")))
                train_carry, outs, metrics = agent.train(train_carry, next(training))
                updates += 1
                if "replay" in outs:
                    replay.update(outs["replay"])
                for name, value in metrics.items():
                    if not np.isfinite(np.asarray(value)).all():
                        raise RuntimeError(f"nonfinite upstream metric: {name}")
                if metrics:
                    emit(dict(event="learner", run_step=actual, learner_step=updates-1,
                              metrics={name: float(np.asarray(value).mean()) for name, value in metrics.items()},
                              lag="upstream returns previous update's metrics"))
            resets = []
            for stream, current in enumerate(obs):
                returns[stream] += float(current["reward"])
                total_rewards[stream] += float(current["reward"])
                lengths[stream] += 1
                if current["is_last"]:
                    row = dict(event="episode", stream=stream, run_step=actual, stream_step=tick,
                               episode=counts[stream], episode_return=returns[stream], episode_length=lengths[stream],
                               terminated=bool(current["is_terminal"]), truncated=bool(results[stream][3]),
                               elapsed_seconds=time.monotonic()-started)
                    emit(row)
                    completed.append(row)
                    counts[stream] += 1
                    returns[stream], lengths[stream] = 0.0, 0
                    resets.append(stream)
            if resets:
                reset_obs = [observation(environments[i].reset(), first=True) for i in resets]
                selected[resets] = policy_and_store(resets, reset_obs)
                emit(dict(event="reset", run_step=actual, streams=resets))
            if tick % 512 == 0:
                budget.check(f"actions{actual}", agent)
                emit(progress("progress"))
                print(f"{actual}/{args.steps} actions; {updates} updates", flush=True)

        state = agent.save()  # Completes outstanding device work before final timing/validation.
        final_parameters = parameter_fingerprints(state["params"])
        if state["counters"]["updates"] != updates or set(final_parameters) != set(initial_parameters):
            raise RuntimeError("invalid final upstream state or update accounting")
        changed = [name for name in final_parameters if final_parameters[name] != initial_parameters[name]]
        if updates and any(not any(name.startswith(prefix) for name in changed) for prefix in ("dyn/", "enc/", "pol/")):
            raise RuntimeError("RSSM, encoder and actor must all change during training")
        np.savez(logdir / "final-state.npz", **state["params"])
        (logdir / "final-counters.json").write_text(json.dumps(state["counters"]) + "\n")
        budget.check("finished", agent)
        final = progress("run_end")
        final.update(reason="budget_complete", learner_updates=updates, first_training_action=first_training_action,
                     completed_episodes=len(completed), changed_parameters=changed,
                     minimum_vulkan_estimated_headroom_bytes=budget.minimum_headroom)
        emit(final)
        (logdir / "comparison-result.json").write_text(json.dumps(final, indent=2) + "\n")
    finally:
        for env in environments:
            env.close()
        budget.close()
        output.close()
