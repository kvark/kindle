"""Bounded resident-batch DreamerV3 timing; no gameplay or learning-quality claim.

Use the pinned upstream Python environment. Profiles are captured after ordinary
timing, with explicit device completion so asynchronous dispatch is not speed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from run_upstream_control import REVISION, validate_source
from upstream_matched import GpuBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=("1m", "12m"), default="12m")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.batch_size not in (8, 16, 32):
        parser.error("bounded benchmark supports B8/B16/B32")
    validate_source(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    import elements
    import jax
    import numpy as np
    import ruamel.yaml
    from dreamerv3.agent import Agent
    from embodied.jax import internal

    budget = GpuBudget(args.output / "gpu-memory.jsonl")
    budget.check("before_jax")
    jax.config.update("jax_default_matmul_precision", "highest")
    configs = ruamel.yaml.YAML(typ="safe").load((args.source / "dreamerv3/configs.yaml").read_text())
    config = elements.Config(configs["defaults"]).update(configs[f"size{args.size}"]).update({
        "batch_size": args.batch_size, "batch_length": 16, "report_length": 16,
        "jax.compute_dtype": args.compute_dtype, "logdir": str(args.output), "seed": 701,
    })
    agent_config = elements.Config(**config.agent, logdir=config.logdir, seed=config.seed,
        jax=config.jax, batch_size=config.batch_size, batch_length=config.batch_length,
        replay_context=1, report_length=16, replica=0, replicas=1)
    obs = {"image": elements.Space(np.uint8, (64, 64, 3)),
           "reward": elements.Space(np.float32),
           **{name: elements.Space(bool) for name in ("is_first", "is_last", "is_terminal")}}
    act = {"action": elements.Space(np.int32, (), 0, 18)}
    started = time.perf_counter()
    agent = Agent(obs, act, agent_config)
    agent.jaxcfg.profiler = False
    carry = agent.init_train(args.batch_size)
    jax.block_until_ready((agent.params, carry))
    construction_seconds = time.perf_counter() - started
    budget.check("constructed", agent)
    data = agent._zeros(agent.spaces, (args.batch_size, 17))
    rng = np.random.default_rng(701)
    data["image"] = rng.integers(0, 256, data["image"].shape, dtype=np.uint8)
    data["action"] = rng.integers(0, 18, data["action"].shape, dtype=np.int32)
    data["reward"] = ((np.arange(17)[None, :] + np.arange(args.batch_size)[:, None]) % 3 - 1).astype(np.float32)
    data["is_first"][:, 0] = True
    # Use the public train wrapper and consume its pending policy synchronization
    # as policy() does. No policy forward pass or replay sampling is timed.
    data = internal.device_put(data, agent.train_sharded)
    jax.block_until_ready(data)
    durations = []

    def update(index):
        nonlocal carry
        seed = agent._seeds(index, agent.train_mirrored)
        jax.block_until_ready(seed)
        before = time.perf_counter()
        carry, outs, metrics = agent.train(carry, dict(data, seed=seed))
        if agent.pending_sync:
            old = agent.policy_params
            agent.policy_params = agent.pending_sync
            jax.tree.map(lambda value: value.delete(), old)
            agent.pending_sync = None
        jax.block_until_ready((agent.params, agent.policy_params, carry, outs))
        elapsed_ms = (time.perf_counter() - before) * 1000
        if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
            raise RuntimeError("nonfinite upstream metrics")
        return elapsed_ms

    try:
        for index in range(16 + 256):
            elapsed_ms = update(index)
            if index >= 16:
                durations.append(elapsed_ms)
        budget.check("timed", agent)
        result = dict(status="timed", upstream_revision=REVISION, model_size=args.size,
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            batch_size=args.batch_size, batch_length=16, replay_context=1, imagination_length=15,
            compute_dtype=args.compute_dtype, matmul_precision="highest", jax_version=jax.__version__,
            construction_seconds=construction_seconds, warmup_updates=16, measured_updates=256,
            consumes_policy_sync_each_update=True,
            profile_updates=0, mean_update_ms=statistics.mean(durations),
            median_update_ms=statistics.median(durations), update_ms=durations,
            minimum_sampled_headroom_bytes=budget.minimum_headroom, game_actions=0,
            limits=["RGB DreamerV3, no CDP or exploration ensemble", "resident synthetic batch; no replay sampling",
                    "not identical native weights, RNG or initial recurrent states; throughput, not numerical parity",
                    "profile updates excluded from ordinary timing; not a learning result or SM-utilization metric"])
        def save_result():
            (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        save_result()
        if args.profile:
            try:
                with jax.profiler.trace(str(args.output / "profile"), create_perfetto_trace=True):
                    for index in range(272, 276):
                        update(index)
                        result["profile_updates"] += 1
            except Exception as error:
                result.update(status="profile_failed", profile_error=str(error))
                save_result()
                raise
        budget.check("finished", agent)
        result.update(status="complete", minimum_sampled_headroom_bytes=budget.minimum_headroom)
        save_result()
        print(json.dumps({key: value for key, value in result.items() if key != "update_ms"}), flush=True)
    finally:
        budget.close()


if __name__ == "__main__":
    main()
