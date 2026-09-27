"""Read completed screening seeds and print a self-contained JSON summary.

Usage: summarize_screening.py seed1009.jsonl seed2017.jsonl seed3019.jsonl
"""

import argparse
import json

from kindle._screening import PROTOCOL, mean_ci, summarize_curves


def summarize(paths):
    starts, runs = [], []
    for path in paths:
        with open(path) as stream:
            rows = [json.loads(line) for line in stream]
        start, end = rows[0], rows[-1]
        if start["protocol"] != PROTOCOL or start["event"] != "start" or end["event"] != "complete":
            raise ValueError(f"incomplete or unsupported screen: {path}")
        if end["seed"] != start["seed"] or end["actions"] != start["steps"]:
            raise ValueError(f"wrong seed/action budget: {path}")
        config = {k: v for k, v in start["config"].items() if k != "seed"}
        if starts:
            other = {k: v for k, v in starts[0]["config"].items() if k != "seed"}
            for key in ("game", "steps", "num_envs", "observation", "rewards", "action_count",
                        "sticky_action_probability", "difficulty_ramping", "pretraining",
                        "exploration_override", "intrinsic_reward", "native_sha256"):
                if start[key] != starts[0][key]:
                    raise ValueError(f"mismatched {key}: {path}")
            if config != other:
                raise ValueError(f"mismatched learner configuration: {path}")
        starts.append(start)
        curve = []
        for point in end["curve"]:
            metrics = point["metrics"]
            curve.append({k: point[k] for k in ("actions", "seconds", "learner_steps", "score", "completed_episodes")}
                         | dict(world_loss=metrics.get("world", {}).get("total_loss"),
                                raw_kl=metrics.get("world", {}).get("raw_kl"),
                                policy_entropy=metrics.get("behavior", {}).get("policy_entropy")))
        runs.append(dict(seed=start["seed"], environment_seeds=start["environment_seeds"],
                         curve=curve, seconds=end["seconds"], construction_seconds=start["construction_seconds"],
                         updates=end["learner_steps"], unfinished_returns=end["unfinished_returns"],
                         unfinished_lengths=end["unfinished_lengths"], completed_by_stream=end["completed_by_stream"],
                         action_histogram=end["action_histogram"],
                         minimum_sampled_headroom_bytes=min(p["memory"]["budget_bytes"] - p["memory"]["usage_bytes"]
                                                           for p in [start, *end["curve"], end])))
    result = summarize_curves(runs)
    result["early_to_final_score_change"] = mean_ci([
        r["curve"][-1]["score"] - r["curve"][0]["score"] for r in runs])
    result.update(recipe={k: v for k, v in starts[0].items() if k not in
                          ("event", "seed", "environment_seeds", "construction_seconds", "memory")},
                  runs=runs, total_run_seconds=sum(r["seconds"] for r in runs),
                  limits=["online training scores, not frozen competence", "one screening recipe, not a method comparison",
                          "CPU MinAtar and feature upload; GPU learning/inference", "only three learner seeds"])
    result["recipe"]["config"].pop("seed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    print(json.dumps(summarize(args.logs), indent=2, allow_nan=False))
