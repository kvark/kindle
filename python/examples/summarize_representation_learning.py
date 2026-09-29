"""Compare completed Phase 2 RL runs, retaining every episode and partial tail.

Use --input METHOD LOG for each native/upstream comparison.jsonl. Only groups
with all three learner seeds receive aggregate curves; partial results remain
explicit. This is an online learning comparison, not frozen competence.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import xml.etree.ElementTree as ET

from kindle._screening import mean_ci, summarize_curves


METHODS = ("upstream", "large", "pretrained_tiny", "initial_tiny", "learned_cnn")
SEEDS = (1009, 2017, 3019)
BASELINES = {"Pong": (-20.7, 14.6), "Breakout": (1.7, 30.5), "Seaquest": (68.4, 42054.7)}
REFERENCE = "https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/baselines.yaml"


def reject_constant(value):
    raise ValueError(f"nonfinite JSON: {value}")


def read_run(path):
    """Stream raw transitions to independently reconcile returns and boundaries."""
    episodes, curve, pending = [], [], {}
    header, final, actual, first_update = None, None, 0, None
    previous_seconds = 0.0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            event = row["event"]
            if final is not None:
                raise ValueError("records after run_end")
            if header is None:
                if event != "run_start" or row["num_envs"] <= 0:
                    raise ValueError("missing run_start")
                header = row
                n = row["num_envs"]
                returns, lengths, counts, totals = [0.] * n, [0] * n, [0] * n, [0.] * n
                continue
            if event == "run_start":
                raise ValueError("duplicate run_start")
            if event == "transition":
                if pending or row["run_step"] != actual + n:
                    raise ValueError("missing transition or episode boundary")
                if any(len(row[key]) != n for key in ("actions", "rewards", "terminated", "truncated")):
                    raise ValueError("incomplete transition batch")
                actual += n
                for i, reward in enumerate(row["rewards"]):
                    returns[i] += reward
                    totals[i] += reward
                    lengths[i] += 1
                    if row["terminated"][i] or row["truncated"][i]:
                        pending[i] = dict(stream=i, run_step=actual, episode=counts[i],
                                          episode_return=returns[i], episode_length=lengths[i],
                                          terminated=row["terminated"][i], truncated=row["truncated"][i])
            elif event == "episode":
                expected = pending.pop(row["stream"], None)
                if expected is None or any(row.get(k) != v for k, v in expected.items()):
                    raise ValueError("episode disagrees with transitions")
                episodes.append(expected | dict(elapsed_seconds=row["elapsed_seconds"]))
                i = row["stream"]
                counts[i] += 1
                returns[i], lengths[i] = 0., 0
            elif event == "learner" and first_update is None:
                first_update = row["run_step"]
            elif event in ("progress", "run_end"):
                if (pending or row["run_step"] != actual or row["total_rewards"] != totals or
                        row["episode_counts"] != counts or row["partial_returns"] != returns or
                        row["partial_lengths"] != lengths):
                    raise ValueError("progress disagrees with transitions")
                point = dict(actions=actual, seconds=row["elapsed_seconds"], learner_steps=row["learner_step"],
                             completed_episodes=len(episodes),
                             score=fmean(e["episode_return"] for e in episodes[-50:]) if episodes else None)
                if curve and point["actions"] == curve[-1]["actions"]:
                    curve[-1] = point  # Final checkpoint time belongs to the final point.
                else:
                    curve.append(point)
                if event == "run_end":
                    final = row
            if "elapsed_seconds" in row:
                seconds = row["elapsed_seconds"]
                if seconds < previous_seconds:
                    raise ValueError("elapsed time moved backward")
                previous_seconds = seconds
    if (final is None or final["reason"] != "budget_complete" or actual != header["steps"] or
            final["learner_updates"] != final["learner_step"] or first_update is None):
        raise ValueError("incomplete fresh training run")
    first_update = final.get("first_training_action", first_update)
    if final["learner_updates"] != 1 + (actual - first_update) // 4:
        raise ValueError("updates disagree with actual-action credit")
    return dict(seed=header["seed"], header=header, final=final, source_sha256=digest.hexdigest(),
                first_training_action=first_update, curve=curve, episodes=episodes)


def summarize(inputs, *, budget=200004):
    groups, seen, recipes = defaultdict(list), set(), {}
    for method, path in inputs:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        run = read_run(path)
        h = run["header"]
        game = h.get("game") or h["environment"].removeprefix("ALE/").removesuffix("-v5")
        game = game.capitalize()
        key = (method, game, run["seed"])
        if game not in BASELINES or run["seed"] not in SEEDS or key in seen:
            raise ValueError("unexpected or duplicate game/learner seed")
        seen.add(key)
        common = dict(steps=budget, num_envs=6, full_action_space=True, sticky_actions=.25,
                      action_repeat=4, noop_max=0, max_episode_frames=100000)
        if any(h.get(k) != v for k, v in common.items()):
            raise ValueError("mismatched Phase 2 environment/budget")
        if h["environment_seeds"] != [(run["seed"] + i * 1000003) % 2**32 for i in range(6)]:
            raise ValueError("mismatched environment seeds")
        if method == "upstream":
            if (h["protocol"] != "phase2-matched-actions-v1" or h["reward_action_aids"] != "none" or
                    h["observation_size"] != 64 or h["compute_dtype"] != "float32"):
                raise ValueError("not the matched upstream control")
            config = {k: h[k] for k in ("batch_size", "batch_length", "train_ratio")}
        else:
            if (h["mode"] != "train" or h["starting_environment_step"] or h["starting_learner_step"] or
                    h["restored_checkpoint"] is not None or h.get("exploration") or h["observation_size"] != "native"):
                raise ValueError("not fresh unassisted native training")
            config = {k: v for k, v in h["config"].items() if k != "seed"}
            if method == "learned_cnn":
                if (config.get("observation_kind") != "rgb64" or
                        h.get("model_provenance", {}).get("perception") is not None or
                        not h.get("learned_rgb_preprocessing") or
                        config["loss_scales"]["reconstruction"] != 1 or config["loss_scales"]["future_prediction"] != 0):
                    raise ValueError("not the jointly learned RGB control")
            elif config.get("observation_kind", "features") != "features":
                raise ValueError("RGB control cannot be labelled frozen JEPA")
        if any(config[k] != v for k, v in dict(batch_size=16, batch_length=64, train_ratio=256).items()):
            raise ValueError("mismatched learning schedule")
        if config != recipes.setdefault(method, config):
            raise ValueError("learner configuration changed within a method")
        groups[method, game].append(run)
    results = []
    for (method, game), runs in sorted(groups.items()):
        runs.sort(key=lambda r: r["seed"])
        aggregate = None
        if len(runs) == 3:
            aggregate = summarize_curves(runs)
            aggregate["protocol"] = "phase2-matched-learning-v1"
            random, human = BASELINES[game]
            for point in aggregate["by_actions"] + aggregate["by_time"]:
                point["human_normalized_score"] = mean_ci([(v-random)/(human-random) for v in point["score"]["seeds"]])
            aggregate["final_score"] = mean_ci([r["curve"][-1]["score"] for r in runs])
            aggregate["final_hns"] = mean_ci([(r["curve"][-1]["score"]-random)/(human-random) for r in runs])
            aggregate["run_seconds"] = mean_ci([r["final"]["elapsed_seconds"] for r in runs])
        results.append(dict(method=method, game=game, aggregate=aggregate, runs=runs))
    required = {(method, game, seed) for method in METHODS for game in BASELINES for seed in SEEDS}
    return dict(status="learning_matrix_complete" if required <= seen else "partial_learning_comparison",
                phase2_complete=False, action_budget=budget, recipes=recipes, results=results,
                normalization=dict(formula="(score-random)/(human-random)", anchors=BASELINES, source=REFERENCE),
                limits=["online last-50 completed episode means, not frozen competence",
                        "every episode and unfinished tail is retained; cutoffs are not silently removed",
                        "equal learner-seed weighting, 10000 percentile bootstrap samples; only three seeds",
                        "time starts before initial policy/encoding; construction is reported separately",
                        "time curves interpolate only within common measured support, never extrapolate",
                        "upstream/native RGB reconstruction versus frozen features changes the whole package",
                        "native RGB uses one GPU bilinear resize and a patch CNN/dense decoder, not the exact upstream CNN",
                        "a complete learning matrix still needs offline evidence and an explicit architecture decision"])


def markdown(result, name):
    lines = ["# Phase 2 matched learning comparison", "", f"[All curves, episodes, tails and configurations]({name}).", "",
             "Online training scores, not frozen competence. Final scores use the last 50 completed",
             "episodes per seed (or all if fewer); 95% intervals resample learner seeds, not episodes.", "",
             "Partial groups list individual seeds; no aggregate or uncertainty is reported until all three finish.", "",
             "| Method | Game | Seeds | Final score [95% CI] | Human-normalized |", "| --- | --- | ---: | ---: | ---: |"]
    for row in result["results"]:
        a = row["aggregate"]
        if a:
            score = f"{a['final_score']['mean']:.3f} [{a['final_score']['ci95'][0]:.3f}, {a['final_score']['ci95'][1]:.3f}]"
        else:
            score = "; ".join(f"{r['seed']}: {r['curve'][-1]['score']:.3f}"
                              if r['curve'][-1]['score'] is not None else f"{r['seed']}: no completed episode"
                              for r in row['runs'])
        hns = f"{a['final_hns']['mean']:.4f}" if a else "—"
        lines.append(f"| {row['method']} | {row['game']} | {len(row['runs'])} | {score} | {hns} |")
    lines += ["", f"Human normalization uses [pinned upstream anchors]({REFERENCE}); 1 is the reference human, not mastery.",
              "", *[f"- {limit}." for limit in result["limits"]], ""]
    return "\n".join(lines)


def plot_svg(result):
    """Render audited online curves, using only the summary's measured support."""
    colors = dict(zip(METHODS, ("#0072b2", "#cc79a7", "#009e73", "#e69f00", "#d55e00")))
    labels = dict(zip(METHODS, ("Upstream Dreamer", "Large JEPA", "Pretrained Tiny", "Initial Tiny", "Joint RGB CNN")))
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="1160", height="875", viewBox="0 0 1160 875",
                      role="img", attrib={"aria-labelledby": "title description"})

    def add(tag, *, parent=root, text=None, **attrs):
        element = ET.SubElement(parent, tag, {k.rstrip("_").replace("_", "-"): str(v) for k, v in attrs.items()})
        element.text = text
        return element

    add("title", id="title", text="Phase 2 online learning: scores versus actions and wall time")
    add("desc", id="description", text="Last-50 completed episode means, not frozen competence. "
        "Partial groups show individual learner seeds; complete three-seed groups show means and 95% bootstrap bands. "
        "No extrapolation beyond measured support. Missing scores are not zero.")
    add("rect", width=1160, height=875, fill="white")
    add("style", text="text{font-family:sans-serif;font-size:12px;fill:#222} .heading{font-size:16px;font-weight:bold}")
    add("text", x=25, y=27, class_="heading", text="Phase 2 · online training scores (not frozen competence)")
    add("text", x=25, y=48, text="Last-50 episode mean. Dashed: seed 1009 / 2017 / 3019. Solid + band: three-seed mean and 95% bootstrap CI.")
    for i, method in enumerate(METHODS):
        x = 25 + i * 225
        add("line", x1=x, x2=x+22, y1=71, y2=71, stroke=colors[method], stroke_width=3)
        add("text", x=x+28, y=75, text=labels[method])

    for row_index, game in enumerate(BASELINES):
        groups = [g for g in result["results"] if g["game"] == game]
        panels = []
        for axis in ("actions", "time"):
            key = "actions" if axis == "actions" else "seconds"
            divisor = 1000 if axis == "actions" else 3600
            traces = []
            for group in groups:
                if group["aggregate"] is not None:
                    points = group["aggregate"][f"by_{axis}"]
                    traces.append((group["method"], None, [
                        (p[key]/divisor, p["score"]["mean"], *p["score"]["ci95"]) for p in points]))
                else:
                    for run in group["runs"]:
                        traces.append((group["method"], run["seed"], [
                            (p[key]/divisor, p["score"], p["score"], p["score"])
                            for p in run["curve"] if p["score"] is not None]))
            panels.append(traces)
        all_points = [p for panel in panels for _, _, points in panel for p in points]
        if any(not math.isfinite(v) for point in all_points for v in point):
            raise ValueError("nonfinite learning curve")
        low = min((p[2] for p in all_points), default=0)
        high = max((p[3] for p in all_points), default=1)
        margin = max(1., (high-low)*.06)
        low, high = low-margin, high+margin
        count = sum(len(g["runs"]) for g in groups)
        for column, traces in enumerate(panels):
            left, top, width, height = 70 + column*575, 120 + row_index*250, 460, 170
            xmax = result["action_budget"]/1000 if column == 0 else max(
                (p[0] for _, _, points in traces for p in points), default=1)
            xmax = max(xmax, 1e-9)

            def xy(x, y):
                return f"{left+width*x/xmax:.2f},{top+height*(high-y)/(high-low):.2f}"

            add("text", x=left, y=top-13, class_="heading", text=f"{game} · {count}/15 runs")
            for tick in range(5):
                fraction = tick/4
                x, y = left+width*fraction, top+height*(1-fraction)
                add("line", x1=left, x2=left+width, y1=y, y2=y, stroke="#ddd")
                add("text", x=left-8, y=y+4, text_anchor="end", text=f"{low+(high-low)*fraction:.3g}")
                add("text", x=x, y=top+height+19, text_anchor="middle", text=f"{xmax*fraction:.3g}")
            add("text", x=left+width/2, y=top+height+39, text_anchor="middle",
                text="Actual actions (thousands, six streams combined)" if column == 0 else "Run wall time (hours, construction excluded)")
            for method, seed, points in traces:
                if not points:
                    continue
                trace = add("g", data_method=method, data_seed=seed if seed is not None else "aggregate")
                add("title", parent=trace, text=f"{labels[method]} · " + (f"seed {seed}" if seed is not None else "three-seed mean and 95% CI"))
                if seed is None:
                    band = [xy(p[0], p[2]) for p in points] + [xy(p[0], p[3]) for p in reversed(points)]
                    add("polygon", parent=trace, points=" ".join(band), fill=colors[method], fill_opacity=.12)
                dash = "none" if seed is None else ("8 3", "3 3", "9 3 2 3")[SEEDS.index(seed)]
                add("polyline", parent=trace, points=" ".join(xy(p[0], p[1]) for p in points),
                    fill="none", stroke=colors[method], stroke_width=2, stroke_dasharray=dash)
                if len(points) == 1:
                    x, y = xy(*points[0][:2]).split(",")
                    add("circle", parent=trace, cx=x, cy=y, r=3, fill=colors[method])
            if not any(points for _, _, points in traces):
                add("text", x=left+width/2, y=top+height/2, text_anchor="middle", text="No completed-episode scores yet")
    add("text", x=25, y=864, text="All episodes (including cutoffs) and unfinished tails remain in JSON. Time means use common measured seed support.")
    return ET.tostring(root, encoding="unicode") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", required=True, metavar=("METHOD", "LOG"))
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    result = summarize([(method, Path(path)) for method, path in args.input])
    path = args.output_prefix.with_suffix(".json")
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    svg = args.output_prefix.with_suffix(".svg")
    svg.write_text(plot_svg(result))
    args.output_prefix.with_suffix(".md").write_text(markdown(result, path.name) + f"\n![Online learning curves]({svg.name})\n")


if __name__ == "__main__":
    main()
