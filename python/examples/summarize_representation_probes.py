"""Publish compact, self-contained offline probe results, including every target.

Generated files may be rebuilt mechanically as completed controls arrive. A
complete offline table is still not Phase 2 completion: matched RL is required.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean


PRIMARY = "native/phase15/random/mean"
MODELS = ("pretrained_tiny", "initial_tiny", "large", "reconstruction_cnn", "raw_rgb56")


def aggregate(fits, *, view="all", motion=False):
    values = [metric["r2"] for fit in fits for name, metric in fit["test"][view].items()
              if name.endswith("_velocity") == motion and metric["r2"] is not None]
    return fmean(values) if values else None


def columns(metrics, names):
    def value(name, key):
        number = metrics[name][key]
        return round(number, 6) if number is not None and key != "count" else number
    return {key: [value(name, key) for name in names] for key in ("count", "r2", "mae", "rmse")}


def summarize(inputs):
    records, sources, targets, seen, primary = [], [], {}, set(), []
    for model, path in inputs:
        if model not in MODELS:
            raise ValueError(f"unknown model label: {model}")
        raw = path.read_bytes()
        result = json.loads(raw)
        method = result["method"]
        if (result["status"] != "complete" or result["limit_clips"] is not None or
                method not in ("ridge", "mlp") or result["protocol"] != "kindle-representation-probes-v1"):
            raise ValueError("only complete full-corpus fits may enter this report")
        if method == "mlp" and not result.get("normalization", "").startswith("training-only F64"):
            raise ValueError("superseded MLP F32 normalization; corrected fits required")
        variants = ({"rgb56/single_frame", "rgb56/two_frames"} if model == "raw_rgb56" else
                    {f"{size}/phase{phase}/{projection}/{pooling}" for size in ("native", "rgb64")
                     for phase in (0, 15) for projection in ("random", "pca") for pooling in ("mean", "space_to_depth")})
        expected = {(game, variant) for game in ("Pong", "Breakout", "Seaquest") for variant in variants}
        if (len(result["results"]) != len(expected) or
                {(r["game"], r["variant"]) for r in result["results"]} != expected):
            raise ValueError("incomplete game/variant coverage")
        if (model, method) in seen:
            raise ValueError("duplicate model/probe result")
        seen.add((model, method))
        sources.append(dict(model=model, probe=method, result_sha256=hashlib.sha256(raw).hexdigest(),
                            normalization=result.get("normalization", "training-only F64 ridge statistics"),
                            **{key: result[key] for key in ("protocol", "encoder_sha256", "feature_manifest_sha256", "visibility_sha256")},
                            seconds=result["seconds"]))
        for row in result["results"]:
            game = row["game"]
            names = list(row["fits"][0]["test"]["all"])
            if names != targets.setdefault(game, names):
                raise ValueError("target order differs between controls")
            fits = []
            if ((method == "ridge" and len(row["fits"]) != 1) or
                    (method == "mlp" and [f["fit"]["seed"] for f in row["fits"]] != [1009, 2017, 3019])):
                raise ValueError("incomplete probe head seeds")
            for fit in row["fits"]:
                test = fit["test"]
                info = fit["fit"]
                # Keep selection information and curves, but omit repeated GPU
                # memory snapshots from each fit. They remain in the raw file.
                info = ({k: v for k, v in info.items() if k != "memory"} if isinstance(info, dict) else info)
                fits.append(dict(selection=info, test={view: columns(test[view], names) for view in ("all", "visible")},
                                 by_trajectory={seed: columns(value, names) for seed, value in test["by_trajectory"].items()}))
            records.append(dict(model=model, probe=method, game=game, variant=row["variant"],
                                reuses_fit_from=row.get("reuses_fit_from"), fits=fits,
                                constant={view: columns(row["constant"][view], names) for view in ("all", "visible")}))
            if row["variant"] == ("rgb56/two_frames" if model == "raw_rgb56" else PRIMARY):
                primary.append(dict(model=model, probe=method, game=game,
                                    position_r2=aggregate(row["fits"]), motion_r2=aggregate(row["fits"], motion=True)))
    result = dict(status="partial_offline_comparison", phase2_complete=False,
                  metric_rounding_decimal_places=6,
                  interpretation="Supervised held-out state decoding, not gameplay, future prediction or an architecture decision.",
                  corpus=dict(clips=6144, arrivals_per_clip=16, clips_per_trajectory=256,
                              train_seeds=[6101, 6113, 6121, 6131], validation_seeds=[7103, 7109], test_seeds=[8101, 8111],
                              split_unit="whole trajectory", policy="uniform random", full_actions=True,
                              repeat=4, sticky_probability=.25, reset_noops=0, source_rgb=[210, 160, 3],
                              velocity="backward displacement / executed ALE frames; not a forecast"),
                  limits=["No test-based model selection; regularization/stopping use validation only",
                          "Primary: all valid RAM targets; secondary: independent visible-sprite masks",
                          "Head seeds are fit variability, not independent RL replicates",
                          "Raw RGB56 has 9408/18816 values versus 3136 learned features",
                          "CNN sees 12288 native training frames including Seaquest, without RAM labels",
                          "Tiny video corpus includes Pong/Breakout, not Seaquest; Large capacity and corpus both differ"],
                  targets=targets, sources=sources, primary=primary, results=records)
    required = {(model, method) for model in MODELS for method in ("ridge", "mlp")}
    if required <= seen:
        result["status"] = "offline_comparison_complete_rl_still_required"
    return result


def markdown(result, json_name):
    complete = result["status"].startswith("offline_comparison_complete")
    title = "Offline representation probes" + ("" if complete else " — partial controls")
    lines = [f"# {title}", "",
             "These are held-out supervised state probes, **not RL results or Phase 2 completion**.",
             f"[All per-target R², errors, counts, trajectory splits and selections]({json_name}).", "",
             "The table uses native phase15/JL64/mean features; raw RGB56 uses two frames.",
             "Values are unweighted means over position or motion targets and probe-head seeds.",
             "Undefined constant-target R² is excluded, not turned into zero. All variants remain in JSON.", "",
             "| Model | Probe | Game | Position R² | Motion R² |", "| --- | --- | --- | ---: | ---: |"]
    show = lambda value: "undefined" if value is None else f"{value:.3f}"
    for row in sorted(result["primary"], key=lambda r: (r["probe"], r["game"], MODELS.index(r["model"]))):
        lines.append(f"| {row['model']} | {row['probe']} | {row['game']} | {show(row['position_r2'])} | {show(row['motion_r2'])} |")
    lines += ["", "Whole trajectories are held out: four train, two validation and two test seeds/game.",
              "RAM is an offline target only. Hyperparameters and checkpoints are selected on validation;",
              "test labels never enter fitting. Secondary visibility scores retain the disclosed Seaquest",
              "sprite/mapping limitations; neither they nor test scores select examples or variants.", "",
              "The frozen encoders expose 7×7×64 values. The stateless reconstruction CNN has no history;",
              "identical phase0/15 fits may be reused only after byte equality of all three feature splits.",
              "Raw RGB56 controls have 3×/6× as many values and are not size-matched encoder baselines.", "",
              "Tiny has 250k prior RGB64 frames from Boxing/Pong/Freeway/Breakout/Qbert. Large uses VideoMix.",
              "The reconstruction CNN trains on 12,288 native frames from this corpus's training split,",
              "including Seaquest. These different experiences must not be attributed solely to architecture.", "",
              "Matched three-seed learning curves remain required before deciding the 2D frontend.",
              "See the [comparison protocol](../experiments/2026-09-27-representation-comparison.md).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", required=True, metavar=("MODEL", "RESULT"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([(model, Path(path)) for model, path in args.input])
    path = args.output_prefix.with_suffix(".json")
    # One compact line per row keeps mechanically regenerated result diffs small.
    rows = result.pop("results")
    header = json.dumps(result, indent=2, allow_nan=False)
    payload = header[:-2] + ',\n  "results": [\n' + ",\n".join("    " + json.dumps(row, separators=(",", ":"), allow_nan=False) for row in rows) + "\n  ]\n}\n"
    path.write_text(payload)
    args.output_prefix.with_suffix(".md").write_text(markdown(result, path.name))


if __name__ == "__main__":
    main()
