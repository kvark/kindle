"""Small-observation packing and seed-level learning curve summaries."""

import numpy as np

PROTOCOL = "kindle-screening-v1"


def pack_minatar(state):
    """Lossless 2x2 space-to-depth: 10x10xC -> 5x5x4C, zero-pad to 7x7x64.

    There is no resize, colour rendering, averaging, random projection or
    pretrained feature extractor. All public MinAtar channels are retained.
    """
    state = np.asarray(state)
    if state.ndim != 3 or state.shape[:2] != (10, 10) or not 1 <= state.shape[2] <= 16:
        raise ValueError("expected a 10x10xC observation with 1 <= C <= 16")
    if not np.isfinite(state).all():
        raise ValueError("observation must be finite")
    channels = state.shape[2]
    packed = state.reshape(5, 2, 5, 2, channels).transpose(0, 2, 1, 3, 4).reshape(5, 5, 4 * channels)
    output = np.zeros((7, 7, 64), dtype=np.float32)
    output[:5, :5, :4 * channels] = packed
    return output.reshape(-1)


def mean_ci(values, *, seed=0, samples=10000):
    """Percentile bootstrap over independent learner seeds, not episodes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("at least three finite independent seed values are required")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (samples, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                seeds=values.tolist())


def summarize_curves(runs):
    """Aligned action grid plus an interpolated common wall-time interval.

    Scores are each seed's last-50-completed-episode online mean. Three-seed
    bootstrap intervals are coarse; they are not frozen competence estimates.
    """
    if len(runs) < 3 or len({r["seed"] for r in runs}) != len(runs):
        raise ValueError("need at least three distinct learner seeds")
    common = sorted(set.intersection(*[
        {p["actions"] for p in r["curve"] if p["score"] is not None} for r in runs]))
    by_actions = []
    for actions in common:
        rows = [next(p for p in r["curve"] if p["actions"] == actions) for r in runs]
        by_actions.append(dict(actions=actions, score=mean_ci([p["score"] for p in rows]),
                               seconds=mean_ci([p["seconds"] for p in rows])))
    curves = [[p for p in r["curve"] if p["score"] is not None] for r in runs]
    if any(not curve for curve in curves):
        raise ValueError("every seed needs a completed episode")
    first = max(c[0]["seconds"] for c in curves)
    last = min(c[-1]["seconds"] for c in curves)
    if first > last:
        raise ValueError("no common wall-time interval")
    by_time = [dict(seconds=float(t), score=mean_ci([
        np.interp(t, [p["seconds"] for p in c], [p["score"] for p in c]) for c in curves]))
        for t in np.linspace(first, last, 16)]
    return dict(protocol=PROTOCOL, seed_count=len(runs), score_kind="online last-50 completed episode mean",
                bootstrap="10000 percentile resamples of independent learner seeds; RNG seed 0",
                time_alignment="linear interpolation inside common measured wall-time interval, no extrapolation",
                by_actions=by_actions, by_time=by_time)
