"""Offline evaluation only. RAM labels never enter an agent or its replay."""

import numpy as np


PROTOCOL = "kindle-representation-probes-v1"
GAMES = ("Pong", "Breakout", "Seaquest")
SPLITS = {"train": (6101, 6113, 6121, 6131), "validation": (7103, 7109), "test": (8101, 8111)}
CLIP_LENGTH = 16
LABEL_SOURCE = "https://github.com/k4ntz/OC_Atari/tree/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram"


def target_names(game):
    if game == "Pong":
        positions = ("ball_x", "ball_y", "player_y", "enemy_y")
    elif game == "Breakout":
        positions = ("ball_x", "ball_y", "player_x")
    elif game == "Seaquest":
        positions = ("player_x", "player_y", *(f"enemy_lane{i}_x" for i in range(4)))
    else:
        raise ValueError(f"unsupported probe game: {game}")
    return (*positions, *(name + "_velocity" for name in positions))


def positions(game, ram):
    """Sprite reference coordinates in source pixels, NaN when absent.

    Seaquest uses the leftmost formation slot in each fixed lane, not a nearest
    enemy whose identity can switch. Fixed enemy-lane y coordinates are omitted.
    These maps are checked against image detections separately, not assumed to
    be infallible ground truth. Off-screen values are not training targets.
    """
    ram = np.asarray(ram)
    if ram.shape != (128,) or ram.dtype != np.uint8:
        raise ValueError("RAM must be uint8[128]")
    r = ram.astype(np.int32)
    if game == "Pong":
        ball = (r[49] - 49, r[54] - 14) if r[54] != 0 and r[49] > 49 else (np.nan, np.nan)
        values = (*ball, r[51] - 13 if r[51] > 13 else np.nan,
                  r[50] - 15 if r[50] > 33 else np.nan)
    elif game == "Breakout":
        ball = (r[99] - 49, r[101] + 9) if 0 < r[101] <= 187 else (np.nan, np.nan)
        values = (*ball, r[72] - 47)
    elif game == "Seaquest":
        player = (np.nan, np.nan) if 0 < r[105] < 15 else (r[70], r[97] + 33)
        enemies = [r[30+i] if r[36+i] & 4 and r[30+i] < 160 else np.nan for i in range(4)]
        values = (*player, *enemies)
    else:
        raise ValueError(f"unsupported probe game: {game}")
    values = np.asarray(values, dtype=np.float32)
    for index, name in enumerate(target_names(game)[:len(values)]):
        limit = 160 if name.endswith("_x") else 210
        if not 0 <= values[index] < limit:
            values[index] = np.nan
    return values


def clip_targets(game, ram, executed_frames):
    """Position at the final arrival and backward velocity in pixels/ALE frame.

    The caller supplies one uninterrupted within-episode clip. A missing object
    or a >32px jump across the last decision invalidates only that velocity.
    The jump rule excludes respawn/slot replacement, not a favorable score.
    """
    if len(ram) != CLIP_LENGTH or len(executed_frames) != CLIP_LENGTH:
        raise ValueError("expected a complete 16-arrival clip")
    elapsed = int(executed_frames[-1])
    if elapsed <= 0:
        raise ValueError("final arrival must follow an actual action")
    previous, current = (positions(game, row) for row in ram[-2:])
    displacement = current - previous
    displacement[np.abs(displacement) > 32] = np.nan
    return np.concatenate((current, displacement / elapsed))


def spatial_features(tokens, projection, *, pooling):
    """Project 14x14 patch tokens, then mean-pool or preserve the four corners.

    Both variants expose exactly 7x7x64 values: mean uses 64 projection columns,
    space-to-depth uses 16. Projection must be fitted on training tokens only.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    projection = np.asarray(projection, dtype=np.float32)
    channels = 64 if pooling == "mean" else 16 if pooling == "space_to_depth" else 0
    if tokens.shape[-3:-1] != (14, 14) or projection.shape != (tokens.shape[-1], channels):
        raise ValueError("expected 14x14 tokens and a matching 64/16-channel projection")
    patches = tokens @ projection
    corners = patches.reshape(*patches.shape[:-3], 7, 2, 7, 2, channels)
    if pooling == "mean":
        return corners.mean(axis=(-4, -2)).reshape(*patches.shape[:-3], 7, 7, 64)
    return corners.swapaxes(-4, -3).reshape(*patches.shape[:-3], 7, 7, 64)


def fixed_projection(width, channels=64):
    """The production xorshift64* Rademacher matrix, not NumPy's RNG."""
    state, mask = 0xD130000300000001, (1 << 64) - 1
    values = []
    scale = np.float32(1) / np.sqrt(np.float32(channels))
    for _ in range(width * channels):
        state ^= state >> 12
        state = (state ^ (state << 25)) & mask
        state ^= state >> 27
        values.append(scale if ((state * 0x2545F4914F6CDD1D) & mask) >> 63 else -scale)
    return np.asarray(values, dtype=np.float32).reshape(width, channels)


def fit_pca(tokens, channels=64):
    """Fit once on a declared training-token sample; no labels or held-out data."""
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim != 2 or not np.isfinite(tokens).all() or min(tokens.shape) < channels:
        raise ValueError("insufficient or invalid training tokens for PCA")
    mean = tokens.mean(axis=0)
    centered = tokens - mean
    covariance = centered.T @ centered / (len(tokens) - 1)
    eigenvalues, axes = np.linalg.eigh(covariance)
    axes = axes[:, -channels:][:, ::-1]
    # Canonical signs stabilize artifacts; repeated eigenvalues still permit
    # equivalent bases. Store actual fitted axes, never refit on test data.
    signs = np.sign(axes[np.argmax(np.abs(axes), axis=0), np.arange(channels)])
    axes *= signs
    return mean.astype(np.float32), axes.astype(np.float32), eigenvalues[::-1]


def regression_metrics(prediction, target):
    """Per-target metrics; missing/constant targets do not manufacture R²."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError("expected equally shaped finite predictions and masked targets")
    result = []
    for index in range(target.shape[1]):
        valid = np.isfinite(target[:, index])
        y, p = target[valid, index], prediction[valid, index]
        if not len(y):
            result.append(dict(count=0, r2=None, mae=None, rmse=None))
            continue
        squared = np.square(p.astype(np.float64)-y)
        variation = float(np.square(y.astype(np.float64)-y.mean()).sum())
        result.append(dict(count=int(len(y)), r2=1-float(squared.sum())/variation if variation > 1e-12 else None,
                           mae=float(np.abs(p-y).mean()), rmse=float(np.sqrt(squared.mean()))))
    return result


def ridge_probe(train_x, train_y, validation_x, validation_y, test_x,
                alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Independent dual ridge, per-target missingness and validation-only alpha.

    Reuse eigendecompositions for targets with identical training masks. Do not
    refit on validation data after selection: linear and MLP use the same data.
    Test labels are deliberately absent from this API.
    """
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("ridge penalties must be finite and positive")
    xs = [np.asarray(x, dtype=np.float64) for x in (train_x, validation_x, test_x)]
    if any(x.ndim != 2 or x.shape[1] != xs[0].shape[1] or not np.isfinite(x).all() for x in xs):
        raise ValueError("incompatible/nonfinite probe features")
    if train_y.shape != (len(xs[0]), validation_y.shape[1]) or len(validation_y) != len(xs[1]):
        raise ValueError("incompatible probe targets")
    predictions = np.zeros((len(test_x), train_y.shape[1]), dtype=np.float64)
    cache, selections = {}, []
    for target in range(train_y.shape[1]):
        valid = np.isfinite(train_y[:, target])
        val_valid = np.isfinite(validation_y[:, target])
        if valid.sum() < 2 or val_valid.sum() < 2:
            raise ValueError(f"insufficient train/validation labels for target {target}")
        key = valid.tobytes()
        if key not in cache:
            x = xs[0][valid]
            mean, scale = x.mean(0), x.std(0)
            scale = np.where(scale > 1e-6, scale, 1.0) * np.sqrt(x.shape[1])
            x = (x-mean)/scale
            gram = x @ x.T
            values, vectors = np.linalg.eigh(gram)
            cache[key] = (np.maximum(values, 0), vectors,
                          ((xs[1]-mean)/scale) @ x.T, ((xs[2]-mean)/scale) @ x.T)
        eigenvalues, eigenvectors, val_kernel, test_kernel = cache[key]
        y = train_y[valid, target].astype(np.float64)
        mean_y = y.mean()
        basis = eigenvectors.T @ (y-mean_y)
        coefficients = eigenvectors @ (basis[:, None] / (eigenvalues[:, None] + np.asarray(alphas)[None]))
        val_predictions = val_kernel @ coefficients + mean_y
        errors = np.square(val_predictions[val_valid] - validation_y[val_valid, target, None]).mean(0)
        selected = int(np.argmin(errors))
        predictions[:, target] = test_kernel @ coefficients[:, selected] + mean_y
        selections.append(dict(alpha=alphas[selected], train_count=int(valid.sum()),
                               validation_count=int(val_valid.sum()), validation_mse=errors.tolist()))
    return predictions, selections
