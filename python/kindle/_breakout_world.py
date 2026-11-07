"""Physical evaluation and fixed-window preparation; labels never enter the model."""

import numpy as np


def training_windows(streams, starts, length, *, allow_resets=False, retain_initial_targets=False):
    """A reset at each truncated context; subsequent arrivals keep incoming actions."""
    if len(streams) != len(starts) or length < 2:
        raise ValueError('one start per stream and at least two arrivals required')
    rows = []
    for stream, start in zip(streams, starts):
        stop = start + length
        if start < 0 or stop > len(stream['action']):
            raise ValueError('incomplete training window')
        if not allow_resets and np.any(stream['action'][start + 1:stop] == -1):
            raise ValueError('training window crosses a reset')
        row = {key: np.array(stream[key][start:stop], copy=True)
               for key in ('rgb', 'action', 'reward', 'terminal')}
        row['action'][0] = -1
        if not retain_initial_targets:
            row['reward'][0], row['terminal'][0] = 0, False
        rows.append(row)
    return {key: np.stack([row[key] for row in rows], axis=1) for key in rows[0]}


def sample_locations(streams, length, count, rng):
    """Uniform over complete windows, including those crossing episode resets."""
    sizes = np.array([max(0, len(stream['action'])-length+1) for stream in streams])
    if length < 2 or count <= 0 or not sizes.sum():
        raise ValueError('no complete windows or invalid sample count')
    ends = sizes.cumsum()
    indices = rng.integers(int(ends[-1]), size=count)
    lanes = np.searchsorted(ends, indices, side='right')
    previous = np.r_[0, ends[:-1]]
    return [(int(lane), int(index-previous[lane])) for lane, index in zip(lanes, indices)]


def evaluation_windows(streams, locations, history=16, horizon=32):
    """Keep terminal arrivals; later resets and unavailable tails are unscored.

    Padding supplies no ground truth. Model commands after an unscored boundary
    are NOOP, and cannot affect an earlier scored causal prediction.
    """
    if not locations or history < 1 or horizon < 1:
        raise ValueError('nonempty locations and positive history/horizon required')
    fields = ('rgb', 'action', 'reward', 'terminal', 'position', 'lives')
    rows = []
    for lane, start in locations:
        stream = streams[lane]
        if start < 0 or start+history >= len(stream['action']):
            raise ValueError('complete observed prefix and one future arrival required')
        length = history+horizon
        available = min(length, len(stream['action'])-start)
        row = {}
        for key in fields:
            values = np.asarray(stream[key][start:start+available])
            padded = np.zeros((length, *values.shape[1:]), values.dtype)
            padded[:available] = values
            row[key] = padded
        row['action'][0], row['reward'][0], row['terminal'][0] = -1, 0, False
        valid = np.arange(history, length) < available
        for index in range(horizon):
            if row['action'][history+index] == -1:
                valid[index:] = False
                break
        row['valid'] = valid
        row['action'][history:][~valid] = 0
        rows.append(row)
    return {key: np.stack([row[key] for row in rows], axis=1) for key in rows[0]}


def first_contact_window(stream, length=32, history=16):
    """Select a fitting clip containing a paddle-height vertical reversal.

    RAM position is used only to choose the fitting example, never as a target
    or model input. This is a contact candidate, not an exact collision oracle.
    """
    y = np.asarray(stream['position'])[:, 1]
    for peak in range(2, len(y) - 2):
        if (np.isfinite(y[peak-2:peak+3]).all() and y[peak] >= 165
                and y[peak] > y[peak-1] and y[peak+1] < y[peak]):
            start = peak - history - 4
            if start >= 0 and start + length <= len(y):
                if not np.any(stream['action'][start+1:start+length] == -1):
                    return start
    raise ValueError('no complete paddle-contact fitting window')


def ball_center(rgb):
    """Conservative red-sprite measurement below the bricks, in resized pixels.

    Absent/ambiguous detections remain missing. No coordinate is manufactured
    from a low-contrast prediction. This excludes upper-wall ball positions.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape != (3, 64, 64) or not np.isfinite(rgb).all():
        raise ValueError('expected finite centered CHW64 RGB')
    image = rgb + .5
    excess = image[0] - np.maximum(image[1], image[2])
    mask = (excess > .12) & (image[0] > .30)
    mask[:28] = False
    mask[56:] = False
    mask[:, :4] = False
    mask[:, 60:] = False
    y, x = np.where(mask)
    if not 1 <= len(x) <= 12:
        return None
    weights = excess[y, x]
    return np.array([np.average(x, weights=weights), np.average(y, weights=weights)])


def image_metrics(target, prediction, context):
    target, prediction, context = (np.asarray(x, np.float64) for x in (target, prediction, context))
    if target.shape != prediction.shape or target.shape != context.shape or target.ndim != 4:
        raise ValueError('equal NCHW images required')
    if target.shape[1:] != (3, 64, 64) or not all(np.isfinite(x).all() for x in (target, prediction, context)):
        raise ValueError('finite CHW64 images required')
    squared = np.square(prediction-target)
    changed = np.max(np.abs(target-context), axis=1) > 1/255
    truth = [ball_center(x) for x in target]
    estimated = [ball_center(x) for x in prediction]
    visible = [i for i, x in enumerate(truth) if x is not None]
    detected = [i for i in visible if estimated[i] is not None]
    errors = [float(np.linalg.norm(estimated[i]-truth[i])) for i in detected]
    paddles = [paddle_center(x) for x in target]
    predicted_paddles = [paddle_center(x) for x in prediction]
    paddle_rows = [i for i, x in enumerate(paddles) if x is not None]
    paddle_errors = [abs(paddles[i]-predicted_paddles[i]) for i in paddle_rows if predicted_paddles[i] is not None]
    return dict(count=len(target), mse=float(squared.mean()) if len(target) else None,
                changed_pixels=int(changed.sum()),
                changed_mse=float(squared.transpose(0, 2, 3, 1)[changed].mean()) if changed.any() else None,
                ball_visible=len(visible), ball_detected=len(detected),
                ball_within_1px=sum(error <= 1 for error in errors)/len(visible) if visible else None,
                ball_error_mean_detected=float(np.mean(errors)) if errors else None,
                ball_error_p95_detected=float(np.percentile(errors, 95)) if errors else None,
                ball_false_detections=sum(t is None and p is not None for t, p in zip(truth, estimated)),
                paddle_visible=len(paddle_rows), paddle_detected=len(paddle_errors),
                paddle_within_1px=sum(x <= 1 for x in paddle_errors)/len(paddle_rows) if paddle_rows else None,
                paddle_mae_detected=float(np.mean(paddle_errors)) if paddle_errors else None,
                bricks=brick_metrics(target, prediction, context))


def paddle_center(rgb):
    if np.shape(rgb) != (3, 64, 64) or not np.isfinite(rgb).all():
        raise ValueError('expected finite centered CHW64 RGB')
    image = np.asarray(rgb, np.float64)+.5
    excess = image[0, 58, 3:61]-np.maximum(image[1, 58, 3:61], image[2, 58, 3:61])
    x = np.flatnonzero((excess > .12) & (image[0, 58, 3:61] > .30))
    if not len(x): return None
    components = sorted(np.split(x, np.flatnonzero(np.diff(x) > 1)+1), key=len, reverse=True)
    if not 2 <= len(components[0]) <= 10 or (len(components) > 1 and len(components[0]) == len(components[1])):
        return None
    x = components[0]
    return float(np.average(x+3, weights=excess[x]))


BRICK_COLORS = np.array([(200, 72, 72), (198, 108, 58), (180, 122, 48),
                        (162, 162, 42), (72, 160, 72), (66, 72, 200)], np.float64)/255


def brick_strength(rgb):
    """Image-only occupancy at the centers of the 6x18 native brick cells."""
    rgb = np.asarray(rgb, np.float64)
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64) or not np.isfinite(rgb).all():
        raise ValueError('expected finite NCHW64 RGB')
    y, x = np.meshgrid((60+6*np.arange(6))*64/210-.5,
                       (12+8*np.arange(18))*64/160-.5, indexing='ij')
    y, x = y.ravel(), x.ravel()
    y0, x0 = y.astype(int), x.astype(int)
    dy, dx = (y-y0)[None, :, None], (x-x0)[None, :, None]
    images = rgb.transpose(0, 2, 3, 1)+.5
    color = (images[:, y0, x0]*(1-dy)*(1-dx)+images[:, y0+1, x0]*dy*(1-dx)
             +images[:, y0, x0+1]*(1-dy)*dx+images[:, y0+1, x0+1]*dy*dx)
    prototype = np.repeat(BRICK_COLORS, 18, axis=0)
    strength = np.sum(color*prototype, axis=-1)/np.sum(prototype*prototype, axis=-1)
    return np.clip(strength.reshape(len(rgb), 6, 18), 0, 1)


def brick_metrics(target, prediction, context):
    truth, estimated, previous = (brick_strength(x) >= .5 for x in (target, prediction, context))
    present, absent = int(truth.sum()), int((~truth).sum())
    false_additions = int((~truth & estimated).sum())
    false_removals = int((truth & ~estimated).sum())
    removed, predicted_removed = previous & ~truth, previous & ~estimated
    matched = int((removed & predicted_removed).sum())
    return dict(cells=int(truth.size), present=present, absent=absent,
        accuracy=float((truth == estimated).mean()) if truth.size else None,
        balanced_error=.5*(false_removals/present+false_additions/absent) if present and absent else None,
        false_additions=false_additions, false_removals=false_removals,
        removed=int(removed.sum()), removed_recall=matched/int(removed.sum()) if removed.any() else None,
        predicted_removed=int(predicted_removed.sum()),
        removed_precision=matched/int(predicted_removed.sum()) if predicted_removed.any() else None)


def event_metrics(target, probability):
    target, probability = np.asarray(target), np.asarray(probability, np.float64)
    if target.dtype != bool or target.shape != probability.shape or target.ndim != 1:
        raise ValueError('boolean targets and equal probability vector required')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('event probabilities must be finite and in [0,1]')
    result = dict(count=len(target), positive=int(target.sum()), negative=int((~target).sum()))
    result['brier'] = float(np.square(probability-target).mean()) if len(target) else None
    result['positive_brier'] = float(np.square(probability[target]-1).mean()) if target.any() else None
    result['negative_brier'] = float(np.square(probability[~target]).mean()) if (~target).any() else None
    result['balanced_brier'] = ((result['positive_brier']+result['negative_brier'])/2
                                if target.any() and (~target).any() else None)
    return result
