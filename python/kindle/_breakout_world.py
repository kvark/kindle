"""Physical evaluation and fixed-window preparation; labels never enter the model."""

import numpy as np


def training_windows(streams, starts, length):
    """A reset at each truncated context; subsequent arrivals keep incoming actions."""
    if len(streams) != len(starts) or length < 2:
        raise ValueError('one start per stream and at least two arrivals required')
    rows = []
    for stream, start in zip(streams, starts):
        stop = start + length
        if start < 0 or stop > len(stream['action']):
            raise ValueError('incomplete training window')
        if np.any(stream['action'][start + 1:stop] == -1):
            raise ValueError('training window crosses a reset')
        row = {key: np.array(stream[key][start:stop], copy=True)
               for key in ('rgb', 'action', 'reward', 'terminal')}
        row['action'][0], row['reward'][0], row['terminal'][0] = -1, 0, False
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
    return dict(count=len(target), mse=float(squared.mean()),
                changed_pixels=int(changed.sum()),
                changed_mse=float(squared.transpose(0, 2, 3, 1)[changed].mean()) if changed.any() else None,
                ball_visible=len(visible), ball_detected=len(detected),
                ball_within_1px=sum(error <= 1 for error in errors)/len(visible) if visible else None,
                ball_error_mean_detected=float(np.mean(errors)) if errors else None,
                ball_error_p95_detected=float(np.percentile(errors, 95)) if errors else None,
                ball_false_detections=sum(t is None and p is not None for t, p in zip(truth, estimated)))


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
