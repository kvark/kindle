"""Ball-tracking hits of a world forecast split by what the ball does inside the horizon.

Reads a `dense-*.npz` written by `breakout_world_scale.py` (probe or train
evaluations) and classifies each scored origin from the recorded RAM ball
path (native pixels, y grows downward) between the last observed frame and
the scored frame: any vertical reversal below y 150 is a paddle bounce, any
reversal at or above it a brick or ceiling bounce. Classes match
`breakout_sticky_oracle.py`, so the two can be read side by side. Only the
first prior draw is scored, as in the dense evaluations.

    python breakout_world_event_split.py RUN/dense-development.npz [...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kindle._breakout_world import ball_center, geometry  # noqa: E402

HISTORY = 16
CLASSES = ('paddle bounce', 'brick or ceiling bounce only', 'no vertical bounce', 'ball absent')


def vertical_reversals(y):
    y = y[np.r_[True, np.diff(y) != 0]]
    return [y[i] for i in range(1, len(y) - 1) if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0]


def split(path):
    a = dict(np.load(path))
    pixel = geometry(a['truth'].shape[-1])['scale']
    out = {}
    for index, h in enumerate(a['horizons'].tolist()):
        counts = {key: [0, 0] for key in CLASSES}
        for c in np.where(a['valid'][h - 1])[0]:
            target = ball_center(a['truth'][index, c].astype(np.float64) / 255 - .5)
            if target is None:
                continue
            estimate = ball_center(a['prediction'][0, index, c])
            hit = estimate is not None and np.linalg.norm(estimate - target) <= pixel
            path_ = a['position'][HISTORY - 1:HISTORY + h, c, :2]
            if np.isnan(path_).any():
                key = 'ball absent'
            else:
                flips = vertical_reversals(path_[:, 1])
                if any(f > 150 for f in flips):
                    key = 'paddle bounce'
                elif flips:
                    key = 'brick or ceiling bounce only'
                else:
                    key = 'no vertical bounce'
            counts[key][0] += 1
            counts[key][1] += hit
        out[h] = counts
    return out


def main():
    for path in sys.argv[1:]:
        print(path)
        for h, counts in split(path).items():
            total, hits = sum(n for n, _ in counts.values()), sum(k for _, k in counts.values())
            print(f'  {h:2d} steps: {hits}/{total}; ' + '; '.join(f'{key} {k}/{n}' for key, (n, k) in counts.items()))


if __name__ == '__main__':
    main()
