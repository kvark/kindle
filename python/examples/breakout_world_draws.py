"""Held-out forecasts of one world checkpoint with all four prior draws.

Per horizon: hits per draw, any-of-four and all-four hits, hits of the mean
image and undetected balls per draw. A sampler's draws differ where the
future is uncertain (the sticky-action oracle's any-of-four ceiling is
.955 / .829 / .612 at 8 / 16 / 32 steps against one-draw .903 / .700 /
.406); a hedger's agree. Reuses the driver's evaluate_group/window_streams.

    python breakout_world_draws.py CHECKPOINT CORPUS OUTPUT
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breakout_world_scale import window_streams, evaluate_group
from breakout_world_corpus import DRAWS, HISTORY
from kindle._breakout_world import ball_center, geometry

checkpoint, heldout, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=False)
import kindle  # noqa: E402
from breakout_world_model import require_device  # noqa: E402

learner = kindle.WorldModelLearner.restore(str(checkpoint))
require_device(learner)
streams, locations, origins = window_streams(heldout, 182009, 128, HISTORY, 32)
evaluate_group(learner, streams, locations, output, 'dense-development', DRAWS)
a = dict(np.load(output / 'dense-development.npz'))
pixel = geometry(a['truth'].shape[-1])['scale']
rows = {}
for index, h in enumerate(a['horizons'].tolist()):
    valid = a['valid'][h - 1]
    truth = a['truth'][index, valid].astype(np.float64) / 255 - .5
    preds = a['prediction'][:, index, valid]
    centers = [ball_center(x) for x in truth]
    visible = [i for i, c in enumerate(centers) if c is not None]
    hit = np.zeros((len(DRAWS), len(visible)), bool)
    missing = np.zeros_like(hit)
    for d in range(len(DRAWS)):
        for j, i in enumerate(visible):
            e = ball_center(preds[d, i])
            missing[d, j] = e is None
            hit[d, j] = e is not None and np.linalg.norm(e - centers[i]) <= pixel
    mean_image = preds.mean(axis=0)
    mean_hit = 0
    for j, i in enumerate(visible):
        e = ball_center(mean_image[i])
        mean_hit += e is not None and np.linalg.norm(e - centers[i]) <= pixel
    rows[h] = dict(visible=len(visible), per_draw=hit.sum(axis=1).tolist(), any=int(hit.any(axis=0).sum()),
                   all=int(hit.all(axis=0).sum()), mean_image=int(mean_hit),
                   missing_per_draw=missing.sum(axis=1).tolist())
    print(h, rows[h], flush=True)
(output / 'draws.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), draws=list(DRAWS), horizons=rows), indent=1))
