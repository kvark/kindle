"""Read-only dataset integrity and independent sprite-color coverage checks."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from atari import sha256_file
from kindle._representation_probe import CLIP_LENGTH, PROTOCOL, clip_targets, positions


def sprite_boxes(game, ram):
    p = positions(game, ram)
    if game == "Pong":
        return [("ball", p[0], p[1], 2, 4, (236, 236, 236)),
                ("player", 140, p[2], 4, 15, (92, 186, 92)),
                ("enemy", 16, p[3], 4, 15, (213, 130, 74))]
    if game == "Breakout":
        return [("ball", p[0], p[1], 2, 4, (200, 72, 72)),
                ("player", p[2], 189, 16, 4, (200, 72, 72))]
    boxes = [("player", p[0], p[1], 16, 11, (187, 187, 53))]
    for lane in range(4):
        submarine = 3 < int(ram[89+lane]) % 8 < 7
        y = 141 - lane*24 + (0 if submarine else int(ram[93])-4)
        boxes.append((f"enemy_lane{lane}", p[2+lane], y, 8, 11 if submarine else 7,
                      (170, 170, 170) if submarine else (92, 186, 92)))
    return boxes


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["protocol"] == PROTOCOL
    seen, counts = set(), defaultdict(lambda: [0, 0])
    for row in manifest["files"]:
        identity = (row["game"], row["seed"])
        assert identity not in seen
        seen.add(identity)
        assert row["seed"] in manifest["splits"][row["split"]]
        path = root / row["file"]
        assert sha256_file(path) == row["sha256"]
        with np.load(path) as data:
            rgb, ram, frames, targets = (data[k] for k in ("rgb", "ram", "executed_frames", "targets"))
            assert rgb.shape == (manifest["clips_per_seed"], CLIP_LENGTH, 210, 160, 3)
            assert rgb.dtype == np.uint8 and ram.shape == (len(rgb), CLIP_LENGTH, 128)
            assert row["actions"] == len(rgb)*CLIP_LENGTH + row["discarded_tail_actions"]
            for pixels, memory, elapsed, target in zip(rgb, ram, frames, targets):
                np.testing.assert_array_equal(clip_targets(row["game"], memory, elapsed), target)
                for name, x, y, w, h, color in sprite_boxes(row["game"], memory[-1]):
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    # Two-pixel margin covers raster edge conventions and the
                    # max pool's one-frame-old tail; no image-derived label fit.
                    x, y = int(x), int(y)
                    window = pixels[-1, max(y-2, 0):min(y+h+2, 210), max(x-2, 0):min(x+w+2, 160)]
                    hit = bool(np.any(np.all(window == color, axis=-1)))
                    counts[f"{row['game']}/{name}"][0] += hit
                    counts[f"{row['game']}/{name}"][1] += 1
    return dict(protocol=PROTOCOL, files=len(seen), clips=len(seen)*manifest["clips_per_seed"],
                actions=sum(r["actions"] for r in manifest["files"]),
                seconds=manifest["seconds"], hashes_and_targets_verified=True,
                sprite_color_coverage={k: dict(hits=v[0], count=v[1], fraction=v[0]/v[1]) for k,v in counts.items()},
                limits=["color coverage validates coarse spatial alignment, not exact subpixel labels",
                        "occlusion/blinking and conditional missing targets remain visible in target counts",
                        "RAM is privileged evaluation data, never agent input"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2, allow_nan=False))
