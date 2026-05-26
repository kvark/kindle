"""Procedural cartoon-grid generator for kindle CNN pretraining.

Mimics ARC-AGI-3 visual style: 64x64 grids of integer cells in 0..15.
Scenes contain a handful of geometric primitives (rects, circles, lines,
crosses, L-shapes) on a background, with random colors and positions.
The goal is not realism but EXPOSURE to the kind of visual structure
the target encoder will see: discrete colored regions, contours,
small distinct objects against a uniform background.

Output: numpy .npz with key 'frames' of shape (N, 64, 64) uint8.
"""
from __future__ import annotations

import argparse
import numpy as np


def _draw_rect(grid: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = grid.shape
    rh = rng.integers(2, 16)
    rw = rng.integers(2, 16)
    ry = rng.integers(0, max(1, h - rh))
    rx = rng.integers(0, max(1, w - rw))
    grid[ry:ry+rh, rx:rx+rw] = color


def _draw_circle(grid: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = grid.shape
    r = rng.integers(2, 10)
    cy = rng.integers(r, h - r)
    cx = rng.integers(r, w - r)
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    grid[mask] = color


def _draw_line(grid: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = grid.shape
    horiz = bool(rng.integers(0, 2))
    if horiz:
        y = rng.integers(0, h)
        x0 = rng.integers(0, w - 1)
        x1 = rng.integers(x0 + 1, w + 1)
        grid[y, x0:x1] = color
    else:
        x = rng.integers(0, w)
        y0 = rng.integers(0, h - 1)
        y1 = rng.integers(y0 + 1, h + 1)
        grid[y0:y1, x] = color


def _draw_cross(grid: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = grid.shape
    r = rng.integers(2, 8)
    cy = rng.integers(r, h - r)
    cx = rng.integers(r, w - r)
    grid[cy-r:cy+r+1, cx] = color
    grid[cy, cx-r:cx+r+1] = color


def _draw_lshape(grid: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = grid.shape
    arm = rng.integers(3, 10)
    cy = rng.integers(0, h - arm)
    cx = rng.integers(0, w - arm)
    grid[cy:cy+arm, cx] = color
    grid[cy+arm-1, cx:cx+arm] = color


DRAW_FNS = [_draw_rect, _draw_circle, _draw_line, _draw_cross, _draw_lshape]


def generate_grid(rng: np.random.Generator) -> np.ndarray:
    """One 64x64 cartoon grid with random shapes on a uniform bg."""
    h = w = 64
    bg = int(rng.integers(0, 16))
    grid = np.full((h, w), bg, dtype=np.uint8)
    n_shapes = int(rng.integers(2, 9))
    used = {bg}
    for _ in range(n_shapes):
        # pick a color != bg with high probability
        c = int(rng.integers(0, 16))
        if c == bg and rng.random() < 0.8:
            c = (c + 1 + int(rng.integers(0, 15))) % 16
        used.add(c)
        fn = DRAW_FNS[int(rng.integers(0, len(DRAW_FNS)))]
        fn(grid, c, rng)
    return grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--out", type=str, default="/tmp/aff_runs/pretrain_grids.npz")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    frames = np.empty((a.n, 64, 64), dtype=np.uint8)
    for i in range(a.n):
        frames[i] = generate_grid(rng)
        if i % 5000 == 0:
            print(f"{i}/{a.n}")
    np.savez_compressed(a.out, frames=frames)
    print(f"wrote {a.n} frames to {a.out}")


if __name__ == "__main__":
    main()
