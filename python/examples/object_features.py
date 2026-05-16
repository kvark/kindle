"""Object-level feature extraction from ARC-AGI grid frames.

ARC-AGI-3 frames are H×W integer grids (typically 64×64, values 0-15).
Each connected region of the same non-background color is an object.
This module extracts top-K objects by area and returns a fixed-size
feature vector that can replace or augment kindle's pooled obs token.

Generalization motivation: pixel-pool encoding produces a different
latent for every layout; object-level features carry over across
levels because they're invariant to position when normalized.

Object features (per object):
- color (1)
- center_y, center_x normalized to [0, 1] (2)
- height, width normalized to [0, 1] (2)
- area normalized to [0, 1] (1)
- num_holes (rough complexity proxy) (1)
Total: 7 dims/object × K objects = 7K dims.

Default K=8 → 56 dims, leaving room within OBS_TOKEN_DIM=64 for an
8-dim global-features vector.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from scipy.ndimage import label as _scipy_label, find_objects as _scipy_find_objects


def _connected_components(grid: np.ndarray, ignore_color: int = 0) -> list[dict]:
    """Per-color connected components via scipy.ndimage.label.
    Much faster than the Python BFS (10-100x on 64x64). Returns list
    of dicts with: color, area, bbox (y0, x0, y1, x1), cells (np
    array of (y,x) coordinates)."""
    objects = []
    h, w = grid.shape
    # Iterate distinct non-background colors
    unique = np.unique(grid)
    for color in unique:
        c = int(color)
        if c == ignore_color:
            continue
        mask = (grid == c)
        labeled, n = _scipy_label(mask)
        if n == 0:
            continue
        slices = _scipy_find_objects(labeled)
        for cc_idx in range(n):
            ys, xs = slices[cc_idx]  # bbox slices
            sub_mask = labeled[ys, xs] == (cc_idx + 1)
            area = int(sub_mask.sum())
            if area == 0:
                continue
            y0, y1 = ys.start, ys.stop - 1
            x0, x1 = xs.start, xs.stop - 1
            objects.append({
                "color": c,
                "area": area,
                "bbox": (y0, x0, y1, x1),
                # Cells stored as relative-to-bbox mask for holes;
                # avoid allocating big coord lists.
                "_sub_mask": sub_mask,
            })
    return objects


def _count_holes(obj: dict, grid_shape: tuple[int, int]) -> int:
    """Rough complexity: count enclosed background regions inside bbox.
    Uses the sub_mask stored on the object dict by _connected_components."""
    y0, x0, y1, x1 = obj["bbox"]
    if (y1 - y0) < 2 or (x1 - x0) < 2:
        return 0
    h, w = y1 - y0 + 1, x1 - x0 + 1
    inside = obj.get("_sub_mask")
    if inside is None:
        return 0
    # Flood-fill from bbox edges in the BACKGROUND of inside.
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for y in range(h):
        if not inside[y, 0] and not visited[y, 0]:
            queue.append((y, 0))
            visited[y, 0] = True
        if not inside[y, w - 1] and not visited[y, w - 1]:
            queue.append((y, w - 1))
            visited[y, w - 1] = True
    for x in range(w):
        if not inside[0, x] and not visited[0, x]:
            queue.append((0, x))
            visited[0, x] = True
        if not inside[h - 1, x] and not visited[h - 1, x]:
            queue.append((h - 1, x))
            visited[h - 1, x] = True
    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and not inside[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    # Holes = background cells not reachable from bbox edges.
    holes = 0
    seen_internal = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if inside[y, x] or visited[y, x] or seen_internal[y, x]:
                continue
            # Found a new hole; flood-fill it
            holes += 1
            queue = deque([(y, x)])
            seen_internal[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and not inside[ny, nx] and not seen_internal[ny, nx]:
                        seen_internal[ny, nx] = True
                        queue.append((ny, nx))
    return holes


_FEATURES_PER_OBJECT = 7
_GLOBAL_FEATURES = 8


def object_token(frame: np.ndarray, k: int = 8, max_color: int = 16) -> np.ndarray:
    """Extract a fixed-size object-level token.

    Returns a length-(k*7 + 8) array, default 64. Top-k objects by area
    occupy the first k*7 dims; remaining 8 dims are global stats.

    Args:
        frame: 2D int array, the ARC-AGI grid.
        k: number of objects to keep (top-k by area).
        max_color: color count for normalization.

    Output dims:
        per-object: [color/max, cy/H, cx/W, h/H, w/W, area/(H*W), holes/8]
        global:     [num_objects/k, total_area/(H*W), distinct_colors/max,
                     mean_object_area/(H*W), max_color_freq/(H*W),
                     0, 0, 0]
    """
    h, w = frame.shape
    objs = _connected_components(frame)
    objs.sort(key=lambda o: -o["area"])  # largest first
    objs = objs[:k]

    out = np.zeros(k * _FEATURES_PER_OBJECT + _GLOBAL_FEATURES, dtype=np.float32)
    for i, o in enumerate(objs):
        y0, x0, y1, x1 = o["bbox"]
        cy = (y0 + y1) / 2.0 / max(1.0, h - 1)
        cx = (x0 + x1) / 2.0 / max(1.0, w - 1)
        oh = (y1 - y0 + 1) / float(h)
        ow = (x1 - x0 + 1) / float(w)
        area = o["area"] / float(h * w)
        holes = _count_holes(o, (h, w)) / 8.0
        base = i * _FEATURES_PER_OBJECT
        out[base + 0] = o["color"] / float(max_color)
        out[base + 1] = cy
        out[base + 2] = cx
        out[base + 3] = oh
        out[base + 4] = ow
        out[base + 5] = area
        out[base + 6] = min(holes, 1.0)

    # Globals
    g_off = k * _FEATURES_PER_OBJECT
    out[g_off + 0] = min(len(objs) / float(k), 1.0)
    out[g_off + 1] = sum(o["area"] for o in objs) / float(h * w)
    distinct_colors = len(set(o["color"] for o in objs))
    out[g_off + 2] = distinct_colors / float(max_color)
    if objs:
        out[g_off + 3] = (sum(o["area"] for o in objs) / len(objs)) / float(h * w)
        # max_color_freq: largest color's total area
        from collections import Counter
        c = Counter()
        for o in objs:
            c[o["color"]] += o["area"]
        out[g_off + 4] = max(c.values()) / float(h * w)
    return out


if __name__ == "__main__":
    # Smoke test on synthetic frames.
    rng = np.random.default_rng(42)
    for trial in range(3):
        # Build a grid with a few colored shapes
        frame = np.zeros((64, 64), dtype=np.int32)
        # Two rectangles + one L-shape
        frame[10:20, 10:25] = 3  # red-ish
        frame[30:35, 40:50] = 5  # blue-ish
        frame[40:55, 10:15] = 7  # vertical
        frame[40:42, 10:30] = 7  # extends right (L)
        tok = object_token(frame, k=8)
        n_objs = sum(1 for i in range(8) if tok[i * 7 + 5] > 0)
        print(f"Trial {trial}: token shape={tok.shape}, n_nonempty_obj_slots={n_objs}")
        print(f"  first-object features: color={tok[0]:.3f} cy={tok[1]:.3f} cx={tok[2]:.3f} h={tok[3]:.3f} w={tok[4]:.3f} area={tok[5]:.3f} holes={tok[6]:.3f}")
        print(f"  globals: n/k={tok[8*7]:.3f} total_area={tok[8*7+1]:.3f} distinct={tok[8*7+2]:.3f}")
