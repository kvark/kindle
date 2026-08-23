"""Object-level feature extraction from ARC-AGI grid frames.

ARC-AGI-3 frames are H×W integer grids (typically 64×64, values 0-15).
Each connected region of the same non-background color is an object.
This module ranks nested components before their larger containers, then
returns a fixed-size feature vector that can replace or augment kindle's
pooled obs token.

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

from collections import deque

import numpy as np


def _connected_components(
    grid: np.ndarray, ignore_color: int | None = None,
) -> list[dict]:
    """Return the 4-connected regions of each non-background color.

    ARC frames are only 64x64, so a local flood fill keeps this optional
    example dependency-free without putting SciPy on the runtime path.
    Colors and components are visited in deterministic raster order. By
    default, the modal color is treated as background; interactive ARC
    games do not consistently use color zero for their canvas.
    """
    objects = []
    h, w = grid.shape
    if ignore_color is None:
        colors, counts = np.unique(grid, return_counts=True)
        ignore_color = int(colors[np.argmax(counts)])
    for color in np.unique(grid):
        c = int(color)
        if c == ignore_color:
            continue
        mask = (grid == c)
        visited = np.zeros((h, w), dtype=bool)
        for start_y in range(h):
            for start_x in range(w):
                if not mask[start_y, start_x] or visited[start_y, start_x]:
                    continue

                queue = deque([(start_y, start_x)])
                visited[start_y, start_x] = True
                cells = []
                y0 = y1 = start_y
                x0 = x1 = start_x
                while queue:
                    y, x = queue.popleft()
                    cells.append((y, x))
                    y0, y1 = min(y0, y), max(y1, y)
                    x0, x1 = min(x0, x), max(x1, x)
                    for ny, nx in (
                        (y - 1, x), (y + 1, x),
                        (y, x - 1), (y, x + 1),
                    ):
                        if (0 <= ny < h and 0 <= nx < w
                                and mask[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                sub_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
                for y, x in cells:
                    sub_mask[y - y0, x - x0] = True
                objects.append({
                    "color": c,
                    "area": len(cells),
                    "bbox": (y0, x0, y1, x1),
                    "_sub_mask": sub_mask,
                })
    return objects


def _containment_depth(obj: dict, objects: list[dict]) -> int:
    """Number of distinct component boxes that geometrically contain `obj`."""
    y0, x0, y1, x1 = obj["bbox"]
    return sum(
        1
        for outer in objects
        if outer is not obj
        and outer["bbox"] != obj["bbox"]
        and outer["bbox"][0] <= y0
        and outer["bbox"][1] <= x0
        and outer["bbox"][2] >= y1
        and outer["bbox"][3] >= x1
    )


def _rank_objects(objects: list[dict]) -> list[dict]:
    """Rank likely controls/pieces ahead of enclosing panels.

    Interactive grids commonly draw a large board component with smaller
    pieces inside its bounding box. Pure area ranking exhausts the fixed
    object budget on backgrounds, panels, and decorations. Bounding-box
    containment is a cheap, game-independent hierarchy signal.
    """
    return sorted(
        objects,
        key=lambda o: (
            -_containment_depth(o, objects),
            -o["area"],
            o["bbox"][0],
            o["bbox"][1],
        ),
    )


def _representative_cell(obj: dict) -> tuple[int, int]:
    """Return the component cell closest to its arithmetic center."""
    y0, x0, _y1, _x1 = obj["bbox"]
    cells = np.argwhere(obj["_sub_mask"])
    center = cells.mean(axis=0)
    # A bounding-box or arithmetic center can fall outside an L/ring.
    # Argmin's raster-order tie break keeps this deterministic.
    cell = cells[np.square(cells - center).sum(axis=1).argmin()]
    return y0 + int(cell[0]), x0 + int(cell[1])


def object_candidates(
    grid: np.ndarray,
    max_n: int = 8,
    max_color: int = 16,
) -> list[tuple[int, int, np.ndarray]]:
    """Ranked object representatives with features for pointer policies.

    Each row is ``(y, x, features)``. The 12 normalized features describe
    candidate identity and geometry, followed by its relative rank and the
    current candidate-set size. A variable-set policy can score these rows
    directly instead of allocating one permanent policy logit per object.
    """
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty 2D array")
    if max_n < 1:
        raise ValueError("max_n must be positive")
    if max_color < 1:
        raise ValueError("max_color must be positive")

    height, width = grid.shape
    all_objects = _connected_components(grid)
    ranked = _rank_objects(all_objects)[:max_n]
    count = len(ranked)
    candidates = []
    for rank, obj in enumerate(ranked):
        y0, x0, y1, x1 = obj["bbox"]
        y, x = _representative_cell(obj)
        features = np.asarray([
            obj["color"] / float(max_color),
            y / max(1.0, height - 1),
            x / max(1.0, width - 1),
            (y0 + y1) * 0.5 / max(1.0, height - 1),
            (x0 + x1) * 0.5 / max(1.0, width - 1),
            (y1 - y0 + 1) / float(height),
            (x1 - x0 + 1) / float(width),
            obj["area"] / float(height * width),
            min(_count_holes(obj, grid.shape) / 8.0, 1.0),
            min(_containment_depth(obj, all_objects) / 8.0, 1.0),
            rank / max(1.0, count - 1),
            min(count / 256.0, 1.0),
        ], dtype=np.float32)
        candidates.append((y, x, features))
    return candidates


def object_centroids(grid: np.ndarray, max_n: int = 8) -> list[tuple[int, int]]:
    """Top-`max_n` salient object representatives as ``(y, x)`` cells.

    Nested objects rank before containers; remaining ties use area and
    bbox position. Every returned coordinate lies on its component.
    """
    objects = _rank_objects(_connected_components(grid))[:max_n]
    return [_representative_cell(obj) for obj in objects]


def project_to_object_centroid(
    grid: np.ndarray,
    x: float,
    y: float,
    max_n: int = 256,
) -> tuple[int, int]:
    """Project an approximate click onto the nearest salient object cell.

    The returned coordinate is ``(x, y)`` for ARC action payloads. Object
    representatives are exactly the ones used by object search, so a learned
    coordinate cannot miss a discrete target merely because of regression
    error. If the frame contains no foreground object, return the rounded,
    bounds-clamped input coordinate.
    """
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty 2D array")
    if max_n < 1:
        raise ValueError("max_n must be positive")

    height, width = grid.shape
    clamped_x = max(0, min(width - 1, round(x)))
    clamped_y = max(0, min(height - 1, round(y)))
    centroids = object_centroids(grid, max_n=max_n)
    if not centroids:
        return clamped_x, clamped_y
    nearest_y, nearest_x = min(
        centroids,
        key=lambda point: (
            (point[1] - x) ** 2 + (point[0] - y) ** 2,
            point[0],
            point[1],
        ),
    )
    return nearest_x, nearest_y


def project_with_object_motion(
    grid: np.ndarray,
    x: float,
    y: float,
    click_history: list[tuple[int, int]],
    max_n: int = 64,
    max_extrapolation_distance: float = 5.0,
) -> tuple[int, int]:
    """Project a click, extrapolating a stable object-space motion first.

    Two identical consecutive displacements are enough to establish the
    motion prior, but extrapolation is used only while the learned coordinate
    remains within ``max_extrapolation_distance`` pixels of it. A deliberate
    large jump therefore breaks a formerly stable sequence. Irregular and
    short histories continue to use the learned coordinate directly. History
    and return values use ARC payload order ``(x, y)``.
    """
    target_x, target_y = x, y
    if len(click_history) >= 3:
        x0, y0 = click_history[-3]
        x1, y1 = click_history[-2]
        x2, y2 = click_history[-1]
        first_delta = (x1 - x0, y1 - y0)
        second_delta = (x2 - x1, y2 - y1)
        if first_delta == second_delta:
            extrapolated_x = x2 + second_delta[0]
            extrapolated_y = y2 + second_delta[1]
            distance_squared = (
                (x - extrapolated_x) ** 2 + (y - extrapolated_y) ** 2
            )
            if distance_squared <= max_extrapolation_distance ** 2:
                target_x = extrapolated_x
                target_y = extrapolated_y
    return project_to_object_centroid(
        grid,
        target_x,
        target_y,
        max_n=max_n,
    )


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


def hybrid_token(frame: np.ndarray, pixel_grid: int = 4, k_objects: int = 6,
                 max_color: int = 16) -> np.ndarray:
    """V2 hybrid encoding: coarse pixel pool + object features.

    Pixel pool preserves spatial precision (still useful for navigation/
    clicking puzzles where exact position matters). Object features add
    layout-invariant structure for cross-level transfer.

    Default layout (fits OBS_TOKEN_DIM=64):
      - 4x4 = 16-dim pixel pool (coarser than 8x8 but keeps spatial)
      - 6 objects × 7 feat = 42 dims
      - 6 globals = 6 dims
      - Total = 64 dims

    Args:
        frame: 2D int grid (typically 64×64)
        pixel_grid: pool side (4 = 4x4 = 16 dims)
        k_objects: number of salient objects to keep
        max_color: color count for normalization
    """
    h, w = frame.shape
    bh, bw = h // pixel_grid, w // pixel_grid
    h_used = bh * pixel_grid
    w_used = bw * pixel_grid
    pixel_pool = (
        frame[:h_used, :w_used].astype(np.float32).reshape(
            pixel_grid, bh, pixel_grid, bw
        ).mean(axis=(1, 3)) / float(max_color)
    ).flatten()
    pdim = pixel_grid * pixel_grid

    objs = _rank_objects(_connected_components(frame))[:k_objects]

    obj_dim = k_objects * _FEATURES_PER_OBJECT
    obj_part = np.zeros(obj_dim, dtype=np.float32)
    for i, o in enumerate(objs):
        y0, x0, y1, x1 = o["bbox"]
        cy = (y0 + y1) / 2.0 / max(1.0, h - 1)
        cx = (x0 + x1) / 2.0 / max(1.0, w - 1)
        oh = (y1 - y0 + 1) / float(h)
        ow = (x1 - x0 + 1) / float(w)
        area = o["area"] / float(h * w)
        holes = _count_holes(o, (h, w)) / 8.0
        base = i * _FEATURES_PER_OBJECT
        obj_part[base + 0] = o["color"] / float(max_color)
        obj_part[base + 1] = cy
        obj_part[base + 2] = cx
        obj_part[base + 3] = oh
        obj_part[base + 4] = ow
        obj_part[base + 5] = area
        obj_part[base + 6] = min(holes, 1.0)

    # 6-dim globals (matches remaining 64 - pdim - obj_dim)
    n_globals = 64 - pdim - obj_dim
    globals_part = np.zeros(n_globals, dtype=np.float32)
    if n_globals > 0:
        globals_part[0] = min(len(objs) / float(k_objects), 1.0)
        if n_globals > 1:
            globals_part[1] = sum(o["area"] for o in objs) / float(h * w)
        if n_globals > 2:
            distinct = len({o["color"] for o in objs})
            globals_part[2] = distinct / float(max_color)
        if n_globals > 3 and objs:
            globals_part[3] = (sum(o["area"] for o in objs) / len(objs)) / float(h * w)
        if n_globals > 4 and objs:
            from collections import Counter
            c = Counter()
            for o in objs:
                c[o["color"]] += o["area"]
            globals_part[4] = max(c.values()) / float(h * w)
        if n_globals > 5:
            globals_part[5] = float(np.unique(frame).size) / float(max_color)

    out = np.concatenate([pixel_pool, obj_part, globals_part]).astype(np.float32)
    assert out.shape[0] == 64, f"hybrid_token size {out.shape[0]} != 64"
    return out


def object_token(frame: np.ndarray, k: int = 8, max_color: int = 16) -> np.ndarray:
    """Extract a fixed-size object-level token.

    Returns a length-(k*7 + 8) array, default 64. Top-ranked salient
    objects occupy the first k*7 dims; remaining 8 dims are global stats.

    Args:
        frame: 2D int array, the ARC-AGI grid.
        k: number of salient objects to keep.
        max_color: color count for normalization.

    Output dims:
        per-object: [color/max, cy/H, cx/W, h/H, w/W, area/(H*W), holes/8]
        global:     [num_objects/k, total_area/(H*W), distinct_colors/max,
                     mean_object_area/(H*W), max_color_freq/(H*W),
                     0, 0, 0]
    """
    h, w = frame.shape
    objs = _rank_objects(_connected_components(frame))[:k]

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
    distinct_colors = len({o["color"] for o in objs})
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
