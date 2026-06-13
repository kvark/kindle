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


def object_centroids(grid: np.ndarray, max_n: int = 8) -> list[tuple[int, int]]:
    """Top-`max_n` object centroids, largest area first (ties broken by
    bbox position for determinism). Returns [(y, x)] ints. Generic
    connected-component salience — no game-specific knowledge. Used by
    the object-click action slots: the policy picks WHICH object to
    click; this provides WHERE that object is."""
    objs = _connected_components(grid)
    objs.sort(key=lambda o: (-o["area"], o["bbox"][0], o["bbox"][1]))
    out = []
    for o in objs[:max_n]:
        y0, x0, y1, x1 = o["bbox"]
        out.append(((y0 + y1) // 2, (x0 + x1) // 2))
    return out


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
        k_objects: top-k objects by area
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

    objs = _connected_components(frame)
    objs.sort(key=lambda o: -o["area"])
    objs = objs[:k_objects]

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
            distinct = len(set(o["color"] for o in objs))
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


# --- Rule inference over object configurations (2026-06-10) ---
#
# Learns the start->win transformation from the agent's OWN level
# completions, per color, in continuous feature space (no symbols, no
# demos, no game knowledge), and transfers the averaged rule to the
# unsolved frontier level as a predicted win configuration. Consumers:
# potential-based shaping toward the prediction, and rule-guided
# archive-restore ranking.

def object_set(frame: np.ndarray, max_objects: int = 16) -> list[dict]:
    """Normalized per-object records for rule inference. Unlike the
    token encoders this keeps ALL distinct objects (up to max_objects
    by area) so small rule-relevant pieces aren't truncated."""
    h, w = frame.shape
    objs = _connected_components(frame)
    objs.sort(key=lambda o: -o["area"])
    out = []
    for o in objs[:max_objects]:
        y0, x0, y1, x1 = o["bbox"]
        out.append({
            "color": int(o["color"]),
            "cy": (y0 + y1) / 2.0 / max(1.0, h - 1),
            "cx": (x0 + x1) / 2.0 / max(1.0, w - 1),
            "area": o["area"] / float(h * w),
        })
    return out


def _color_stats(objset: list[dict]) -> dict:
    """Per-color aggregate: count, mean position, total area."""
    stats = {}
    for o in objset:
        c = o["color"]
        s = stats.setdefault(c, {"n": 0, "cy": 0.0, "cx": 0.0, "area": 0.0})
        s["n"] += 1
        s["cy"] += o["cy"]
        s["cx"] += o["cx"]
        s["area"] += o["area"]
    for s in stats.values():
        s["cy"] /= s["n"]
        s["cx"] /= s["n"]
    return stats


def extract_color_rules(pairs: list[tuple[list[dict], list[dict]]]) -> dict:
    """Aggregate (start_objset, win_objset) pairs into per-color rules.

    For each color seen at start, collect across pairs:
      - survival: fraction of pairs where the color still exists at win
      - relative move: (win_pos - start_pos) per pair
      - absolute target: win_pos per pair
    The rule uses whichever of relative/absolute clusters more tightly
    across wins (lower positional variance) — a generic disambiguation
    between "move X by delta" and "bring X to place P" rules.
    Returns {color: {survives, mode, dy, dx, ty, tx, d_area}}.
    """
    by_color: dict = {}
    for start, win in pairs:
        ss = _color_stats(start)
        ws = _color_stats(win)
        for c, s in ss.items():
            rec = by_color.setdefault(
                c, {"seen": 0, "alive": 0, "rel": [], "abs": [], "darea": []}
            )
            rec["seen"] += 1
            if c in ws:
                wv = ws[c]
                rec["alive"] += 1
                rec["rel"].append((wv["cy"] - s["cy"], wv["cx"] - s["cx"]))
                rec["abs"].append((wv["cy"], wv["cx"]))
                rec["darea"].append(wv["area"] - s["area"])
    rules = {}
    for c, rec in by_color.items():
        if rec["seen"] == 0:
            continue
        survives = rec["alive"] / rec["seen"] >= 0.5
        rule = {"survives": survives, "mode": "rel",
                "dy": 0.0, "dx": 0.0, "ty": 0.0, "tx": 0.0, "d_area": 0.0}
        if survives and rec["rel"]:
            rel = np.asarray(rec["rel"], dtype=np.float32)
            ab = np.asarray(rec["abs"], dtype=np.float32)
            var_rel = float(rel.var(axis=0).sum())
            var_abs = float(ab.var(axis=0).sum())
            if var_abs < var_rel:
                rule["mode"] = "abs"
                rule["ty"], rule["tx"] = (float(v) for v in ab.mean(axis=0))
            else:
                rule["dy"], rule["dx"] = (float(v) for v in rel.mean(axis=0))
            rule["d_area"] = float(np.mean(rec["darea"]))
        rules[c] = rule
    return rules


def predict_win_stats(start_objset: list[dict], rules: dict) -> dict:
    """Apply per-color rules to a start configuration. Returns the
    predicted per-color stats at the win state ({color: {n, cy, cx,
    area}}). Colors without a rule are predicted unchanged (weakest
    assumption)."""
    pred = {}
    for c, s in _color_stats(start_objset).items():
        rule = rules.get(c)
        if rule is None:
            pred[c] = dict(s)
            continue
        if not rule["survives"]:
            continue  # predicted to disappear
        p = dict(s)
        if rule["mode"] == "abs":
            p["cy"], p["cx"] = rule["ty"], rule["tx"]
        else:
            p["cy"] = min(1.0, max(0.0, s["cy"] + rule["dy"]))
            p["cx"] = min(1.0, max(0.0, s["cx"] + rule["dx"]))
        p["area"] = max(0.0, s["area"] + rule["d_area"])
        pred[c] = p
    return pred


def config_distance(objset: list[dict], pred_stats: dict) -> float:
    """Distance between a current configuration and a predicted win
    configuration, in [0, ~2]. Per predicted color: positional error
    (L2 of mean positions) + presence mismatch; colors present but
    predicted-absent count as mismatch. Mean over the union."""
    cur = _color_stats(objset)
    colors = set(cur) | set(pred_stats)
    if not colors:
        return 0.0
    total = 0.0
    for c in colors:
        s, p = cur.get(c), pred_stats.get(c)
        if s is None or p is None:
            total += 1.0  # presence mismatch
            continue
        dy, dx = s["cy"] - p["cy"], s["cx"] - p["cx"]
        total += min(1.0, (dy * dy + dx * dx) ** 0.5 * 2.0)
        total += min(1.0, abs(s["area"] - p["area"]) * 8.0)
    return total / len(colors)


def extract_relation_rules(
    pairs: list[tuple[list[dict], list[dict]]],
    var_threshold: float = 0.02,
) -> dict:
    """Pairwise-relational win constraints (rule-inference v2).

    For every color pair (c1, c2) present together in win states,
    collect the relative offset between their mean positions across
    wins. Pairs whose win-state offsets cluster tightly (variance
    below threshold) are treated as part of the win condition —
    "c1 ends up at offset (dy, dx) from c2" — regardless of where
    either started. This captures arrangement rules (adjacency,
    alignment, containment-by-position) that per-color motion rules
    (v1) cannot express. Returns {(c1, c2): (dy, dx)} with c1 < c2.
    """
    rels: dict = {}
    for _start, win in pairs:
        ws = _color_stats(win)
        colors = sorted(ws)
        for i, c1 in enumerate(colors):
            for c2 in colors[i + 1:]:
                rels.setdefault((c1, c2), []).append(
                    (ws[c1]["cy"] - ws[c2]["cy"],
                     ws[c1]["cx"] - ws[c2]["cx"])
                )
    rules = {}
    for key, offsets in rels.items():
        if len(offsets) < 2:
            continue
        a = np.asarray(offsets, dtype=np.float32)
        if float(a.var(axis=0).sum()) < var_threshold:
            rules[key] = (float(a[:, 0].mean()), float(a[:, 1].mean()))
    return rules


def relation_distance(objset: list[dict], rel_rules: dict) -> float:
    """Mean violation of the relational win constraints by the current
    configuration, in [0, 1]. Pairs with a missing color contribute a
    full violation (the arrangement cannot be satisfied without both
    parties present). 0.0 when no rules exist."""
    if not rel_rules:
        return 0.0
    cur = _color_stats(objset)
    total = 0.0
    for (c1, c2), (ty, tx) in rel_rules.items():
        s1, s2 = cur.get(c1), cur.get(c2)
        if s1 is None or s2 is None:
            total += 1.0
            continue
        dy = (s1["cy"] - s2["cy"]) - ty
        dx = (s1["cx"] - s2["cx"]) - tx
        total += min(1.0, (dy * dy + dx * dx) ** 0.5 * 2.0)
    return total / len(rel_rules)


def nearest_target_distance(
    objset: list[dict],
    avatar_color: int,
    target_colors: set,
    fallback_rel: dict | None = None,
) -> float | None:
    """Rule v3 waypoint potential: distance from the controllable
    (avatar) object to the NEAREST remaining object of a color the
    learned rule says must disappear. When none remain, falls back to
    the v2 relational target for the avatar's color pair (the exit
    leg). Returns None when the avatar color is absent (no gradient
    available). Distances are normalized grid units; the per-leg form
    gives a monotone approach gradient that color-MEAN distances
    cannot (collecting one of three pickups makes the mean jump).
    """
    avatars = [o for o in objset if o["color"] == avatar_color]
    if not avatars:
        return None
    av = max(avatars, key=lambda o: o["area"])
    targets = [o for o in objset if o["color"] in target_colors]
    if targets:
        best = min(
            ((o["cy"] - av["cy"]) ** 2 + (o["cx"] - av["cx"]) ** 2) ** 0.5
            for o in targets
        )
        # +1 per remaining target keeps the potential monotone across
        # pickups: collecting one strictly lowers phi even if the next
        # target is farther than the previous one was.
        return min(1.0, best * 2.0) + float(len(targets))
    if fallback_rel:
        # Exit leg: satisfy the tightest relational constraint that
        # involves the avatar color.
        cur = _color_stats(objset)
        best = None
        for (c1, c2), (ty, tx) in fallback_rel.items():
            if avatar_color not in (c1, c2):
                continue
            other = c2 if c1 == avatar_color else c1
            if other not in cur:
                continue
            sign = 1.0 if c1 == avatar_color else -1.0
            dy = (av["cy"] - cur[other]["cy"]) - sign * ty
            dx = (av["cx"] - cur[other]["cx"]) - sign * tx
            d = (dy * dy + dx * dx) ** 0.5
            if best is None or d < best:
                best = d
        if best is not None:
            return min(1.0, best * 2.0)
    return 0.0


# Public alias for harness-side per-color aggregation (rule v3).
color_stats = _color_stats


def bfs_target_distance(
    frame: np.ndarray,
    avatar_color: int,
    target_colors: set,
    background_color: int | None = None,
):
    """Rule v3.1: wall-aware PATH distance from the avatar to the
    nearest reachable target, by BFS over passable cells. Straight-line
    distance points through walls in a maze — the gradient is wrong
    wherever a wall sits between avatar and goal; path distance is the
    correct potential. Passable = background (modal) color, target
    colors, or avatar color; everything else is a wall ("you can't
    walk through things" — no game knowledge). +1 per remaining target
    keeps phi monotone across pickups. Returns normalized distance, or
    None when the avatar is absent or no target is reachable (caller
    falls back to the straight-line / relational forms).
    """
    from collections import deque
    a = np.asarray(frame)
    if background_color is None:
        vals, counts = np.unique(a, return_counts=True)
        background_color = int(vals[counts.argmax()])
    av = np.argwhere(a == avatar_color)
    if av.size == 0:
        return None
    tgt_mask = np.isin(a, list(target_colors)) if target_colors else None
    n_targets = int(tgt_mask.sum() and len(np.unique(a[tgt_mask]))) if target_colors else 0
    # passable: background OR target OR avatar
    passable = (a == background_color) | (a == avatar_color)
    if target_colors:
        passable |= np.isin(a, list(target_colors))
    H, W = a.shape
    ay, ax = av.mean(axis=0).astype(int)
    # snap the start to the nearest passable cell (avatar block edges)
    if not passable[ay, ax]:
        cand = av[0]
        ay, ax = int(cand[0]), int(cand[1])
    seen = np.zeros((H, W), dtype=bool)
    q = deque([(ay, ax, 0)])
    seen[ay, ax] = True
    while q:
        y, x, dist = q.popleft()
        if target_colors and tgt_mask[y, x] and not (a[y, x] == avatar_color):
            # reached a target cell
            norm = dist / float(H + W)
            n_left = len(np.unique(a[tgt_mask])) if tgt_mask.any() else 0
            return min(1.0, norm) + float(n_left)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and passable[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx, dist + 1))
    return None
