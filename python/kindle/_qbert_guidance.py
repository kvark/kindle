"""Optional Qbert pixel guidance, selected per episode and optionally action-limited."""

from collections import deque

import numpy as np

from ._pixel_guidance import EpisodePixelGuidance


GUIDANCE_KIND = "episode-qbert-pixels-v1"
BOUNDED_GUIDANCE_KIND = "bounded-episode-qbert-pixels-v1"
GUIDANCE_KINDS = (GUIDANCE_KIND, BOUNDED_GUIDANCE_KIND)
GUIDANCE_SEED_XOR = 0x5142_4552_5450_5801
ROW_X = ((77.5,), (65.5, 93.5), (53.5, 77.5, 105.5),
         (41.5, 65.5, 93.5, 117.5), (29.5, 53.5, 77.5, 105.5, 129.5),
         (17.5, 41.5, 65.5, 93.5, 117.5, 141.5))
ROW_Y = (36, 64, 93, 122, 151, 180)
TILES = tuple((r, c) for r, xs in enumerate(ROW_X) for c in range(len(xs)))
MOVES = ((3, 1, 1), (5, 1, 0), (4, -1, -1), (2, -1, 0))


def components(mask):
    ys, xs = np.where(mask)
    remaining = set(zip(xs.tolist(), ys.tolist()))
    result = []
    while remaining:
        point = remaining.pop()
        pending, points = [point], [point]
        while pending:
            x, y = pending.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    point = (x + dx, y + dy)
                    if point in remaining:
                        remaining.remove(point)
                        pending.append(point)
                        points.append(point)
        xs, ys = zip(*points)
        result.append(dict(count=len(points), box=(min(xs), min(ys), max(xs), max(ys))))
    return result


class PixelPolicy:
    """First-pyramid rule; its only input is the current max-pooled RGB image."""

    def __init__(self):
        self.lit = {tile: False for tile in TILES}
        self.previous = self.target = None
        self.stable = self.age = 0

    def action(self, frame):
        if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
            raise ValueError("Qbert guidance requires native210x160 RGB8 frames")
        for tile in TILES:
            r, c = tile
            x, y = round(ROW_X[r][c]), ROW_Y[r]
            patch = frame[y-2:y+3, x-10:x+10]
            yellow = int(np.all(patch == [210, 210, 64], axis=-1).sum())
            blue = int(np.all(patch == [45, 87, 176], axis=-1).sum())
            if yellow >= 8:
                self.lit[tile] = True
            elif blue >= 8:
                self.lit[tile] = False
        mask = np.all(frame == [181, 83, 40], axis=-1)
        mask[:16] = False
        candidates = [v for v in components(mask) if v['count'] >= 6
                      and 5 <= v['box'][3] - v['box'][1] + 1 <= 28
                      and 2 <= v['box'][2] - v['box'][0] + 1 <= 16]
        sprite = max(candidates, key=lambda v: v['count']) if candidates else None
        tile = None
        if sprite:
            x0, y0, x1, y1 = sprite['box']
            x = (x0 + x1) / 2
            near = [(abs(x - ROW_X[r][c]) + abs(y1 - ROW_Y[r]), (r, c))
                    for r, c in TILES if abs(x - ROW_X[r][c]) <= 5 and abs(y1 - ROW_Y[r]) <= 4]
            if near:
                tile = min(near)[1]
        self.stable = self.stable + 1 if tile is not None and tile == self.previous else 1 if tile is not None else 0
        self.previous = tile
        if self.target is not None:
            self.age += 1
            if self.age >= 6 and self.stable >= 2 and tile == self.target:
                self.target = None
            elif self.age >= 16 and self.stable >= 2 and tile is not None:
                self.target = None
            elif self.age >= 32:
                self.target = None
            return 0
        if tile is None or self.stable < 2:
            return 0
        pending, visited = deque([(tile, [])]), {tile}
        while pending:
            here, route = pending.popleft()
            if here != tile and not self.lit[here]:
                action, self.target = route[0]
                self.age = 0
                return action
            r, c = here
            for action, dr, dc in MOVES:
                nxt = (r + dr, c + dc)
                if nxt in self.lit and nxt not in visited:
                    visited.add(nxt)
                    pending.append((nxt, route + [(action, nxt)]))
        return 0


class QbertGuidance(EpisodePixelGuidance):
    """Qbert episode selection with the original pixel rule and random schedule."""

    def __init__(self, config, streams, action_count):
        super().__init__(config, streams, action_count, game="Qbert", kinds=GUIDANCE_KINDS,
                         bounded_kind=BOUNDED_GUIDANCE_KIND, seed_xor=GUIDANCE_SEED_XOR,
                         policy_factory=PixelPolicy, valid_actions=(0, 2, 3, 4, 5))
