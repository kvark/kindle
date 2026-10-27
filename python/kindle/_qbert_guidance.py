"""Optional Qbert pixel guidance, selected per episode and optionally action-limited."""

from collections import Counter, deque

import numpy as np

from ._pixel_guidance import EpisodePixelGuidance


GUIDANCE_KIND = "episode-qbert-pixels-v1"
BOUNDED_GUIDANCE_KIND = "bounded-episode-qbert-pixels-v1"
CONTINUATION_GUIDANCE_KIND = "bounded-episode-qbert-continuation-pixels-v1"
GUIDANCE_KINDS = (GUIDANCE_KIND, BOUNDED_GUIDANCE_KIND, CONTINUATION_GUIDANCE_KIND)
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


BLUE = (45, 87, 176)
YELLOW = (210, 210, 64)
ORANGE = (181, 83, 40)


class _RepeatingPixelPolicy(PixelPolicy):

    def __init__(self):
        super().__init__()
        self.command = 0
        self.origin = None

    def action(self, frame):
        action = super().action(frame)
        if action:
            self.command = action
            self.origin = self.previous
            return action
        if self.command and self.target is not None and (1 <= self.age <= 3) and (self.previous == self.origin) and (self.stable >= 2):
            return self.command
        self.command = 0
        return 0


class _AvoidingPixelPolicy(_RepeatingPixelPolicy):

    def distance(self, start, goals):
        pending = deque([(start, 0)])
        visited = {start}
        while pending:
            tile, cost = pending.popleft()
            if tile in goals:
                return cost
            for _, dr, dc in MOVES:
                nxt = (tile[0] + dr, tile[1] + dc)
                if nxt in self.lit and nxt not in visited:
                    visited.add(nxt)
                    pending.append((nxt, cost + 1))
        return 21

    def action(self, frame):
        action = super().action(frame)
        if not action or self.age != 0 or self.previous is None:
            return action
        mask = np.all(frame == [146, 70, 192], axis=-1)
        mask[:20] = False
        enemies = set()
        for sprite in components(mask):
            if sprite['count'] < 6:
                continue
            x0, y0, x1, y1 = sprite['box']
            x = (x0 + x1) / 2
            distance, tile = min(((abs(x - ROW_X[r][c]) + abs(y1 - ROW_Y[r]), (r, c)) for r, c in TILES))
            if distance <= 30:
                enemies.add(tile)
        if not enemies:
            return action

        def hazard(tile):
            distance = self.distance(tile, enemies)
            return 100 if distance == 0 else 10 if distance == 1 else 1 if distance == 2 else 0
        original = hazard(self.target)
        if original < 10:
            return action
        unlit = {tile for tile, value in self.lit.items() if not value}
        choices = []
        for order, (move, dr, dc) in enumerate(MOVES):
            nxt = (self.previous[0] + dr, self.previous[1] + dc)
            if nxt in self.lit:
                choices.append((hazard(nxt), self.distance(nxt, unlit), order, move, nxt))
        best = min(choices)
        if best[0] < original:
            _, _, _, action, self.target = best
            self.command = action
        return action


class ContinuationPixelPolicy:

    def __init__(self):
        self.policy = _AvoidingPixelPolicy()
        self.base = BLUE
        self.goal = YELLOW
        self.candidate = None
        self.stable = 0
        self.completed_goal = None
        self.flips = 0
        self.goal_history = [YELLOW]

    def board_base(self, frame, goal):
        counts = Counter()
        for r, xs in enumerate(ROW_X):
            for x in xs:
                colors, ns = np.unique(frame[ROW_Y[r] - 2:ROW_Y[r] + 3, round(x) - 10:round(x) + 10].reshape(-1, 3), axis=0, return_counts=True)
                for rgb, n in zip(colors, ns):
                    color = tuple((int(v) for v in rgb))
                    if n >= 8 and color not in [(0, 0, 0), ORANGE, goal]:
                        counts[color] += 1
        return counts.most_common(1)[0][0] if counts else self.base

    def action(self, frame):
        if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
            raise ValueError('Native RGB8 required')
        colors, counts = np.unique(frame[4:14, 30:72].reshape(-1, 3), axis=0, return_counts=True)
        candidates = [(int(n), tuple((int(v) for v in rgb))) for rgb, n in zip(colors, counts) if np.any(rgb) and n >= 8]
        if candidates:
            color = max(candidates)[1]
            self.stable = self.stable + 1 if color == self.candidate else 1
            self.candidate = color
            if self.stable >= 8 and color != self.completed_goal:
                self.completed_goal = None
                if color != self.goal:
                    self.goal = color
                    self.base = self.board_base(frame, color)
                    self.goal_history.append(color)
                    self.policy = _AvoidingPixelPolicy()
        else:
            self.candidate = None
            self.stable = 0
        observed = frame.copy()
        observed[np.all(frame == BLUE, axis=-1) | np.all(frame == YELLOW, axis=-1)] = [0, 0, 0]
        observed[np.all(frame == self.base, axis=-1)] = BLUE
        observed[np.all(frame == self.goal, axis=-1)] = YELLOW
        action = self.policy.action(observed)
        if all(self.policy.lit.values()):
            self.flips += 1
            self.completed_goal = self.goal
            self.base, self.goal = (self.goal, self.base)
            self.goal_history.append(self.goal)
            self.policy = _AvoidingPixelPolicy()
            self.candidate = None
            self.stable = 0
            return 0
        return action


class QbertGuidance(EpisodePixelGuidance):
    """Qbert episode selection with the original pixel rule and random schedule."""

    def __init__(self, config, streams, action_count):
        continuation = isinstance(config, dict) and config.get("kind") == CONTINUATION_GUIDANCE_KIND
        super().__init__(config, streams, action_count, game="Qbert", kinds=GUIDANCE_KINDS,
                         bounded_kind=CONTINUATION_GUIDANCE_KIND if continuation else BOUNDED_GUIDANCE_KIND,
                         seed_xor=GUIDANCE_SEED_XOR,
                         policy_factory=ContinuationPixelPolicy if continuation else PixelPolicy,
                         valid_actions=(0, 2, 3, 4, 5))
