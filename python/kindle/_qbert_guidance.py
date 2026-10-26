"""Optional whole-episode Qbert guidance from native RGB, with independent lane draws."""

from collections import deque
import hashlib
import math
import random

import numpy as np


GUIDANCE_KIND = "episode-qbert-pixels-v1"
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


class QbertGuidance:
    """Bernoulli choice per episode, not per action; reset only completed lanes."""

    def __init__(self, config, streams, action_count):
        if not isinstance(config, dict) or set(config) != {"kind", "probability", "seed"}:
            raise ValueError("invalid Qbert guidance configuration fields")
        if config["kind"] != GUIDANCE_KIND:
            raise ValueError("unknown Qbert guidance kind")
        probability = config["probability"]
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 < probability <= 1):
            raise ValueError("Qbert guidance probability must be in (0, 1]")
        if type(streams) is not int or streams <= 0 or type(action_count) is not int or action_count != 18:
            raise ValueError("Qbert guidance requires positive streams and the full18 actions")
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**64:
            raise ValueError("Qbert guidance seed must be an unsigned 64-bit integer")
        self._config = dict(config)
        self._rngs = [random.Random(((config["seed"] + lane) % 2**64) ^ GUIDANCE_SEED_XOR)
                      for lane in range(streams)]
        self._selected = [False] * streams
        self._policies = [None] * streams
        self._counts = [0] * streams
        self._episodes = [0] * streams
        self._guided_episodes = [0] * streams
        self.frame_sha256 = [None] * streams
        self.reset(list(range(streams)))

    @property
    def config(self):
        return dict(self._config)

    @property
    def overridden_actions(self):
        return list(self._counts)

    @property
    def episode_counts(self):
        return dict(started_episodes=list(self._episodes), guided_episodes=list(self._guided_episodes))

    def reset(self, streams):
        if (any(type(lane) is not int or not 0 <= lane < len(self._rngs) for lane in streams)
                or len(streams) != len(set(streams))):
            raise ValueError("invalid or repeated Qbert guidance reset stream")
        for lane in streams:
            self._selected[lane] = self._rngs[lane].random() < self._config["probability"]
            self._policies[lane] = PixelPolicy() if self._selected[lane] else None
            self._episodes[lane] += 1
            self._guided_episodes[lane] += self._selected[lane]

    def verify_actions(self, overrides):
        """Check episode selection/counts only; pixel choices need environment replay."""
        if not isinstance(overrides, list) or len(overrides) != len(self._rngs):
            raise ValueError("wrong Qbert guidance action batch")
        for lane, action in enumerate(overrides):
            if self._selected[lane]:
                if type(action) is not int or action not in (0, 2, 3, 4, 5):
                    raise ValueError("invalid Qbert guidance action")
            elif action is not None:
                raise ValueError("override outside selected Qbert episode")
        for lane in range(len(overrides)):
            self._counts[lane] += self._selected[lane]

    def actions(self, frames):
        if len(frames) != len(self._rngs):
            raise ValueError("wrong Qbert guidance frame batch")
        overrides = [policy.action(frame) if policy else None
                     for policy, frame in zip(self._policies, frames)]
        self.frame_sha256 = [hashlib.sha256(frame.tobytes()).hexdigest() if policy else None
                             for policy, frame in zip(self._policies, frames)]
        self.verify_actions(overrides)
        return overrides
