"""Optional capped Breakout guidance from native RGB and previous chosen actions.

The fixed response coefficients were calibrated from the retained9001 CPU
reference trace. See docs/results/2026-10-26-cdp-breakout-pixel-reference.md.
This is a reference policy, not a learned Kindle model or a RAM-based teacher.
"""

import itertools
import numpy as np

from ._pixel_guidance import EpisodePixelGuidance

GUIDANCE_KIND = "bounded-episode-breakout-pixels-v1"
GUIDANCE_KINDS = (GUIDANCE_KIND,)
GUIDANCE_SEED_XOR = 0x4252_4541_4B50_5801
ACTION_IDS = {"NOOP": 0, "FIRE": 1, "RIGHT": 3, "LEFT": 4}
RESPONSE_WEIGHTS = np.array([-0.20973869463450853, 0.1227875991027573,
                             8.681938841928321, 11.996340450736303])
ACTION_PLANS = np.array(list(itertools.product((0, -1, 1), repeat=5)), dtype=float)


class _TrackingPolicy:

    def __init__(self):
        self.previous = None
        self.previous_paddle = None
        self.horizontal_speed = None
        self.last_debug = {}

    def action(self, frame):
        if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
            raise ValueError("Breakout guidance requires native210x160 RGB8 frames")
        rgb = frame.astype(np.int16)
        red = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 120) & (rgb[:, :, 2] < 120) & (rgb[:, :, 0] > 1.5 * np.maximum(rgb[:, :, 1], rgb[:, :, 2]))
        paddle = red[185:198, 8:152]
        row = int(np.argmax(paddle.sum(axis=1)))
        xs = np.flatnonzero(paddle[row])
        ball = np.argwhere(red[90:185, 8:152])
        if len(xs) < 5:
            self.previous = None
            self.previous_paddle = None
            self.horizontal_speed = None
            self.last_debug = dict(ball_seen=bool(len(ball)), paddle_seen=False)
            return ACTION_IDS['FIRE']
        px = float(xs.mean() + 8)
        py = float(row + 185)
        vp = 0 if self.previous_paddle is None else px - self.previous_paddle
        self.previous_paddle = px
        if not 0 < len(ball) <= 32:
            self.previous = None
            self.horizontal_speed = None
            self.last_debug = dict(ball_seen=False, paddle_seen=True, paddle_x=px, paddle_velocity=vp)
            return ACTION_IDS['FIRE']
        bx = float(ball[:, 1].mean() + 8)
        by = float(ball[:, 0].mean() + 90)
        target = bx
        dx = dy = None
        projected = False
        side_correction = False
        if self.previous is not None:
            dx = bx - self.previous[0]
            dy = by - self.previous[1]
            if 0 < dy <= 16 and abs(dx) <= 16:
                if self.horizontal_speed is not None and abs(dx) < 0.75 * self.horizontal_speed and (bx < 18 or bx > 142):
                    dx = self.horizontal_speed if bx < 18 else -self.horizontal_speed
                    side_correction = True
                distance = max(0, py - 2 - by)
                raw = bx + dx * distance / dy
                phase = (raw - 10) % (2 * 139)
                target = 10 + (phase if phase <= 139 else 2 * 139 - phase)
                half = float(xs[-1] - xs[0] + 1) / 2
                target = max(8 + half, min(152 - half, target))
                projected = True
            if abs(dx) >= 3 and abs(dx) <= 16:
                self.horizontal_speed = abs(dx)
        self.previous = (bx, by)
        stop_x = px + 0.75 * vp
        delta = target - stop_x
        direction = 'RIGHT' if delta > 4 else 'LEFT' if delta < -4 else 'NOOP'
        self.last_debug = dict(ball_seen=True, paddle_seen=True, ball_x=bx, ball_y=by, paddle_x=px, paddle_velocity=vp, stop_x=stop_x, target_x=target, dx=dx, dy=dy, projected=projected, side_correction=side_correction)
        return ACTION_IDS[direction]


class PixelPolicy(_TrackingPolicy):

    def __init__(self):
        super().__init__()
        self.weights = RESPONSE_WEIGHTS
        self.plans = ACTION_PLANS
        self.previous_command = 0
        self.valid_landing = None

    def action(self, frame):
        fallback = super().action(frame)
        d = self.last_debug
        if not d.get('ball_seen') or not d.get('paddle_seen'):
            self.previous_command = 0
            self.valid_landing = None
            return fallback
        target = d['target_x']
        dy = d['dy']
        by = d['ball_y']
        if d['projected'] and by <= 180:
            self.valid_landing = (target, dy)
        if d['projected'] and by > 180 and (self.valid_landing is not None):
            target, dy = self.valid_landing
        steps = (187 - by) / dy if d['projected'] and dy and (dy > 0) else 20
        steps = max(0.1, steps)
        position = np.full(len(self.plans), d['paddle_x'])
        velocity = np.full(len(self.plans), d['paddle_velocity'])
        previous = np.full(len(self.plans), self.previous_command)
        predictions = [position.copy()]
        for t in range(5):
            velocity = self.weights[0] + self.weights[1] * velocity + self.weights[2] * self.plans[:, t] + self.weights[3] * previous
            next_position = np.clip(position + velocity, 15.5, 147.5)
            velocity = next_position - position
            position = next_position
            previous = self.plans[:, t]
            predictions.append(position.copy())
        if steps <= 5:
            lower = int(steps)
            fraction = steps - lower
            impact = predictions[lower] if lower == 5 else predictions[lower] * (1 - fraction) + predictions[lower + 1] * fraction
            error = (impact - target) ** 2
        else:
            error = (position - target) ** 2 + 0.15 * velocity ** 2
        error += 0.02 * np.sum(self.plans ** 2, axis=1)
        best = int(np.argmin(error))
        command = int(self.plans[best, 0])
        self.previous_command = command
        d.update(response_target_x=target, collision_steps=steps, selected_plan=self.plans[best].tolist(), predicted_positions=[float(v[best]) for v in predictions], response_cost=float(error[best]))
        return ACTION_IDS['RIGHT' if command > 0 else 'LEFT' if command < 0 else 'NOOP']

class BreakoutGuidance(EpisodePixelGuidance):
    """Capped per-episode guidance; full18 action indices, with no RAM inputs."""

    def __init__(self, config, streams, action_count):
        super().__init__(config, streams, action_count, game="Breakout", kinds=GUIDANCE_KINDS,
                         bounded_kind=GUIDANCE_KIND, seed_xor=GUIDANCE_SEED_XOR,
                         policy_factory=PixelPolicy, valid_actions=(0, 1, 3, 4))
