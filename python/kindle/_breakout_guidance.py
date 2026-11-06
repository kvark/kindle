"""Optional capped Breakout guidance from native RGB and previous chosen actions.

The fixed response coefficients were calibrated from the retained9001 CPU
reference trace. See docs/results/2026-10-26-cdp-breakout-pixel-reference.md.
This is a reference policy, not a learned Kindle model or a RAM-based teacher.
"""

import hashlib
import itertools
import numpy as np

from ._pixel_guidance import EpisodePixelGuidance

GUIDANCE_KIND = "bounded-episode-breakout-pixels-v1"
CORRECTIVE_GUIDANCE_KIND = "bounded-episode-breakout-corrective-pixels-v1"
BLOCK_CORRECTIVE_GUIDANCE_KIND = "bounded-block-breakout-corrective-pixels-v1"
INITIAL_CORRECTIVE_GUIDANCE_KIND = "bounded-initial-episodes-breakout-corrective-pixels-v1"
GUIDANCE_KINDS = (GUIDANCE_KIND, CORRECTIVE_GUIDANCE_KIND, BLOCK_CORRECTIVE_GUIDANCE_KIND,
                  INITIAL_CORRECTIVE_GUIDANCE_KIND)
GUIDANCE_SEED_XOR = 0x4252_4541_4B50_5801
ACTION_IDS = {"NOOP": 0, "FIRE": 1, "RIGHT": 3, "LEFT": 4}
MINIMAL_ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")
MINIMAL_TO_FULL = (0, 1, 3, 4)
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

    def __init__(self, *, edge_reset=False):
        super().__init__()
        self.edge_reset = edge_reset
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
            if self.edge_reset:
                stationary_edge = ((position <= 16) | (position >= 147)) & (np.abs(velocity) < .01)
                previous = np.where(stationary_edge, 0, previous)
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

class FeedbackPixelPolicy:
    """Use the executed horizontal command in the calibrated response model."""

    def __init__(self):
        self.teacher = PixelPolicy(edge_reset=True)
        self.pending = None
        self.rejected = 0

    def propose(self, frame):
        if self.pending is not None:
            raise ValueError('Previous proposal needs action feedback')
        self.pending = self.teacher.action(frame)
        return self.pending

    def feedback(self, action):
        if self.pending is None:
            raise ValueError('No pending proposal')
        if type(action) is not int or not 0 <= action < 18:
            raise ValueError('Executed action must be a full18 integer')
        self.teacher.previous_command = (1 if action in (3, 6, 8, 11, 14, 16) else
                                         -1 if action in (4, 7, 9, 12, 15, 17) else 0)
        self.rejected += action != self.pending
        self.pending = None


class _MinimalPolicy:
    """Translate the adapter vocabulary around the unchanged full18 teacher."""

    def __init__(self, factory):
        self.policy = factory()

    def action(self, frame):
        return MINIMAL_TO_FULL.index(self.policy.action(frame))

    def propose(self, frame):
        return MINIMAL_TO_FULL.index(self.policy.propose(frame))

    @property
    def pending(self):
        return None if self.policy.pending is None else MINIMAL_TO_FULL.index(self.policy.pending)

    def feedback(self, action):
        self.policy.feedback(MINIMAL_TO_FULL[action])


class BreakoutGuidance(EpisodePixelGuidance):
    """Capped episode or block selection with the existing pixel teacher."""

    def __init__(self, config, streams, action_count, *, action_meanings=None):
        kind = config.get("kind") if isinstance(config, dict) else None
        self.corrective_labels = kind in (CORRECTIVE_GUIDANCE_KIND, BLOCK_CORRECTIVE_GUIDANCE_KIND,
                                         INITIAL_CORRECTIVE_GUIDANCE_KIND)
        self._block_length = None
        self._initial_episodes = None
        self._ever_guided = None
        schedule = config
        if kind == BLOCK_CORRECTIVE_GUIDANCE_KIND:
            self._block_length = config.get("block_length")
            if type(self._block_length) is not int or not 0 < self._block_length < 2**32:
                raise ValueError("Breakout guidance block_length must be a positive u32")
            schedule = {key: value for key, value in config.items() if key != "block_length"}
        if kind == INITIAL_CORRECTIVE_GUIDANCE_KIND:
            self._initial_episodes = config.get("initial_episodes")
            if type(self._initial_episodes) is not int or not 0 < self._initial_episodes < 2**32:
                raise ValueError("Breakout guidance initial_episodes must be a positive u32")
            schedule = {key: value for key, value in config.items() if key != "initial_episodes"}
        self.expert_labels = []
        self._action_count = action_count
        factory = FeedbackPixelPolicy if self.corrective_labels else PixelPolicy
        minimal = action_meanings is not None
        if minimal and (not isinstance(action_meanings, (list, tuple)) or action_count != 4
                        or tuple(action_meanings) != MINIMAL_ACTION_MEANINGS):
            raise ValueError("Breakout guidance requires the declared NOOP/FIRE/RIGHT/LEFT vocabulary")
        super().__init__(schedule, streams, action_count, game="Breakout", kinds=GUIDANCE_KINDS,
                         bounded_kind=kind,
                         seed_xor=GUIDANCE_SEED_XOR,
                         policy_factory=(lambda: _MinimalPolicy(factory)) if minimal else factory,
                         valid_actions=(0, 1, 2, 3) if minimal else (0, 1, 3, 4),
                         **(dict(expected_action_count=4) if minimal else {}))
        self.expert_labels = [None] * len(self._policies)
        self._config = dict(config)

    @property
    def selected_episodes(self):
        return list(self._ever_guided) if self._block_length else super().selected_episodes

    @property
    def episode_counts(self):
        counts = super().episode_counts
        if self._block_length:
            counts.update(started_blocks=list(self._blocks), guided_blocks=list(self._guided_blocks))
        return counts

    def reset(self, streams):
        super().reset(streams)
        if self._initial_episodes:
            for lane in streams:
                if self._episodes[lane] <= self._initial_episodes:
                    # Preserve the ordinary per-episode draw, then force the initial selection.
                    self._guided_episodes[lane] += not self._selected[lane]
                    self._selected[lane] = True
        if self._block_length:
            if self._ever_guided is None:
                self._ever_guided = [False] * len(self._rngs)
                self._block_indices = [0] * len(self._rngs)
                self._blocks = [0] * len(self._rngs)
                self._guided_blocks = [0] * len(self._rngs)
            for lane in streams:
                self._ever_guided[lane] = self._selected[lane]
                self._block_indices[lane] = 0
                self._blocks[lane] += 1
                self._guided_blocks[lane] += self._selected[lane]
        if self.corrective_labels:
            for lane in streams:
                if self._policies[lane] is None:
                    self._policies[lane] = self._policy_factory()
                if self.expert_labels:
                    self.expert_labels[lane] = None

    def _prepare_blocks(self):
        if not self._block_length:
            return
        for lane, age in enumerate(self._episode_actions):
            if age >= self._max_actions:
                continue
            block = age // self._block_length
            if block != self._block_indices[lane]:
                self._selected[lane] = self._rngs[lane].random() < self._config["probability"]
                self._block_indices[lane] = block
                self._blocks[lane] += 1
                self._guided_blocks[lane] += self._selected[lane]
                if self._selected[lane] and not self._ever_guided[lane]:
                    self._ever_guided[lane] = True
                    self._guided_episodes[lane] += 1

    def verify_actions(self, overrides):
        # Draw only when the next action is requested, never after a terminal action.
        self._prepare_blocks()
        super().verify_actions(overrides)

    def actions(self, frames):
        if not self.corrective_labels:
            return super().actions(frames)
        if len(frames) != len(self._policies):
            raise ValueError('wrong Breakout guidance frame batch')
        if any(policy.pending is not None for policy in self._policies):
            raise ValueError('Previous proposals need action feedback')
        if any(frame.shape != (210, 160, 3) or frame.dtype != np.uint8 for frame in frames):
            raise ValueError('Native RGB8 required')
        self._prepare_blocks()
        self.expert_labels = [policy.propose(frame) for policy, frame in zip(self._policies, frames)]
        self.frame_sha256 = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        overrides = [label if active else None for label, active in zip(self.expert_labels, self.active_guidance)]
        self.verify_actions(overrides)
        return overrides

    def feedback(self, actions):
        if not self.corrective_labels:
            raise ValueError('Action feedback requires corrective guidance')
        if (not isinstance(actions, list) or len(actions) != len(self._policies)
                or any(type(action) is not int or not 0 <= action < self._action_count for action in actions)
                or any(policy.pending is None for policy in self._policies)):
            raise ValueError('Invalid Breakout action feedback batch')
        for policy, action in zip(self._policies, actions):
            policy.feedback(action)
