"""Shared bounded episode selection and accounting for RGB guidance rules."""

import hashlib
import math
import random


class EpisodePixelGuidance:
    """Bernoulli choice per episode, not per action; reset only completed lanes."""

    def __init__(self, config, streams, action_count, *, game, kinds, bounded_kind,
                 seed_xor, policy_factory, valid_actions):
        self._game = game
        self._policy_factory = policy_factory
        self._valid_actions = valid_actions
        if not isinstance(config, dict):
            raise ValueError(f"invalid {self._game} guidance configuration fields")
        if config.get("kind") not in kinds:
            raise ValueError(f"unknown {self._game} guidance kind")
        fields = {"kind", "probability", "seed"}
        self._max_actions = None
        if config["kind"] == bounded_kind:
            fields.add("max_actions")
            self._max_actions = config.get("max_actions")
            if type(self._max_actions) is not int or not 0 < self._max_actions < 2**32:
                raise ValueError(f"{self._game} guidance max_actions must be a positive u32")
        if set(config) != fields:
            raise ValueError(f"invalid {self._game} guidance configuration fields")
        probability = config["probability"]
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 < probability <= 1):
            raise ValueError(f"{self._game} guidance probability must be in (0, 1]")
        if type(streams) is not int or streams <= 0 or type(action_count) is not int or action_count != 18:
            raise ValueError(f"{self._game} guidance requires positive streams and the full18 actions")
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**64:
            raise ValueError(f"{self._game} guidance seed must be an unsigned 64-bit integer")
        self._config = dict(config)
        self._rngs = [random.Random(((config["seed"] + lane) % 2**64) ^ seed_xor)
                      for lane in range(streams)]
        self._selected = [False] * streams
        self._policies = [None] * streams
        self._counts = [0] * streams
        self._episodes = [0] * streams
        self._guided_episodes = [0] * streams
        self._episode_actions = [0] * streams
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

    @property
    def selected_episodes(self):
        return list(self._selected)

    @property
    def active_guidance(self):
        return [selected and (self._max_actions is None or actions < self._max_actions)
                for selected, actions in zip(self._selected, self._episode_actions)]

    def reset(self, streams):
        if (any(type(lane) is not int or not 0 <= lane < len(self._rngs) for lane in streams)
                or len(streams) != len(set(streams))):
            raise ValueError(f"invalid or repeated {self._game} guidance reset stream")
        for lane in streams:
            self._selected[lane] = self._rngs[lane].random() < self._config["probability"]
            self._policies[lane] = self._policy_factory() if self._selected[lane] else None
            self._episodes[lane] += 1
            self._guided_episodes[lane] += self._selected[lane]
            self._episode_actions[lane] = 0

    def verify_actions(self, overrides):
        """Check episode selection/counts only; pixel choices need environment replay."""
        if not isinstance(overrides, list) or len(overrides) != len(self._rngs):
            raise ValueError(f"wrong {self._game} guidance action batch")
        active = self.active_guidance
        for lane, action in enumerate(overrides):
            if active[lane]:
                if type(action) is not int or action not in self._valid_actions:
                    raise ValueError(f"invalid {self._game} guidance action")
            elif action is not None:
                raise ValueError(f"override outside active {self._game} guidance")
        for lane in range(len(overrides)):
            self._counts[lane] += active[lane]
            self._episode_actions[lane] += 1

    def actions(self, frames):
        if len(frames) != len(self._rngs):
            raise ValueError(f"wrong {self._game} guidance frame batch")
        active = self.active_guidance
        overrides = [policy.action(frame) if enabled else None
                     for policy, frame, enabled in zip(self._policies, frames, active)]
        self.frame_sha256 = [hashlib.sha256(frame.tobytes()).hexdigest() if enabled else None
                             for frame, enabled in zip(frames, active)]
        self.verify_actions(overrides)
        return overrides
