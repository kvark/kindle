"""Training-only return boundaries; rewards and episode flags pass through."""

import gymnasium as gym


LIFE_BOUNDARY_KIND = "breakout-raw-frame-life-loss-v1"


class LifeBoundaryMonitor(gym.Wrapper):
    """Wrap raw frame stepping so pooling cannot hide a loss followed by a gain."""

    def __init__(self, env):
        super().__init__(env)
        self.losses = self.gains = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.losses = self.gains = 0
        return result

    def step(self, action):
        before = self.unwrapped.ale.lives()
        result = self.env.step(action)
        change = self.unwrapped.ale.lives() - before
        self.losses += max(-change, 0)
        self.gains += max(change, 0)
        return result

    def take_changes(self):
        changes = self.losses, self.gains
        self.losses = self.gains = 0
        return changes
