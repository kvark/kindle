"""Emulator oracle for Breakout ball and paddle tracking under sticky actions (CPU only).

Plays the world corpus teacher (FeedbackPixelPolicy plus random command
blocks, block probability .025) with per-frame sticky actions emulated in a
wrapper exactly as ALE applies them, snapshots the emulator at random
origins, and replays the same recorded 32 future actions 32 times with fresh
sticky draws. A replay tracks the recorded continuation at a horizon when its
RAM ball position is within 1.5 px at 96x96 (native dx*96/160, dy*96/210),
the dense evaluation's threshold; balls outside the detector window below the
bricks are not scored, and origins whose ball leaves play within 32 actions
are dropped. The mean over replays is the tracking score of a perfect
simulator scored with one draw; the replay with the most neighbours
approximates the best point forecast. Paddle tracking uses the same
threshold on RAM paddle x.

With stickiness 0 every replay reproduces the recorded continuation, and the
per-frame emulation matches ALE's own sticky actions in distribution.

    python breakout_sticky_oracle.py OUT.json [ORIGINS=256] [STICKY=.25]
"""
import json, sys
import numpy as np
import gymnasium as gym
import ale_py
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atari import DreamerAtariPreprocessing
from kindle._breakout_guidance import FeedbackPixelPolicy
from kindle._representation_probe import positions

P = float(sys.argv[3]) if len(sys.argv) > 3 else .25
K, H, ORIGINS, SEED = 32, 32, int(sys.argv[2]) if len(sys.argv) > 2 else 256, 931001
HORIZONS = (1, 4, 8, 16, 32)

class Sticky(gym.Wrapper):
    def __init__(self, env, rng):
        super().__init__(env); self.rng = rng; self.last = 0
    def reset(self, **kw):
        self.last = 0; return self.env.reset(**kw)
    def step(self, action):
        if self.rng.random() < P:
            action = self.last
        self.last = action
        return self.env.step(action)

gym.register_envs(ale_py)
inner = gym.make('ALE/Breakout-v5', frameskip=1, repeat_action_probability=0.0, full_action_space=True)
sticky = Sticky(inner, np.random.default_rng(SEED + 1))
env = DreamerAtariPreprocessing(sticky, noop_max=0, screen_size=None, max_episode_frames=108000)
ale = inner.unwrapped.ale
policy_rng, origin_rng, replay_rng = (np.random.default_rng(SEED + k) for k in (2, 3, 4))
frame, _ = env.reset(seed=SEED)
teacher, block, command = FeedbackPixelPolicy(), 0, 0

def ram_ball():
    p = positions('Breakout', ale.getRAM())
    return np.array(p[:3], np.float64)  # ball x, ball y, paddle x

def replay(state, last, actions):
    ale.restoreState(state)
    previous, out = last, []
    for action in actions:
        dead = False
        for _ in range(4):
            a = previous if replay_rng.random() < P else action
            previous = a
            ale.act(a)
            if ale.game_over():
                dead = True; break
        out.append(ram_ball() if not dead else np.full(3, np.nan))
        if dead:
            out += [np.full(3, np.nan)] * (len(actions) - len(out)); break
    return np.array(out)

def close(a, b):
    d = np.hypot((a[..., 0] - b[..., 0]) * 96 / 160, (a[..., 1] - b[..., 1]) * 96 / 210)
    return d <= 1.5

pending, results, ticks = [], [], 0
while len(results) < ORIGINS:
    proposed = teacher.propose(frame)
    if block == 0 and policy_rng.random() < .025:
        block, command = int(policy_rng.integers(4, 13)), int(policy_rng.integers(18))
    action = command if block else int(proposed)
    if block:
        block -= 1
    teacher.feedback(action)
    if not pending or ticks - pending[-1]['tick'] >= 8:
        if origin_rng.random() < 1 / 24:
            pending.append(dict(tick=ticks, state=ale.cloneState(), last=sticky.last, origin_y=ram_ball()[1], actions=[], truth=[]))
    frame, reward, terminal, truncated, _ = env.step(action)
    ticks += 1
    for item in pending:
        if len(item['actions']) < H and item['tick'] < ticks:
            item['actions'].append(action)
            item['truth'].append(ram_ball() if not terminal else np.full(3, np.nan))
    done = [item for item in pending if len(item['actions']) == H]
    if terminal:
        pending = [item for item in pending if item in done]
    for item in done:
        pending.remove(item)
        if np.isnan(np.array(item['truth'])[:, :2]).any():
            continue  # lost ball / reset within the horizon: excluded, as most are in the image metric
        now = ale.cloneState()
        replays = np.stack([replay(item['state'], item['last'], item['actions']) for _ in range(K)])
        ale.restoreState(now)
        truth = np.array(item['truth'])
        row = dict(tick=item['tick'])
        yy = np.concatenate([[item['origin_y']], truth[:, 1]])
        for h in HORIZONS:
            t, r = truth[h - 1], replays[:, h - 1]
            if not (92 <= t[1] <= 184 and 10 <= t[0] <= 150):
                continue  # the image metric scores only balls visible below the bricks
            seg = yy[:h + 1]; seg = seg[np.r_[True, np.diff(seg) != 0]]
            flips = [seg[i] for i in range(1, len(seg) - 1) if (seg[i] - seg[i - 1]) * (seg[i + 1] - seg[i]) < 0]
            sample = float(np.nanmean(close(r, t))) if not np.isnan(r).all() else 0.
            neighbours = np.array([close(r, r[i]).sum() for i in range(K)])
            mode = r[int(np.argmax(neighbours))]
            paddle_sample = float(np.mean(np.abs(r[:, 2] - t[2]) * 96 / 160 <= 1.5)) if not np.isnan(t[2]) else None
            row[h] = dict(paddle_sample=paddle_sample, sample=sample, mode=bool(close(mode, t)), spread=float(np.nanstd(r[:, 0]) * 96 / 160),
                          paddle=bool(any(f > 150 for f in flips)), upper=bool(any(f <= 150 for f in flips)))
        results.append(row)
        if len(results) % 32 == 0:
            print(len(results), 'origins at tick', ticks, flush=True)
    if terminal or truncated:
        frame, _ = env.reset(); teacher, block = FeedbackPixelPolicy(), 0
        pending.clear()

def stats(rows):
    return dict(n=len(rows), sample=float(np.mean([r['sample'] for r in rows])) if rows else None,
                mode=float(np.mean([r['mode'] for r in rows])) if rows else None)
summary = {}
for h in HORIZONS:
    rows = [r[h] for r in results if h in r]
    paddles = [r['paddle_sample'] for r in rows if r['paddle_sample'] is not None]
    summary[h] = dict(all=stats(rows), paddle_sample=float(np.mean(paddles)), x_spread_px96=float(np.mean([r['spread'] for r in rows])),
                      paddle_bounce=stats([r for r in rows if r['paddle']]),
                      upper_bounce_only=stats([r for r in rows if r['upper'] and not r['paddle']]),
                      no_vertical_bounce=stats([r for r in rows if not r['upper'] and not r['paddle']]))
print(json.dumps(summary, indent=1))
json.dump(dict(seed=SEED, sticky=P, replays=K, origins=len(results), summary=summary, rows=results),
          open(sys.argv[1], 'w'), indent=0)
