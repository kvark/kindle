"""Vectorized Atari training harness for kindle (BatchAgent + DQN CNN encoder).

Target: hard focus on training throughput. Uses gymnasium vectorized envs
(SyncVectorEnv by default; AsyncVectorEnv when --async is set) and
measures where time goes: env.step, frame conversion, agent.act,
agent.observe. A baseline for later optimizations (external-buffer
plumbing, pure-GPU framecpy).

Standard Atari preprocessing (defaults, configurable from the CLI):
- frame skip 4 in AtariPreprocessing
- sticky-action probability 0.25 in ALE
- 84×84 grayscale
- framestack=4 → (C=4, H=84, W=84) per lane per step
- uint8 → float32 / 255 at hand-off to kindle

Reward routing: the raw per-step env reward is SIGNED (Pong point won
= +1, point lost = −1) and is fed through kindle's first-class
extrinsic-reward channel (`set_extrinsic_reward`, scaled by
`extrinsic_reward_alpha`, default --extrinsic-alpha 1.0). The old
default routed |reward| through the homeostatic channel, whose
`-max(0, |value−target|−tol)` form is symmetric — a point won and a
point lost both scored −0.5, training the agent to avoid scoring
events of either sign. That legacy path is kept only behind
--legacy-homeo-reward for A/B comparison.

Usage:
    python python/examples/atari_batch.py --preset extrinsic_grpo \
        --game Breakout --lanes 64 --steps 1875
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import numpy as np

# --- Ablation blocks ------------------------------------------------------
#
# Mirror arc_agi3_batch.py's design: named blocks compose via --preset /
# --enable / --disable. Explicit --flag overrides block defaults.

BLOCKS: dict[str, dict[str, object]] = {
    "rnd": {
        "rnd_alpha": 1.0,
    },
    "m8": {
        "delta_goal_alpha": 0.3,
        "delta_goal_threshold": 0.0,
        "delta_goal_merge_radius": 0.1,
        "delta_goal_surprise_threshold": 0.5,
    },
    "xeps": {
        "xeps_alpha": 0.3,
        "xeps_grid_resolution": 0.05,
    },
    "planner": {
        "planner_horizon": 4,
        "planner_samples": 32,
        "planner_every": 20,
    },
    "extrinsic_grpo": {
        "use_adam": True,
        "adam_eps": 1e-4,
        "grad_clip_norm": 0.5,
        "grad_clip_every": 10,
        "n_step": 64,
        "gamma": 0.99,
        "rollout_length": 8,
        "advantage_normalize": True,
        "use_grpo": True,
        "use_sil": True,
        "sil_loss_coef": 0.5,
        "value_loss_coef": 0.0,
        "entropy_floor_frac": 0.8,
        "recon_loss_coef": 1.0,
        "recon_visual_target": True,
    },
}

PRESETS: dict[str, list[str]] = {
    "none": [],
    "rnd_only": ["rnd"],
    "extrinsic_grpo": ["extrinsic_grpo"],
    "everything": ["rnd", "m8", "xeps", "planner"],
}


def _apply_blocks(args: argparse.Namespace, raw_argv: list[str]) -> list[str]:
    explicit: set[str] = set()
    for tok in raw_argv:
        if tok.startswith("--"):
            name = tok[2:].split("=", 1)[0]
            explicit.add(name.replace("-", "_"))

    active: list[str] = list(PRESETS.get(args.preset, []))
    if args.enable:
        for b in args.enable.split(","):
            b = b.strip()
            if b and b not in active:
                if b not in BLOCKS:
                    raise SystemExit(f"unknown block {b!r}; available: {sorted(BLOCKS)}")
                active.append(b)
    if args.disable:
        drop = {b.strip() for b in args.disable.split(",") if b.strip()}
        active = [b for b in active if b not in drop]

    for block_name in active:
        for arg_name, val in BLOCKS[block_name].items():
            if arg_name in explicit:
                continue
            setattr(args, arg_name, val)
    return sorted(active)


# Atari input shape after preprocessing.
N_CHANNELS = 4
FRAME_H = 84
FRAME_W = 84
FRAME_SIZE = N_CHANNELS * FRAME_H * FRAME_W  # 28224 floats per lane


def _make_env(
    game: str,
    seed: int,
    frame_skip: int = 4,
    repeat_action_probability: float = 0.25,
):
    """Factory returning a fully-preprocessed Atari env callable for
    gymnasium's vector-env API."""
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    def _thunk():
        env = gym.make(
            f"ALE/{game}-v5",
            frameskip=1,
            repeat_action_probability=repeat_action_probability,
        )
        env = AtariPreprocessing(
            env,
            frame_skip=frame_skip,
            screen_size=FRAME_W,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,  # we normalize on the Rust/kindle side eventually
        )
        env = FrameStackObservation(env, stack_size=N_CHANNELS)
        env.reset(seed=seed)
        return env

    return _thunk


def _foreground_residual(frames: np.ndarray) -> np.ndarray:
    """Return |pixel - per-frame median| as float32 without changing shape.

    Atari screens often devote most pixels to one flat background color. A
    raw reconstruction objective can minimize loss by encoding that constant
    while ignoring tiny controllable objects. Median subtraction is a robust,
    game-agnostic approximation of modal-background removal whenever the
    background occupies at least half the frame.
    """
    background = np.median(frames, axis=(-2, -1), keepdims=True).astype(
        np.float32, copy=False
    )
    residual = frames.astype(np.float32, copy=False) - background
    return np.abs(residual, out=residual)


def _write_frames_into_shared(
    obs_batch: np.ndarray, dest: np.ndarray, foreground_residual: bool = False
) -> None:
    """obs_batch: (num_envs, C, H, W) uint8 from gymnasium.
    dest: same-shape writable numpy view over kindle's shared host
    allocation (float32). Normalizes uint8→float32/255 and writes the
    result into `dest` in one pass — no intermediate allocation and
    no Python list conversion. The destination memory is the
    Vulkan-imported buffer; when this function returns, the WM
    session already "sees" the new frames.
    """
    if obs_batch.ndim == 5 and obs_batch.shape[-1] == 1:
        obs_batch = obs_batch[..., 0]
    if foreground_residual:
        residual = _foreground_residual(obs_batch)
        np.divide(residual, 255.0, out=dest, dtype=np.float32)
    else:
        # np.divide with an out= parameter performs the cast + scale in
        # a single loop and writes directly into the shared allocation.
        np.divide(obs_batch, 255.0, out=dest, dtype=np.float32, casting="unsafe")


def _visual_obs_tokens(
    obs_batch: np.ndarray, foreground_residual: bool = False
) -> list[list[float]]:
    """Compress the four-frame Atari stack to a meaningful 64-D token.

    The CNN consumes the full 4×84×84 stack through the visual buffer,
    while token-domain intrinsic mechanisms (RND, order, M8 and xeps) need
    their own compact view. Pooling each history frame to 4×4 preserves
    coarse spatial layout and motion across the four frames: 4×4×4 = 64.
    """
    frames = obs_batch
    if frames.ndim == 5 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if frames.ndim != 4:
        raise ValueError(f"expected NCHW Atari frames, got shape {frames.shape}")
    n, channels, height, width = frames.shape
    if channels != N_CHANNELS or height % 4 or width % 4:
        raise ValueError(
            f"expected N×{N_CHANNELS}×H×W with H/W divisible by 4, "
            f"got {frames.shape}"
        )
    token_frames = _foreground_residual(frames) if foreground_residual else frames
    pooled = token_frames.reshape(n, channels, 4, height // 4, 4, width // 4).mean(
        axis=(3, 5), dtype=np.float32
    )
    pooled *= np.float32(1.0 / 255.0)
    return pooled.reshape(n, 64).tolist()


def _transition_observations(
    reset_obs: np.ndarray, dones: np.ndarray, infos: dict[str, Any]
) -> np.ndarray:
    """Recover terminal observations from a SAME_STEP vector transition.

    Gymnasium returns reset observations for lanes that ended and stores their
    actual terminal observations in ``infos["final_obs"]``. The agent must
    learn from the terminal frame, while its next action must see the reset
    frame. Return the former without mutating the latter.
    """
    if not np.any(dones):
        return reset_obs
    if "final_obs" not in infos:
        raise RuntimeError(
            "vector env omitted final_obs for a completed SAME_STEP transition"
        )
    transition_obs = reset_obs.copy()
    final_obs = infos["final_obs"]
    for lane in np.flatnonzero(dones):
        if final_obs[lane] is None:
            raise RuntimeError(f"vector env omitted final_obs for lane {lane}")
        transition_obs[lane] = final_obs[lane]
    return transition_obs


def _pong_motion_tokens(
    obs_batch: np.ndarray,
    last_paddle_y: np.ndarray | None = None,
 ) -> tuple[np.ndarray, np.ndarray]:
    """Extract model-visible Pong ball/paddle motion features from pixels.

    Connected components isolate the ball and controlled paddle. The first 17
    entries contain presence, current/previous geometry, velocity, a wall-
    reflected intercept, and paddle error; the remaining universal token slots
    are zero. A supervised Kindle policy and the executable controller consume
    this exact same state representation.
    """
    frames = obs_batch
    if frames.ndim == 5 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if frames.ndim != 4 or frames.shape[1] < 2:
        raise ValueError(f"expected N×stack×H×W Pong frames, got {frames.shape}")

    n, _, height, width = frames.shape
    if last_paddle_y is None:
        last_paddle_y = np.full(n, height / 2.0, dtype=np.float32)
    elif last_paddle_y.shape != (n,):
        raise ValueError(f"expected {n} paddle positions, got {last_paddle_y.shape}")

    import cv2

    def objects(frame: np.ndarray):
        background = float(np.median(frame))
        mask = (np.abs(frame.astype(np.float32) - background) > 20.0).astype(
            np.uint8
        )
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, 8
        )
        balls: list[tuple[float, float, int]] = []
        paddles: list[tuple[float, float, int]] = []
        for component in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[component])
            cx, cy = (float(v) for v in centroids[component])
            if (
                ball_x_lo <= x <= ball_x_hi
                and y_lo <= y < y_hi
                and w <= 5
                and h <= 5
                and 2 <= area <= 16
            ):
                balls.append((cx, cy, area))
            if (
                x >= paddle_x_lo
                and y_lo <= y < y_hi
                and w <= 5
                and 5 <= h <= 13
                and area >= 12
            ):
                paddles.append((cx, cy, area))
        ball = max(balls, key=lambda item: item[2])[:2] if balls else None
        paddle = max(paddles, key=lambda item: item[2])[:2] if paddles else None
        return ball, paddle

    def reflect_y(y: float) -> float:
        span = float(y_hi - y_lo)
        phase = (y - y_lo) % (2.0 * span)
        return y_lo + (phase if phase <= span else 2.0 * span - phase)

    tokens = np.zeros((n, 64), dtype=np.float32)
    next_paddle_y = last_paddle_y.astype(np.float32, copy=True)
    y_lo, y_hi = max(1, int(height * 0.16)), min(height - 1, int(height * 0.92))
    ball_x_lo, ball_x_hi = int(width * 0.14), int(width * 0.86)
    paddle_x_lo = int(width * 0.86)

    for lane in range(n):
        previous_ball, _previous_paddle = objects(frames[lane, -2])
        current_ball, current_paddle = objects(frames[lane, -1])
        if current_paddle is not None:
            paddle_y = current_paddle[1]
            next_paddle_y[lane] = paddle_y
        else:
            paddle_y = float(last_paddle_y[lane])

        target_y = float(height / 2.0)
        velocity_x = 0.0
        velocity_y = 0.0
        if current_ball is not None and previous_ball is not None:
            velocity_x = current_ball[0] - previous_ball[0]
            velocity_y = current_ball[1] - previous_ball[1]
            if velocity_x > 0.05:
                frames_to_paddle = (float(paddle_x_lo) - current_ball[0]) / velocity_x
                target_y = reflect_y(current_ball[1] + velocity_y * frames_to_paddle)

        token = tokens[lane]
        token[0] = float(current_ball is not None)
        token[1] = float(previous_ball is not None)
        if current_ball is not None:
            token[2] = current_ball[0] / max(1.0, width - 1)
            token[3] = current_ball[1] / max(1.0, height - 1)
            token[14] = (paddle_x_lo - current_ball[0]) / max(1.0, width - 1)
        if previous_ball is not None:
            token[4] = previous_ball[0] / max(1.0, width - 1)
            token[5] = previous_ball[1] / max(1.0, height - 1)
        # Pong motion is only a few pixels per frame. Scaling by the full
        # 84-pixel extent made the decision boundary numerically tiny for an
        # MLP; retain sign/magnitude in a well-conditioned [-1, 1] range.
        token[6] = np.clip(velocity_x / 5.0, -1.0, 1.0)
        token[7] = np.clip(velocity_y / 5.0, -1.0, 1.0)
        token[8] = float(current_paddle is not None)
        token[9] = paddle_y / max(1.0, height - 1)
        token[10] = target_y / max(1.0, height - 1)
        token[11] = np.clip((target_y - paddle_y) / 10.0, -1.0, 1.0)
        token[12] = float(velocity_x > 0.05)
        token[13] = float(velocity_x < -0.05)
        token[15] = float(current_ball is None)
        token[16] = float(last_paddle_y[lane]) / max(1.0, height - 1)

    return tokens, next_paddle_y


def _pong_heuristic_actions(
    obs_batch: np.ndarray,
    last_paddle_y: np.ndarray | None = None,
    deadband: float = 2.0,
) -> tuple[list[int], np.ndarray]:
    """Track Pong's right paddle using the learner-visible motion token."""
    tokens, next_paddle_y = _pong_motion_tokens(obs_batch, last_paddle_y)
    actions: list[int] = []
    for token in tokens:
        if token[0] < 0.5:
            actions.append(1)  # FIRE: start/restart a rally.
        elif token[11] < -deadband / 10.0:
            actions.append(2)  # RIGHT maps to paddle-up in ALE Pong.
        elif token[11] > deadband / 10.0:
            actions.append(3)  # LEFT maps to paddle-down.
        else:
            actions.append(0)

    return actions, next_paddle_y


def _run_controller_only(
    args: argparse.Namespace,
    envs: Any,
    obs_batch: np.ndarray,
    num_actions: int,
) -> int:
    """Evaluate a random or Pong controller without constructing Kindle.

    This keeps calibrated task baselines cheap and makes it practical to run
    frame-skip-1 Pong episodes, which are several times longer than the standard
    frame-skip-4 training stream.
    """
    random_rng = np.random.default_rng(args.seed)
    pong_paddle_y: np.ndarray | None = None
    cur_scores = np.zeros(args.lanes, dtype=np.float64)
    ep_returns: list[list[float]] = [[] for _ in range(args.lanes)]
    positive_events = 0
    negative_events = 0
    t0 = time.time()

    for step in range(args.steps):
        if args.random_policy:
            actions = random_rng.integers(num_actions, size=args.lanes).tolist()
        else:
            actions, pong_paddle_y = _pong_heuristic_actions(
                obs_batch, pong_paddle_y, deadband=1.0
            )
        obs_batch, rewards, terms, truncs, _infos = envs.step(
            np.asarray(actions, dtype=np.int64)
        )
        cur_scores += rewards
        positive_events += int(np.count_nonzero(rewards > 0))
        negative_events += int(np.count_nonzero(rewards < 0))
        dones = np.logical_or(terms, truncs)
        for lane in np.flatnonzero(dones):
            ep_returns[lane].append(float(cur_scores[lane]))
            cur_scores[lane] = 0.0
        if pong_paddle_y is not None:
            pong_paddle_y[dones] = FRAME_H / 2.0

        if args.log_every and step > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            recent = [
                score
                for lane_scores in ep_returns
                for score in lane_scores[-args.recent_episodes_per_lane :]
            ]
            recent_mean = sum(recent) / len(recent) if recent else float("nan")
            print(
                f"step={step:>6} eps={sum(map(len, ep_returns)):>3} "
                f"avg_ret={recent_mean:+6.1f} | "
                f"{step * args.lanes / max(1e-3, elapsed):5.0f} env-steps/s"
            )

    envs.close()
    elapsed = time.time() - t0
    returns = [score for lane_scores in ep_returns for score in lane_scores]
    recent = [
        score
        for lane_scores in ep_returns
        for score in lane_scores[-args.recent_episodes_per_lane :]
    ]
    mean_return = sum(returns) / len(returns) if returns else float("nan")
    recent_mean = sum(recent) / len(recent) if recent else float("nan")
    print(f"\n--- {args.game} controller summary ---")
    print(f"total env-steps: {args.steps * args.lanes}")
    print(f"episodes: {len(returns)}, mean return: {mean_return:+.2f}")
    print(f"reward events: +{positive_events}/-{negative_events}")
    print(f"recent episodes: {len(recent)}, recent mean return: {recent_mean:+.2f}")
    print(
        f"wall: {elapsed:.1f}s, throughput: "
        f"{args.steps * args.lanes / max(1e-3, elapsed):.0f} env-steps/s"
    )

    checks: list[tuple[bool, str]] = []
    if args.require_recent_return is not None:
        checks.append(
            (
                bool(recent) and recent_mean >= args.require_recent_return,
                f"recent return {recent_mean:+.2f} >= {args.require_recent_return:+.2f}",
            )
        )
    if args.require_min_episodes is not None:
        checks.append(
            (
                len(returns) >= args.require_min_episodes,
                f"episodes {len(returns)} >= {args.require_min_episodes}",
            )
        )
    if checks:
        passed = all(ok for ok, _ in checks)
        details = "; ".join(
            f"{'PASS' if ok else 'FAIL'} {description}"
            for ok, description in checks
        )
        print(f"controller gate: {'PASS' if passed else 'FAIL'} ({details})")
        return 0 if passed else 2
    return 0


def homeo_for(step_reward: float) -> list[dict[str, float]]:
    """LEGACY (--legacy-homeo-reward only): per-step env reward as a
    homeo variable, value = -step_reward, target = 0, tolerance = 0.5.

    Kindle's homeostatic reward is `-max(0, |value - target| - tol)` —
    SYMMETRIC in the deviation. A Pong point won (+1) and a point lost
    (−1) both land at −0.5: the sign of the env reward is discarded and
    the agent is trained to avoid scoring events of either sign. The
    default path instead passes the signed reward through kindle's
    extrinsic channel (`set_extrinsic_reward`); this function is kept
    only so old runs can be reproduced/compared. A homeo variable is
    only appropriate for genuinely symmetric drives ("keep X near
    target"), which per-step Atari score reward is not."""
    return [
        {"value": -step_reward, "target": 0.0, "tolerance": 0.5},
    ]


class EventBalancer:
    """Rebalances per-step reward so rare events get amplified.

    Stock on-policy RL sees 21 losses per 1 win on Pong (random policy),
    so accumulated gradient tells the policy "avoid actions during
    losses" — which happens to be "don't chase the ball." This
    rebalancer tracks running counts of positive and negative events
    per lane and scales the rarer sign up so each event class
    contributes equally to the accumulated gradient.

    At balance (pos_count == neg_count) scale is 1 for both. In
    Pong's 21:1 regime, scale for positive events becomes ~21 and
    for negative stays 1 (or vice versa depending on which is rarer).

    Not pure task-agnostic — it embeds a prior that rare events are
    important — but it's a generic prior applicable to any sparse-
    reward env, not Atari-specific.
    """

    def __init__(self, num_lanes: int, eps: float = 1.0):
        import numpy as np
        self.pos = np.full(num_lanes, eps, dtype=np.float64)
        self.neg = np.full(num_lanes, eps, dtype=np.float64)

    def scale(self, rewards: np.ndarray) -> np.ndarray:
        """Return rebalanced rewards in-place-safe."""
        import numpy as np
        # Update counts first, so current-step scaling reflects the
        # distribution AFTER this step's contribution is known.
        self.pos += (rewards > 0).astype(np.float64)
        self.neg += (rewards < 0).astype(np.float64)
        # Scale factor amplifies the rarer sign to match the other.
        out = rewards.astype(np.float32).copy()
        pos_scale = self.neg / self.pos  # >1 when positives rare
        neg_scale = self.pos / self.neg  # >1 when negatives rare
        out[rewards > 0] *= pos_scale[rewards > 0].astype(np.float32)
        out[rewards < 0] *= neg_scale[rewards < 0].astype(np.float32)
        return out


class Profiler:
    """Tiny cumulative-time instrument with named sections."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def tick(self, section: str, elapsed: float) -> None:
        self.totals[section] = self.totals.get(section, 0.0) + elapsed
        self.counts[section] = self.counts.get(section, 0) + 1

    def report(self, total_elapsed: float) -> None:
        print("\n--- profile (cumulative wall time) ---")
        rows = sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)
        for section, t in rows:
            pct = 100.0 * t / max(1e-9, total_elapsed)
            mean_ms = 1000.0 * t / max(1, self.counts[section])
            print(f"  {section:<18} {t:>7.2f}s  {pct:>5.1f}%  mean={mean_ms:>6.3f}ms")
        print(f"  {'TOTAL':<18} {total_elapsed:>7.2f}s")


def main() -> int:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium isn't installed. Try: pip install 'gymnasium[atari]'", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="none",
                        help="Named block combination. 'extrinsic_grpo' is the "
                        "validated Atari learning recipe; 'rnd_only' adds RND; "
                        "'everything' stacks intrinsic/planner primitives for "
                        "collision testing. 'none' runs bare extrinsic Kindle.")
    parser.add_argument("--enable", default=None,
                        help=f"Comma-separated blocks to add. Blocks: {','.join(sorted(BLOCKS.keys()))}.")
    parser.add_argument("--disable", default=None,
                        help="Comma-separated blocks to remove from the active set.")
    parser.add_argument("--game", default="Pong",
                        help="ALE game id (e.g. Pong, Breakout, Boxing, MsPacman).")
    parser.add_argument("--lanes", type=int, default=8,
                        help="Parallel env count (vectorized). Default 8.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Synchronous steps per lane.")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="ALE frames advanced per agent decision. Default 4.")
    parser.add_argument("--repeat-action-probability", type=float, default=0.25,
                        help="ALE sticky-action probability. v5 default 0.25.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--random-policy", action="store_true",
                        help="Sample seeded uniform actions instead of Kindle "
                        "while retaining identical preprocessing/bookkeeping. "
                        "Use for reproducible task baselines.")
    parser.add_argument("--pong-heuristic", action="store_true",
                        help="Execute a pixel-motion Pong controller as an "
                        "expert baseline instead of the Kindle policy.")
    parser.add_argument("--deterministic-eval", action="store_true",
                        help="Choose the Kindle policy mode instead of sampling. "
                        "Intended for evaluation, not exploration/training.")
    parser.add_argument("--controller-only", action="store_true",
                        help="Evaluate --random-policy or --pong-heuristic "
                        "without constructing or training a Kindle agent.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-policy", type=float, default=0.0,
                        help="Policy/value optimizer LR. 0 uses Kindle's "
                        "derived learning-rate/2 default.")
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Visual latent width. 64 matches the measured "
                        "per-game frame rank while avoiding a 16-D bottleneck.")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--encoder", choices=["cnn_dqn", "cnn"],
                        default="cnn_dqn",
                        help="cnn_dqn preserves Atari's spatial layout and is "
                        "the default. cnn globally averages the feature map and "
                        "is retained only as a fast, spatially lossy ablation.")
    parser.add_argument("--async-envs", action="store_true",
                        help="Use gymnasium AsyncVectorEnv (multiprocessing). "
                        "Default is SyncVectorEnv (single process).")
    parser.add_argument("--foreground-residual", action="store_true",
                        help="Feed |pixel - per-frame median| to Kindle so "
                        "small foreground objects dominate a flat background. "
                        "Useful for Pong-like screens; off by default.")
    parser.add_argument("--reward-homeostatic", type=float, default=0.0,
                        help="Weight on the homeostatic primitive. Disabled by "
                        "default; only meaningful here with --legacy-homeo-reward.")
    parser.add_argument("--legacy-homeo-reward", action="store_true",
                        help="Restore the legacy reward routing: |env reward| "
                        "through the SYMMETRIC homeostatic channel "
                        "(-max(0, |value|-tol)) — a point won and a point lost "
                        "both score −0.5. Sign-blind; kept only for A/B "
                        "comparison against the extrinsic-channel default.")
    parser.add_argument("--reward-surprise", type=float, default=0.0)
    parser.add_argument("--reward-novelty", type=float, default=0.0)
    parser.add_argument("--reward-order", type=float, default=0.0)
    parser.add_argument("--advantage-clamp", type=float, default=None)
    parser.add_argument("--entropy-beta", type=float, default=None)
    parser.add_argument("--entropy-floor", type=float, default=None)
    parser.add_argument("--entropy-floor-frac", type=float, default=0.0,
                        help="Set entropy floor to this fraction of log(action_count). "
                        "Ignored when --entropy-floor is explicit.")
    parser.add_argument("--n-step", type=int, default=None,
                        help="Advantage lookahead horizon. Default 1 (single-step). "
                        "N ≥ 2 trains on an N-step γ-discounted Monte-Carlo return — "
                        "needed when reward events are many steps downstream of the "
                        "causal action (Pong scoring, Breakout brick bounces).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor for n-step returns. Default 0.95. "
                        "Try 0.99 on long-horizon envs.")
    parser.add_argument("--value-bootstrap", action="store_true",
                        help="TD-bootstrap the value-head target "
                        "(V_target = Σ γ^k r_{t+k} + γ^n·V(s_{t+n})). "
                        "Densifies sparse-reward TD gradient.")
    parser.add_argument("--gae-lambda", type=float, default=0.0,
                        help="GAE λ for advantage estimation. 0 disables. "
                        "0.95 is PPO default. Decouples value target from "
                        "advantage; fights value-kills-advantage collapse.")
    parser.add_argument("--value-loss-coef", type=float, default=1.0,
                        help="Value-head loss coefficient (PPO vf_coef). "
                        "1.0 = default (value dominates under shared LR on "
                        "dense reward). 0.5 or lower rebalances policy-vs-"
                        "value gradient weights.")
    parser.add_argument("--use-adam", action="store_true",
                        help="Use Adam for the trainable graphs. Recommended for CNN runs.")
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0,
                        help="Global gradient-norm limit. 0 disables clipping.")
    parser.add_argument("--grad-clip-every", type=int, default=1)
    parser.add_argument("--policy-update-interval", type=int, default=1)
    parser.add_argument("--rollout-length", type=int, default=1,
                        help="Delayed policy rollout length; 1 updates every step.")
    parser.add_argument("--advantage-normalize", action="store_true")
    parser.add_argument("--use-grpo", action="store_true",
                        help="Normalize advantages across same-task Atari lanes.")
    parser.add_argument("--use-grpo-episode", action="store_true")
    parser.add_argument("--use-sil", action="store_true")
    parser.add_argument("--sil-loss-coef", type=float, default=1.0)
    parser.add_argument("--sil-event-filter", action="store_true",
                        help="Replay only trajectories containing a positive "
                        "task-reward event.")
    parser.add_argument("--sil-event-horizon", type=int, default=0,
                        help="With event-filtered SIL, immediately retain this "
                        "many actions leading to each positive reward event. "
                        "Zero keeps whole-episode replay.")
    parser.add_argument("--sil-buffer-capacity", type=int, default=10_000)
    parser.add_argument("--use-ppo", action="store_true")
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-n-epochs", type=int, default=1)
    parser.add_argument("--recompute-base-v", action="store_true")
    parser.add_argument("--balance-events", action="store_true",
                        help="Harness-side rebalance per-step rewards so rare "
                        "events (positive in Pong/Breakout) get amplified to "
                        "contribute equally to the accumulated policy "
                        "gradient. Without it, kindle's policy commits in the "
                        "wrong direction on imbalanced envs (observed on Pong "
                        "at 400k env-steps: entropy drops 1.79→1.68 but "
                        "avg_return drifts −20.2→−21.0).")
    parser.add_argument("--extrinsic-alpha", type=float, default=1.0,
                        help="Weight on kindle's first-class extrinsic-reward "
                        "primitive. When > 0 (default 1.0), the harness supplies "
                        "the raw env reward ± directly to kindle's per-step reward "
                        "channel (additive alongside surprise/novelty/homeo/…). "
                        "Unlike homeo's distance-to-target, this is signed and "
                        "passes through — the correct integration for "
                        "per-step ±1 Atari-style reward. Set 0 to disable.")
    parser.add_argument("--rnd-alpha", type=float, default=0.0)
    parser.add_argument("--delta-goal-alpha", type=float, default=None)
    parser.add_argument("--delta-goal-threshold", type=float, default=None)
    parser.add_argument("--delta-goal-merge-radius", type=float, default=None)
    parser.add_argument("--delta-goal-surprise-threshold", type=float, default=None)
    parser.add_argument("--xeps-alpha", type=float, default=0.0)
    parser.add_argument("--xeps-grid-resolution", type=float, default=None)
    parser.add_argument("--planner-horizon", type=int, default=0)
    parser.add_argument("--planner-samples", type=int, default=None)
    parser.add_argument("--planner-every", type=int, default=0)
    parser.add_argument("--planner-noise-sigma", type=float, default=0.0,
                        help="GRAM stochastic WM: per-element N(0,σ²) noise "
                        "added to each planner WM rollout step. Default 0 = "
                        "deterministic. Try 0.05–0.2 for Atari latents.")
    parser.add_argument("--wm-stochastic", type=int, default=0,
                        help="Enable learned per-state σ-head; planner noise "
                        "scales by σ_learned(z,a) × planner_noise_sigma.")
    parser.add_argument("--wm-sigma-loss-coef", type=float, default=0.5)
    parser.add_argument("--recon-loss-coef", type=float, default=1.0,
                        help="Anti-collapse reconstruction loss on the WM/CNN "
                        "encoder (decode z back to the frame). Without it, the "
                        "encoder can collapse to a near-constant z: the WM loss "
                        "goes tiny (trivially predicting the constant) while the "
                        "policy sees no state and stays uniform. The ARC GRPO "
                        "default 1.0 reconstructs the compact motion token; "
                        "the ARC recipe uses 10.0 with a visual target.")
    parser.add_argument("--recon-visual-target", action="store_true",
                        help="Reconstruct the downsampled visual frame instead "
                        "of the coarse 64-D motion token. Use whenever "
                        "--recon-loss-coef > 0 here.")
    parser.add_argument("--recent-episodes-per-lane", type=int, default=5)
    parser.add_argument("--require-recent-return", type=float, default=None,
                        help="Exit 2 unless the final recent mean reaches this value.")
    parser.add_argument("--require-min-episodes", type=int, default=None,
                        help="Exit 2 unless at least this many episodes complete.")
    parser.add_argument("--require-latent-std", type=float, default=None,
                        help="Exit 2 unless final mean per-dimension latent std "
                        "across lanes reaches this anti-collapse threshold.")
    args = parser.parse_args()
    active_blocks = _apply_blocks(args, sys.argv[1:])
    if args.use_grpo and not args.advantage_normalize:
        parser.error("--use-grpo requires --advantage-normalize")
    if args.use_grpo_episode and not args.use_grpo:
        parser.error("--use-grpo-episode requires --use-grpo")
    if args.sil_event_filter and not args.use_sil:
        parser.error("--sil-event-filter requires --use-sil")
    if args.sil_event_horizon < 0:
        parser.error("--sil-event-horizon must be non-negative")
    if args.sil_event_horizon > 0 and not args.sil_event_filter:
        parser.error("--sil-event-horizon requires --sil-event-filter")
    if args.sil_buffer_capacity <= 0:
        parser.error("--sil-buffer-capacity must be positive")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be positive")
    if not 0.0 <= args.repeat_action_probability <= 1.0:
        parser.error("--repeat-action-probability must be in [0, 1]")
    if args.ppo_n_epochs > 1 and not args.use_ppo:
        parser.error("--ppo-n-epochs > 1 requires --use-ppo")
    if args.random_policy and args.pong_heuristic:
        parser.error("--random-policy and --pong-heuristic are mutually exclusive")
    if args.deterministic_eval and (args.random_policy or args.pong_heuristic):
        parser.error("--deterministic-eval applies only to the Kindle policy")
    if args.pong_heuristic and args.game.lower() != "pong":
        parser.error("--pong-heuristic requires --game Pong")
    if args.controller_only and not (args.random_policy or args.pong_heuristic):
        parser.error("--controller-only requires --random-policy or --pong-heuristic")
    if args.controller_only and args.require_latent_std is not None:
        parser.error("--require-latent-std is unavailable with --controller-only")
    if active_blocks:
        print(f"active blocks: {','.join(active_blocks)}")
    else:
        print("active blocks: (none — bare kindle)")

    # --- Build vectorized envs ---
    thunks = [
        _make_env(
            args.game,
            args.seed + i,
            args.frame_skip,
            args.repeat_action_probability,
        )
        for i in range(args.lanes)
    ]
    # SAME_STEP returns a fresh reset observation immediately for completed
    # lanes and preserves the terminal observation in info["final_obs"]. This
    # lets us train on the true terminal transition without silently dropping
    # the next action, as Gymnasium's NEXT_STEP default would do.
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
    if args.async_envs:
        envs = gym.vector.AsyncVectorEnv(
            thunks, shared_memory=True, autoreset_mode=autoreset_mode
        )
    else:
        envs = gym.vector.SyncVectorEnv(thunks, autoreset_mode=autoreset_mode)
    obs_batch, _info = envs.reset(seed=args.seed)
    # gymnasium returns (num_envs, C, H, W, 1) when grayscale_newaxis=True or
    # (num_envs, C, H, W) when False. We configured False, so expect 4D.
    print(
        f"env={args.game} lanes={args.lanes} obs={obs_batch.shape} "
        f"dtype={obs_batch.dtype} action_space={envs.single_action_space} "
        f"policy={'random' if args.random_policy else 'pong-heuristic' if args.pong_heuristic else 'kindle-greedy' if args.deterministic_eval else 'kindle'}"
    )
    num_actions = int(envs.single_action_space.n)
    if args.controller_only:
        return _run_controller_only(args, envs, obs_batch, num_actions)

    import kindle

    if args.entropy_floor is None and args.entropy_floor_frac > 0:
        args.entropy_floor = args.entropy_floor_frac * math.log(num_actions)

    # --- Build kindle BatchAgent with CNN encoder ---
    # With CNN encoder, obs_dim is the SECONDARY obs-token width (fed to
    # reward primitives that need a flat token, e.g. order digest). It
    # must be ≤ OBS_TOKEN_DIM=64. The raw 84×84×4 frame travels via
    # set_visual_obs(), not the obs token.
    agent_kwargs: dict[str, Any] = {
        "obs_dim": 64,
        "num_actions": num_actions,
        "batch_size": args.lanes,
        # All vector lanes are independent copies of the same Atari task.
        "env_ids": [0] * args.lanes,
        "seed": args.seed,
        "learning_rate": args.lr,
        "lr_policy": args.lr_policy if args.lr_policy > 0 else None,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "encoder_kind": args.encoder,
        "encoder_channels": N_CHANNELS,
        "encoder_height": FRAME_H,
        "encoder_width": FRAME_W,
    }
    for name in [
        "rnd_alpha", "delta_goal_alpha", "delta_goal_threshold",
        "delta_goal_merge_radius", "delta_goal_surprise_threshold",
        "xeps_alpha", "xeps_grid_resolution", "planner_samples",
        "reward_homeostatic", "reward_surprise", "reward_novelty", "reward_order",
        "advantage_clamp", "entropy_beta", "entropy_floor",
        "n_step", "gamma",
        "extrinsic_alpha",
    ]:
        v = getattr(args, name, None)
        if v is not None:
            agent_kwargs[
                {
                    "rnd_alpha": "rnd_reward_alpha",
                    "xeps_alpha": "xeps_reward_alpha",
                    "extrinsic_alpha": "extrinsic_reward_alpha",
                }.get(
                    name, name
                )
            ] = v
    if args.planner_horizon > 0:
        agent_kwargs["planner_horizon"] = args.planner_horizon
    if args.planner_noise_sigma > 0.0:
        agent_kwargs["planner_noise_sigma"] = args.planner_noise_sigma
    if args.wm_stochastic:
        agent_kwargs["wm_stochastic"] = True
        agent_kwargs["wm_sigma_loss_coef"] = args.wm_sigma_loss_coef
    if args.value_bootstrap:
        agent_kwargs["value_bootstrap"] = True
    if args.gae_lambda > 0:
        agent_kwargs["gae_lambda"] = args.gae_lambda
    if args.value_loss_coef != 1.0:
        agent_kwargs["value_loss_coef"] = args.value_loss_coef
    agent_kwargs.update(
        use_adam=args.use_adam,
        adam_eps=args.adam_eps,
        grad_clip_every=args.grad_clip_every,
        policy_update_interval=args.policy_update_interval,
        rollout_length=args.rollout_length,
        advantage_normalize=args.advantage_normalize,
        use_grpo=args.use_grpo,
        use_grpo_episode=args.use_grpo_episode,
        use_sil=args.use_sil,
        sil_loss_coef=args.sil_loss_coef,
        sil_event_filter=args.sil_event_filter,
        sil_event_horizon=args.sil_event_horizon,
        sil_buffer_capacity=args.sil_buffer_capacity,
        use_ppo=args.use_ppo,
        ppo_clip_eps=args.ppo_clip_eps,
        ppo_n_epochs=args.ppo_n_epochs,
        recompute_base_v=args.recompute_base_v,
    )
    if args.grad_clip_norm > 0:
        agent_kwargs["grad_clip_norm"] = args.grad_clip_norm
    if args.recon_loss_coef > 0.0:
        agent_kwargs["recon_loss_coef"] = args.recon_loss_coef
        agent_kwargs["recon_visual_target"] = args.recon_visual_target

    agent = kindle.BatchAgent(**agent_kwargs)

    # --- Frame delivery path (pure device-local) ---
    # Meganeura allocates every graph input as Memory::Shared —
    # DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT. Writing into the
    # memoryview lands bytes directly in the GPU-side buffer, no
    # staging, no explicit upload, no external-memory import. One
    # memcpy per frame is the theoretical minimum for CPU-produced
    # observations.
    frame_mv = agent.visual_obs_memoryview()
    frame_buf = np.frombuffer(frame_mv, dtype=np.float32).reshape(
        args.lanes, N_CHANNELS, FRAME_H, FRAME_W
    )
    assert frame_buf.flags.writeable, "frame_buf must be writable"
    print(
        f"agent ready ({args.encoder} {N_CHANNELS}×{FRAME_H}×{FRAME_W} "
        f"→ {args.latent_dim}); "
        f"visual_obs buffer: {agent.visual_obs_host_size() / (1 << 10):.1f} KiB "
        f"(Memory::Shared, device-local + host-visible)"
    )

    balancer: EventBalancer | None = (
        EventBalancer(args.lanes) if args.balance_events else None
    )

    # --- Training loop ---
    cur_scores = np.zeros(args.lanes, dtype=np.float64)
    ep_returns: list[list[float]] = [[] for _ in range(args.lanes)]
    ep_count = 0
    positive_events = 0
    negative_events = 0
    prof = Profiler()
    random_rng = np.random.default_rng(args.seed)
    pong_paddle_y: np.ndarray | None = None
    # The full frame stack drives the CNN; this compact motion-preserving view
    # drives token-domain intrinsic rewards and memories.
    obs_token_small = _visual_obs_tokens(obs_batch, args.foreground_residual)
    t0 = time.time()

    def push_frame(obs_batch_u8: np.ndarray) -> None:
        """Normalize uint8 NCHW → float32/255 directly into the
        device-local, host-visible `visual_obs` buffer. After this
        call, the WM session is ready to consume the new frame on
        its next step — no further agent API call needed."""
        _write_frames_into_shared(
            obs_batch_u8, frame_buf, foreground_residual=args.foreground_residual
        )

    # Initial frame for step 0's act(). Inside the loop the frame is
    # pushed once per step, just before observe(); the same buffer
    # content then serves the NEXT iteration's act() — re-pushing the
    # unchanged obs_batch before act() (the old behavior) doubled the
    # reported frame-conversion cost for no effect.
    push_frame(obs_batch)

    for step in range(args.steps):
        # 0) Periodic planner invocation (queues actions for the next K
        # act() calls). Triggered by `--planner-every` (set by the
        # `planner` block to 20).
        if (
            args.planner_horizon > 0
            and args.planner_every > 0
            and step > 0
            and step % args.planner_every == 0
        ):
            agent.plan_and_queue(num_actions)

        # 1) Sample actions (batched). The visual slot already holds
        # the current frame (pushed before the loop / at the end of
        # the previous iteration).
        t = time.time()
        if args.random_policy:
            actions = random_rng.integers(num_actions, size=args.lanes).tolist()
        elif args.pong_heuristic:
            # Forward the current state through Kindle even when the controller
            # supplies the executed action. This caches the matching V(s) and
            # entropy for regular learning/SIL; the sampled action is discarded.
            agent.act(obs_token_small)
            actions, pong_paddle_y = _pong_heuristic_actions(
                obs_batch, pong_paddle_y, deadband=1.0
            )
        else:
            actions = agent.act(obs_token_small, args.deterministic_eval)
        prof.tick("agent.act", time.time() - t)

        # 2) Step envs.
        t = time.time()
        actions_np = np.array(actions, dtype=np.int64)
        next_obs, rewards, terms, truncs, infos = envs.step(actions_np)
        prof.tick("envs.step", time.time() - t)

        cur_scores += rewards
        positive_events += int(np.count_nonzero(rewards > 0))
        negative_events += int(np.count_nonzero(rewards < 0))
        dones = np.logical_or(terms, truncs)
        transition_obs = _transition_observations(next_obs, dones, infos)

        # 3) True post-action frame for observe()'s WM pass. On a terminal
        # transition this differs from next_obs, which already contains the
        # SAME_STEP reset frame.
        t = time.time()
        push_frame(transition_obs)
        prof.tick("push_frame", time.time() - t)
        transition_tokens = _visual_obs_tokens(
            transition_obs, args.foreground_residual
        )

        t = time.time()
        shaped = balancer.scale(rewards) if balancer is not None else rewards
        if args.extrinsic_alpha > 0:
            # Raw env reward, signed, passed through kindle's extrinsic
            # primitive (mixed as extrinsic_reward_alpha × reward inside
            # observe). Shaped=balanced copy if --balance-events is on.
            agent.set_extrinsic_reward(shaped.astype(np.float32, copy=False))
        # Homeo channel only under --legacy-homeo-reward: its distance-
        # to-target form is symmetric (sign-blind), see homeo_for().
        homeos = (
            [homeo_for(float(shaped[i])) for i in range(args.lanes)]
            if args.legacy_homeo_reward
            else None
        )
        agent.observe(transition_tokens, [int(a) for a in actions], homeostatic=homeos)
        prof.tick("agent.observe", time.time() - t)

        # Episode bookkeeping.
        for i, done in enumerate(dones):
            if done:
                ep_returns[i].append(float(cur_scores[i]))
                cur_scores[i] = 0.0
                ep_count += 1
                agent.mark_boundary(i)

        # The next act sees the reset observation for completed lanes. For
        # ordinary transitions transition_obs is next_obs, so the frame is
        # already staged and no second conversion is needed.
        obs_batch = next_obs
        if np.any(dones):
            t = time.time()
            push_frame(obs_batch)
            prof.tick("push_reset_frame", time.time() - t)
            obs_token_small = _visual_obs_tokens(
                obs_batch, args.foreground_residual
            )
        else:
            obs_token_small = transition_tokens

        if args.log_every and step > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = step * args.lanes / max(1e-3, elapsed)
            all_recent = [
                r
                for lane_rets in ep_returns
                for r in lane_rets[-args.recent_episodes_per_lane:]
            ]
            avg_ret = (
                sum(all_recent) / len(all_recent)
                if all_recent
                else float("nan")
            )

            diags = agent.diagnostics()
            d = diags[0]

            def _safe(x, default=float("nan")):
                return float(x) if x is not None else default

            wm = _safe(d.get("loss_world_model"))
            pi = _safe(d.get("loss_policy"))
            ent = _safe(d.get("policy_entropy"))
            surp = _safe(d.get("reward_surprise"))
            nov = _safe(d.get("reward_novelty"))
            hom = _safe(d.get("reward_homeo"))
            rew = _safe(d.get("reward_mean"))
            # Latent-collapse canary: per-dim std of z across lanes,
            # averaged over dims. Near-zero + tiny WM loss + max
            # entropy together mean the encoder collapsed to a
            # constant and the policy is state-blind.
            z = np.asarray(agent.latents(), dtype=np.float32)
            z_std = float(z.std(axis=0).mean()) if z.size else 0.0
            _sil_attempted, sil_fired, _sil_active = agent.sil_counters()
            print(
                f"step={step:>6} eps={ep_count:>3} avg_ret={avg_ret:+6.1f} "
                f"| wm={wm:.3f} pi={pi:+7.2f} ent={ent:.2f} zstd={z_std:.3f} "
                f"r={rew:+5.2f} surp={surp:+4.2f} nov={nov:+4.2f} hom={hom:+6.2f} "
                f"sil={agent.sil_buffer_size()}/{sil_fired} "
                f"| {sps:5.0f} env-steps/s"
            )

    envs.close()

    # --- Summary ---
    elapsed = time.time() - t0
    sps = args.steps * args.lanes / max(1e-3, elapsed)
    total_episodes = sum(len(r) for r in ep_returns)
    returns = [r for lane in ep_returns for r in lane]
    mean_ret = sum(returns) / len(returns) if returns else float("nan")
    final_recent = [
        r
        for lane_rets in ep_returns
        for r in lane_rets[-args.recent_episodes_per_lane:]
    ]
    recent_mean = (
        sum(final_recent) / len(final_recent) if final_recent else float("nan")
    )
    final_z = np.asarray(agent.latents(), dtype=np.float32)
    final_z_std = float(final_z.std(axis=0).mean()) if final_z.size else 0.0
    print(f"\n--- {args.game} summary ---")
    print(f"total env-steps: {args.steps * args.lanes}")
    print(f"episodes: {total_episodes}, mean return: {mean_ret:+.2f}")
    print(f"reward events: +{positive_events}/-{negative_events}")
    print(
        f"recent episodes: {len(final_recent)}, "
        f"recent mean return: {recent_mean:+.2f}, latent std: {final_z_std:.4f}"
    )
    print(f"wall: {elapsed:.1f}s, throughput: {sps:.0f} env-steps/s")
    prof.report(elapsed)

    gate_checks: list[tuple[bool, str]] = []
    if args.require_recent_return is not None:
        gate_checks.append(
            (
                bool(final_recent) and recent_mean >= args.require_recent_return,
                f"recent return {recent_mean:+.2f} >= {args.require_recent_return:+.2f}",
            )
        )
    if args.require_min_episodes is not None:
        gate_checks.append(
            (
                total_episodes >= args.require_min_episodes,
                f"episodes {total_episodes} >= {args.require_min_episodes}",
            )
        )
    if args.require_latent_std is not None:
        gate_checks.append(
            (
                final_z_std >= args.require_latent_std,
                f"latent std {final_z_std:.4f} >= {args.require_latent_std:.4f}",
            )
        )
    if gate_checks:
        passed = all(ok for ok, _ in gate_checks)
        details = "; ".join(
            f"{'PASS' if ok else 'FAIL'} {description}"
            for ok, description in gate_checks
        )
        print(f"learning gate: {'PASS' if passed else 'FAIL'} ({details})")
        return 0 if passed else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
