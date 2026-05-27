"""Generate pretraining frames from ARC-AGI-3 *training* games (NOT eval).

Runs random-action episodes across all available games, collecting one
frame per micro-step. Output is a stacked uint8 (N, 64, 64) array matching
gen_grids.py format so train_dae.py can consume it interchangeably.

Used as the realistic alternative to procedural grids — gives the encoder
genuine ARC-AGI-3 visual statistics (sprite shapes, color palettes,
spatial layouts) that procedural rectangles/circles don't reproduce.

The rules permit pretraining on *any* data that isn't the eval set. The
games hit here are the public training games (whatever the local
arc_agi toolkit exposes).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    try:
        from arc_agi import Arcade
        from arcengine import GameAction, GameState
    except ImportError as exc:
        print(f"missing arc_agi toolkit: {exc}", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--out", default="/tmp/aff_runs/pretrain_arc_frames.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ep-steps", type=int, default=200,
                   help="Reset env after this many random actions per ep.")
    p.add_argument("--exclude-prefixes", default=None,
                   help="Comma-separated game prefixes to skip (e.g. eval set).")
    args = p.parse_args()

    arcade = Arcade()
    all_envs = arcade.get_environments()
    # Dedup by 4-char prefix (same logic as harness).
    seen: set[str] = set()
    env_infos = []
    for e in sorted(all_envs, key=lambda x: x.game_id):
        k = e.game_id[:4]
        if k in seen:
            continue
        seen.add(k)
        env_infos.append(e)
    if args.exclude_prefixes:
        ex = [x.strip() for x in args.exclude_prefixes.split(",") if x.strip()]
        env_infos = [e for e in env_infos
                     if not any(e.game_id.startswith(x) for x in ex)]
    print(f"using {len(env_infos)} games: {[e.game_id[:4] for e in env_infos]}")

    action_by_value = {int(a.value): a for a in GameAction}

    rng = np.random.default_rng(args.seed)
    frames = np.empty((args.n, 64, 64), dtype=np.uint8)
    n_written = 0

    while n_written < args.n:
        # Round-robin across games
        for info in env_infos:
            if n_written >= args.n:
                break
            env = arcade.make(info.game_id)
            obs = env.reset()
            avail = list(obs.available_actions) or [1]
            for _ in range(args.max_ep_steps):
                if n_written >= args.n:
                    break
                if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER,
                                  GameState.WIN):
                    obs = env.reset()
                    avail = list(obs.available_actions) or [1]
                    continue
                if obs.frame:
                    f = np.asarray(obs.frame[0], dtype=np.uint8)
                    if f.shape == (64, 64):
                        frames[n_written] = f
                        n_written += 1
                        if n_written % 2000 == 0:
                            print(f"  {n_written}/{args.n}")
                # Random action
                a_val = int(avail[rng.integers(0, len(avail))])
                action = action_by_value[a_val]
                if action.is_complex():
                    obs = env.step(action, data={
                        "x": int(rng.integers(0, 64)),
                        "y": int(rng.integers(0, 64)),
                    })
                else:
                    obs = env.step(action)
                if obs is None:
                    break
                if list(obs.available_actions):
                    avail = list(obs.available_actions)

    np.savez_compressed(args.out, frames=frames)
    print(f"wrote {n_written} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
