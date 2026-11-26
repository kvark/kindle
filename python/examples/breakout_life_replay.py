"""Per-life and per-descent play statistics from frozen Breakout evaluation logs.

CPU only; reads logs and never writes beside them. Each log's recorded actions
are replayed in emulators built from the log's own header (seeds, protocol,
sticky probability), every reward and boundary is checked against the log, and
RAM supplies what the log omits: the life each action belongs to and the ball
and paddle positions. A descent is a downward run of the ball that ends either
in a turn upward below row 150 (returned) or in a lost life (missed). Speeds
are source pixels per agent action over that run.

    python breakout_life_replay.py LOG.jsonl... [--json OUTPUT.json]
"""
import argparse
import collections
import json
from pathlib import Path
import statistics

import ale_py
import gymnasium as gym

from atari import ATARI_PROTOCOLS, DreamerAtariPreprocessing

HIT_BINS = ((0, 0, "serve"), (1, 3, "1-3"), (4, 11, "4-11"), (12, 10**9, "12+"))


def ball_and_paddle(env):
    ram = env.unwrapped.ale.getRAM()
    ball = (int(ram[99]) - 48, int(ram[101]) + 9) if 0 < ram[101] <= 187 else None
    return ball, int(ram[72]) - 39          # ball centre x and top y; centre of the 16-pixel paddle


def replay(path):
    """Return (lives, descents) for the log's counted natural episodes."""
    with open(path) as source:
        events = [json.loads(line) for line in source]
    header = events[0]
    protocol = ATARI_PROTOCOLS[header["atari_protocol"]]
    environments = []
    for seed in header["environment_seeds"]:
        env = gym.make(header["environment"], frameskip=1, repeat_action_probability=header["sticky_actions"],
                       full_action_space=protocol.full_action_space)
        env = DreamerAtariPreprocessing(env, noop_max=protocol.noop_max,
                                        max_episode_frames=protocol.max_episode_frames, screen_size=None)
        env.reset(seed=seed)
        environments.append(env)
    limit = header.get("evaluation_episodes_per_stream")
    count = len(environments)
    remaining = [env.unwrapped.ale.lives() for env in environments]
    first = remaining[0]
    episode, hits, steps = [0] * count, [0] * count, [0] * count
    returns, track, paddles = [0.0] * count, [[] for _ in range(count)], [[] for _ in range(count)]
    lives, descents = [], []

    def descent(stream):
        points = track[stream]
        index = len(points) - 1
        while index > 0 and points[index][1] > points[index - 1][1]:
            index -= 1
        run = points[index:]
        if len(run) < 2:
            return None
        return (run[-1][1] - run[0][1]) / (len(run) - 1), abs(run[-1][0] - run[0][0]) / (len(run) - 1)

    for event in events[1:]:
        if event["event"] != "transition":
            continue
        for stream, env in enumerate(environments):
            _, reward, terminated, truncated, _ = env.step(event["actions"][stream])
            if (float(reward) != float(event["rewards"][stream]) or bool(terminated) != event["terminated"][stream]
                    or bool(truncated) != event["truncated"][stream]):
                raise ValueError(f"{path}: replay differs from the log at run step {event['run_step']}")
            counted = limit is None or episode[stream] < limit
            life = first - remaining[stream]
            returns[stream] += float(reward)
            steps[stream] += 1
            ball, paddle = ball_and_paddle(env)
            if ball is not None:
                points = track[stream]
                if len(points) >= 2 and ball[1] < points[-1][1] and points[-1][1] > points[-2][1] and points[-1][1] > 150:
                    speed = descent(stream)
                    if speed and counted:
                        descents.append(dict(life=life, hits=hits[stream], returned=True,
                                             vertical=speed[0], horizontal=speed[1]))
                    hits[stream] += 1
                points.append(ball)
                paddles[stream].append(paddle)
            now = env.unwrapped.ale.lives()
            if now < remaining[stream] or terminated or truncated:
                speed = descent(stream)
                offset = moving = None
                if track[stream]:
                    offset = track[stream][-1][0] - paddles[stream][-1]
                    travel = paddles[stream][-1] - paddles[stream][max(len(paddles[stream]) - 4, 0)]
                    moving = "still" if travel == 0 else "toward" if (travel > 0) == (offset > 0) else "away"
                if counted:
                    if speed:
                        descents.append(dict(life=life, hits=hits[stream], returned=False,
                                             vertical=speed[0], horizontal=speed[1]))
                    lives.append(dict(life=life, episode_return=returns[stream], actions=steps[stream],
                                      hits=hits[stream], miss_offset=None if offset is None else abs(offset),
                                      paddle=moving))
                hits[stream], steps[stream], returns[stream] = 0, 0, 0.0
                track[stream], paddles[stream] = [], []
            remaining[stream] = now
            if terminated or truncated:
                env.reset()
                remaining[stream] = env.unwrapped.ale.lives()
                episode[stream] += 1
    return lives, descents


def rate(rows):
    return dict(descents=len(rows), returned=sum(row["returned"] for row in rows) / len(rows) if rows else None)


def summarize(lives, descents):
    by_life = []
    for index in sorted({row["life"] for row in lives}):
        rows = [row for row in lives if row["life"] == index]
        offsets = [row["miss_offset"] for row in rows if row["miss_offset"] is not None]
        by_life.append(dict(
            life=index + 1, lives=len(rows),
            mean_return=statistics.mean(row["episode_return"] for row in rows),
            median_return=statistics.median(row["episode_return"] for row in rows),
            zero_return=sum(row["episode_return"] == 0 for row in rows),
            mean_actions=statistics.mean(row["actions"] for row in rows),
            mean_hits=statistics.mean(row["hits"] for row in rows),
            median_miss_offset=statistics.median(offsets) if offsets else None,
            paddle=dict(collections.Counter(row["paddle"] for row in rows if row["paddle"])),
            **rate([row for row in descents if row["life"] == index])))
    hit_bins = lambda rows: {name: rate([row for row in rows if low <= row["hits"] <= high])
                             for low, high, name in HIT_BINS}
    vertical = {name: rate([row for row in descents if low <= row["vertical"] < high])
                for low, high, name in ((0, 6, "<6"), (6, 9, "6-9"), (9, 13, "9-13"), (13, 10**9, ">=13"))}
    return dict(all=rate(descents), by_life=by_life, by_hits=hit_bins(descents),
                first_life_by_hits=hit_bins([row for row in descents if row["life"] == 0]),
                by_vertical_speed=vertical)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    gym.register_envs(ale_py)
    results, pooled = {}, ([], [])
    for path in args.logs:
        lives, descents = replay(path)
        pooled[0].extend(lives)
        pooled[1].extend(descents)
        results[str(path)] = summarize(lives, descents)
    results["pooled"] = summarize(*pooled)
    text = lambda cell: "  -  " if cell["returned"] is None else f"{cell['returned']:.2f} ({cell['descents']})"
    for name, result in results.items():
        print(name)
        for row in result["by_life"]:
            print(f"  life {row['life']}: return {row['mean_return']:6.2f} (median {row['median_return']:4.1f}, "
                  f"zero {row['zero_return']:2d}/{row['lives']}) actions {row['mean_actions']:6.1f} "
                  f"hits {row['mean_hits']:5.2f} returned {text(row)} "
                  f"miss offset {row['median_miss_offset']} px paddle {row['paddle']}")
        print("  returned by hits so far: " + ", ".join(f"{k} {text(v)}" for k, v in result["by_hits"].items()))
        print("  first life by hits so far: " + ", ".join(f"{k} {text(v)}" for k, v in result["first_life_by_hits"].items()))
        print("  returned by vertical speed: " + ", ".join(f"{k} {text(v)}" for k, v in result["by_vertical_speed"].items()))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
