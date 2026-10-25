"""CPU-only cross-interpreter check of the shared Phase 2 Atari protocol.

Run in Kindle and upstream environments, then compare trace_sha256. Deliberate
short artificial cutoffs exercise reset/bootstrapping without long training.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

import numpy as np

from upstream_matched import make_environments, observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--natural", action="store_true", help="1536 ticks with the real 100000-frame cutoff")
    args = parser.parse_args()
    records = []
    for game in ("pong", "breakout", "seaquest"):
        envs, initial = make_environments(game, 1009, 6)
        digest = hashlib.sha256()
        counts = dict(actions=0, resets=6, terminals=0, cutoffs=0)

        def observe(row, stream):
            digest.update(row["image"].tobytes())
            digest.update(envs[stream].unwrapped.ale.getRAM().tobytes())
            digest.update(struct.pack("<f???", row["reward"], row["is_first"], row["is_last"], row["is_terminal"]))

        try:
            for stream, (env, row) in enumerate(zip(envs, initial)):
                if not args.natural:
                    env._max_episode_frames = 256
                observe(row, stream)
            for step in range(1536 if args.natural else 160):
                for stream, env in enumerate(envs):
                    result = env.step((step * 7 + stream * 5) % 18)
                    row = observation(result)
                    counts["actions"] += 1
                    counts["terminals"] += int(result[2])
                    counts["cutoffs"] += int(result[3])
                    observe(row, stream)
                    if row["is_last"]:
                        observe(observation(env.reset(), first=True), stream)
                        counts["resets"] += 1
            records.append(dict(game=game, trace_sha256=digest.hexdigest(), **counts,
                                executed_action_frames=[e.executed_action_frames for e in envs]))
        finally:
            for env in envs:
                env.close()
    result = dict(protocol="phase2-shared-ale-trace-v1", seed=1009, streams=6,
                  forced_cutoff_frames=None if args.natural else 256, records=records,
                  versions={name: importlib.metadata.version(name) for name in ("gymnasium", "ale-py", "numpy", "pillow")})
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
