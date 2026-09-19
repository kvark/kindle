"""Record raw Atari frames while the unchanged runner evaluates a policy.

Example:
    python examples/record_atari.py DINO ALE/Pong-v5 --restore CHECKPOINT \
        --evaluate --steps 10000 --atari-protocol published \
        --output runs/pong-video.jsonl --video runs/pong.mp4

Requires ffmpeg. The movie contains each actual emulator step (including reset
no-ops) at nominal 60 Hz, not wall-clock acting latency or the resized agent input.
Reset images add no movie frames. Recording is not a throughput benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import gymnasium as gym
import numpy as np

import atari


class AtariVideo(gym.Wrapper):
    def __init__(self, environment, path: Path, ffmpeg: str):
        super().__init__(environment)
        height, width, channels = environment.observation_space.shape
        if channels != 3 or height % 2 or width % 2:
            raise ValueError("recording requires even-sized RGB frames")
        self.frame_count = 0
        self.frame_sha256 = hashlib.sha256()
        self._closed = False
        self._encoder = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-n", "-f", "rawvideo",
             "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
             "-framerate", "60", "-i", "pipe:0", "-an", "-c:v", "libx264",
             "-threads", "1", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )

    def step(self, action):
        result = self.env.step(action)
        frame = result[0]
        if frame.shape != self.observation_space.shape or frame.dtype != np.uint8:
            raise ValueError("emulator frame shape or dtype changed")
        pixels = frame.tobytes(order="C")
        self._encoder.stdin.write(pixels)
        self.frame_sha256.update(pixels)
        self.frame_count += 1
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._encoder.stdin.close()
            finally:
                status = self._encoder.wait()
            if status:
                raise RuntimeError(f"ffmpeg failed with exit status {status}")
        finally:
            self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, runner_args = parser.parse_known_args()
    flags = {arg.split("=", 1)[0] for arg in runner_args}
    if not flags.intersection({"--evaluate", "--random-policy"}):
        parser.error("record only frozen evaluation or a random-policy control")
    if "--append-output" in flags:
        parser.error("a recording requires a fresh run log")
    manifest = args.video.with_suffix(args.video.suffix + ".json")
    targets = [args.video, args.output, manifest]
    if len({path.resolve() for path in targets}) != len(targets):
        parser.error("video, run log and manifest must be different files")
    for path in targets:
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
        if not path.parent.is_dir():
            parser.error(f"missing output directory: {path.parent}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required for recording")

    original_make, original_argv = atari.gym.make, sys.argv
    recorder = None

    def make_recorded(*positional, **keywords):
        nonlocal recorder
        if recorder is not None:
            raise RuntimeError("expected exactly one Atari environment")
        environment = original_make(*positional, **keywords)
        try:
            recorder = AtariVideo(environment, args.video, ffmpeg)
        except BaseException:
            environment.close()
            raise
        return recorder

    try:
        atari.gym.make = make_recorded
        sys.argv = [atari.__file__, *runner_args, "--output", str(args.output)]
        atari.main()
    finally:
        atari.gym.make, sys.argv = original_make, original_argv
        if recorder is not None:
            recorder.close()

    with args.output.open() as source:
        header = json.loads(next(source))
        end = None
        for line in source:
            end = json.loads(line)
    if end is None or end["event"] != "run_end" or end["learner_updates"] != 0:
        raise ValueError("recording has no complete zero-update run")
    if recorder.frame_count != end["executed_action_frames"] + end["reset_noop_frames"]:
        raise ValueError("recorded and executed emulator frame counts differ")
    metadata = {
        "format": 1, "mode": header["mode"], "environment": header["environment"],
        "seed": header["seed"], "nominal_fps": 60,
        "frames": recorder.frame_count, "raw_frame_sha256": recorder.frame_sha256.hexdigest(),
        "run_log": str(args.output), "run_log_sha256": atari.sha256_file(args.output),
        "video": str(args.video), "video_sha256": atari.sha256_file(args.video),
        "recorder_sha256": atari.sha256_file(__file__),
        "runner_sha256": header["runner_sha256"],
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
    }
    with manifest.open("x") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
