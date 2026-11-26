"""Raw RGB positive controls for offline probes, not an agent observation path.

RGB56 single-frame (9408 scalars) and two-frame (18816 scalars) deliberately
exceed the 3136-dimensional representation budget. They test label decodability,
not a size-matched learned-encoder claim. Neither includes actions or RAM.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from atari import sha256_file
from kindle._representation_probe import PROTOCOL


def pixel_features(clips):
    frames = np.asarray([[np.asarray(Image.fromarray(frame).resize((56, 56), Image.Resampling.BILINEAR))
                          for frame in clip[-2:]] for clip in clips], dtype=np.float32) / 255
    return {"rgb56/single_frame": frames[:, -1].reshape(len(clips), -1),
            "rgb56/two_frames": frames.reshape(len(clips), -1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads((args.dataset / "manifest.json").read_text())
    if source["protocol"] != PROTOCOL or source["status"] != "complete":
        parser.error("complete matching corpus required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=PROTOCOL, status="running", architecture="raw_rgb56",
                    encoder_sha256=None, limit_clips=None,
                    dataset_manifest_sha256=sha256_file(args.dataset / "manifest.json"), files=[],
                    limits=["9408/18816 features, not a size-matched encoder", "offline diagnostic resize only"])
    for row in source["files"]:
        if sha256_file(args.dataset / row["file"]) != row["sha256"]:
            raise ValueError(f"dataset changed: {row['file']}")
        with np.load(args.dataset / row["file"]) as data:
            features = pixel_features(data["rgb"])
        path = args.output / row["file"]
        with path.open("xb") as output:
            np.savez_compressed(output, **features)
        manifest["files"].append(dict(game=row["game"], split=row["split"], seed=row["seed"], file=row["file"],
                                       count=len(next(iter(features.values()))), sha256=sha256_file(path), variants=list(features)))
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
