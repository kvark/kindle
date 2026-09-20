"""Deterministic, pixel-only video batches for the native Tiny learner.

Recordings are uint8 [time, height, width, 3] NPY arrays with uint32 episode
IDs. Clips never cross episode or recording boundaries. All random choices
derive from (seed, step, split); no hidden loader RNG state needs restoring.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FRAMES = 16
STRIDE = 2
PATCH = 16
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
SAMPLER = "tiny-video-token-major-crop-bilinear-imagenet-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Recording:
    def __init__(self, frames, episodes):
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
                or min(frames.shape) < 1 or episodes.dtype != np.uint32
                or episodes.shape != (len(frames),)):
            raise ValueError("invalid RGB recording or episode IDs")
        if np.any(episodes[1:] < episodes[:-1]):
            raise ValueError("episode IDs must be monotonic")
        edges = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, len(episodes)]
        counts = np.diff(edges) - (FRAMES - 1) * STRIDE
        usable = counts > 0
        self.starts = edges[:-1][usable]
        self.cumulative = np.cumsum(counts[usable], dtype=np.int64)
        if not len(self.starts):
            raise ValueError("recording contains no complete within-episode clip")
        self.frames = frames

    def sample(self, rng):
        choice = int(rng.integers(self.cumulative[-1]))
        segment = int(np.searchsorted(self.cumulative, choice, side="right"))
        previous = int(self.cumulative[segment - 1]) if segment else 0
        start = int(self.starts[segment]) + choice - previous
        indices = start + np.arange(FRAMES) * STRIDE
        return np.asarray(self.frames[indices]), start


class Corpus:
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve()
        self.identity = sha256_file(path)
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("format") != 1:
            raise ValueError("unknown video corpus format")
        self.recordings = {"train": [], "validation": []}
        seen = set()
        for record in self.manifest["recordings"]:
            split = record["split"]
            if split not in self.recordings:
                raise ValueError("recordings must use train or validation split")
            arrays = []
            for kind in ("frames", "episodes"):
                filename = record[kind]["file"]
                artifact = (path.parent / filename).resolve()
                if artifact.parent != path.parent:
                    raise ValueError("corpus artifacts must be direct children of its directory")
                actual = sha256_file(artifact)
                if actual != record[kind]["sha256"]:
                    raise ValueError(f"corpus hash mismatch: {filename}")
                if kind == "frames":
                    if actual in seen:
                        raise ValueError("duplicate recording across corpus or splits")
                    seen.add(actual)
                arrays.append(np.load(artifact, mmap_mode="r", allow_pickle=False))
            if len(arrays[0]) != record["observations"]:
                raise ValueError("corpus observation count mismatch")
            self.recordings[split].append(Recording(*arrays))
        if any(not records for records in self.recordings.values()):
            raise ValueError("both whole-recording splits are required")


def crop_box(height, width, scale, rng):
    for _ in range(10):
        area = height * width * rng.uniform(*scale)
        ratio = np.exp(rng.uniform(np.log(0.75), np.log(4 / 3)))
        crop_width, crop_height = round(np.sqrt(area * ratio)), round(np.sqrt(area / ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            x = int(rng.integers(width - crop_width + 1))
            y = int(rng.integers(height - crop_height + 1))
            return x, y, x + crop_width, y + crop_height
    side = min(height, width)
    x, y = (width - side) // 2, (height - side) // 2
    return x, y, x + side, y + side


def retained_patches(clip, size, box, ids):
    """Resize in uint8, gather retained patches, then normalize only those pixels."""
    if (clip.dtype != np.uint8 or clip.shape[0] != FRAMES or clip.ndim != 4
            or clip.shape[-1] != 3 or size not in (96, 224)):
        raise ValueError("invalid video crop")
    grid = size // PATCH
    if (ids.ndim != 1 or ids.dtype != np.uint32 or len(np.unique(ids)) != len(ids)
            or np.any(ids >= FRAMES * grid * grid)):
        raise ValueError("invalid retained token IDs")
    pixels = np.stack([np.asarray(Image.fromarray(frame).crop(box).resize(
        (size, size), Image.Resampling.BILINEAR)) for frame in clip])
    patches = pixels.reshape(FRAMES, grid, PATCH, grid, PATCH, 3).transpose(0, 1, 3, 5, 2, 4)
    selected = patches[ids // (grid * grid), (ids // grid) % grid, ids % grid].astype(np.float32)
    selected *= np.float32(1 / 255)
    selected -= MEAN
    selected /= STD
    return selected.reshape(len(ids), 768)


def batch(corpus, model, seed, step, split="train"):
    if split not in ("train", "validation") or seed < 0 or step < 0:
        raise ValueError("invalid sampler seed, step or split")
    count, local_views = model["batch"], model["local_views"]
    if (not 2 <= count <= 128 or not 1 <= local_views <= 4
            or not 1 <= model["projector_output"] <= 256
            or not 1 <= model["directions"] <= 1024):
        raise ValueError("unsupported pretraining dimensions")
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, int(split == "validation")]))
    result = {
        "global_patches": np.empty((157, count, 768), dtype=np.float32),
        "global_ids": np.empty((157, count), dtype=np.uint32),
        "local_patches": np.empty((29, count * local_views, 768), dtype=np.float32),
        "local_ids": np.empty((29, count * local_views), dtype=np.uint32),
    }
    examples = []
    records = corpus.recordings[split]
    for clip_index in range(count):
        record_index = int(rng.integers(len(records)))
        clip, start = records[record_index].sample(rng)
        examples.append((record_index, start))
        for view in range(local_views + 1):
            global_view = view == 0
            name, size, kept, scale = ("global", 224, 157, (0.4, 1.0)) if global_view else (
                "local", 96, 29, (0.1, 0.4))
            column = clip_index if global_view else (view - 1) * count + clip_index
            box = crop_box(clip.shape[1], clip.shape[2], scale, rng)
            ids = rng.choice(FRAMES * (size // PATCH) ** 2, kept, replace=False).astype(np.uint32)
            result[name + "_patches"][:, column] = retained_patches(clip, size, box, ids)
            result[name + "_ids"][:, column] = ids
    directions = rng.standard_normal((model["projector_output"], model["directions"]), dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    result["directions"] = directions
    return result, examples
