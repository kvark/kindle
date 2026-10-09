"""Native batched frozen-encoder diagnostics on an already recorded corpus.

Phase 15 consumes all sixteen causal arrivals; phase 0 re-encodes the SAME final
frame after reset. PCA fits only native phase-15 training tokens, once per model,
and is then held fixed for both input sizes, both phases and every game/split.
Diagnostic readbacks/projections are analysis, not a change to the acting path.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from atari import sha256_file
from kindle import _native
from kindle._representation_probe import PROTOCOL, fit_pca, fixed_projection, spatial_features


def memory(encoder):
    value = encoder.gpu_memory_budget
    if value is None or value["budget_bytes"] - value["usage_bytes"] < 2 << 30:
        raise RuntimeError(f"less than 2GiB Vulkan estimated headroom: {value}")
    return value


def token_batch(encoder, clips, input_size, *, phase, parity):
    streams = list(range(len(clips)))
    if getattr(encoder, "architecture", None) == "cnn":
        frames = [clip[-1] for clip in clips]
        if input_size == "rgb64":
            frames = [np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR)) for frame in frames]
        encoder.process(frames + [frames[-1]] * (encoder.patch_token_shape[0] - len(frames)))
        tokens = np.frombuffer(encoder.patch_tokens(), dtype="<f4").reshape(encoder.patch_token_shape)[:len(clips)]
        if not np.isfinite(tokens).all():
            raise RuntimeError("nonfinite CNN tokens")
        return tokens
    for arrival in (range(16) if phase == 15 else (15,)):
        frames = [clip[arrival] for clip in clips]
        if input_size == "rgb64":
            frames = [np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR)) for frame in frames]
        pooled = encoder.encode_batch(streams, frames, [arrival == 0 or phase == 0] * len(clips))
    tokens = np.frombuffer(encoder.patch_tokens(), dtype="<f4").reshape(encoder.patch_token_shape)[:len(clips)]
    if not np.isfinite(tokens).all():
        raise RuntimeError("nonfinite patch tokens")
    # Exercise the native -> Python tensor layout and exact production JL/pool
    # path, not just two implementations of the alternative projections.
    expected = spatial_features(tokens, fixed_projection(tokens.shape[-1]), pooling="mean").reshape(len(clips), -1)
    actual = np.asarray(pooled, dtype=np.float32)
    error = float(np.linalg.norm(expected.astype(np.float64) - actual) / max(np.linalg.norm(actual), 1e-12))
    parity.append(error)
    if error > 2e-5:
        raise RuntimeError(f"native production projection/pool differs: relative L2 {error}")
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--architecture", choices=("tiny", "large", "cnn"), default="tiny")
    parser.add_argument("--num-streams", type=int, default=6)
    parser.add_argument("--limit-clips", type=int, help="explicit smoke-only cap per recording")
    parser.add_argument("--plan-cache", type=Path)
    args = parser.parse_args()
    if args.num_streams <= 0 or (args.limit_clips is not None and args.limit_clips <= 0):
        parser.error("stream count and optional clip limit must be positive")
    source = json.loads((args.dataset / "manifest.json").read_text())
    if source["protocol"] != PROTOCOL or source["status"] != "complete":
        parser.error("dataset must be a completed matching corpus")
    for row in source["files"]:
        if sha256_file(args.dataset / row["file"]) != row["sha256"]:
            raise ValueError(f"dataset bytes changed: {row['file']}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.architecture == "cnn":
        encoder = _native.ReconstructionEncoder(batch=args.num_streams)
        encoder.load(str(args.encoder))
    else:
        encoder = _native.LeVJepaPerception(str(args.encoder), str(args.plan_cache) if args.plan_cache else None,
                                            architecture=args.architecture, num_streams=args.num_streams)
    device = encoder.gpu_device
    if device["device_name"] != "NVIDIA GeForce RTX 5080" or device["driver_info"] != "580.178.04" or device["is_software_emulated"]:
        raise RuntimeError(f"unexpected native device: {device}")
    manifest = dict(protocol=PROTOCOL, status="running", architecture=args.architecture,
                    encoder_sha256=sha256_file(args.encoder), native_sha256=sha256_file(_native.__file__),
                    dataset_manifest_sha256=sha256_file(args.dataset / "manifest.json"),
                    limit_clips=args.limit_clips, num_streams=args.num_streams, device=device,
                    construction_seconds=time.monotonic() - started, memory=[memory(encoder)], files=[])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parity, pca_samples, pca_sources = [], [], []
    # Equal-game sample from each game's first TRAIN recording. At most 64
    # clip endpoints per game, 32768 patch tokens total, independent of labels.
    for game in source["games"]:
        row = next(r for r in source["files"] if r["game"] == game and r["split"] == "train")
        with np.load(args.dataset / row["file"]) as data:
            clips = data["rgb"][::4][:64]
        if args.limit_clips:
            clips = clips[:args.limit_clips]
        pca_sources.append(dict(file=row["file"], indices=list(range(0, 4*len(clips), 4))))
        for start in range(0, len(clips), args.num_streams):
            tokens = token_batch(encoder, clips[start:start+args.num_streams], "native", phase=15, parity=parity)
            pca_samples.append(tokens.reshape(-1, tokens.shape[-1]).copy())
        manifest["memory"].append(memory(encoder))
    samples = np.concatenate(pca_samples)
    indices = np.random.default_rng(2709).permutation(len(samples))[:32768]
    center, pca, eigenvalues = fit_pca(samples[indices])
    del pca_samples, samples
    random64 = fixed_projection(pca.shape[0])
    # Reuse the same random directions, with the JL16 normalization, so this
    # contrast changes spatial packing rather than the random draw assignment.
    random16 = random64[:, :16] * 2
    with (args.output / "projection.npz").open("xb") as stream:
        np.savez(stream, center=center, pca=pca, eigenvalues=eigenvalues, random64=random64, random16=random16)
    manifest["pca"] = dict(sources=pca_sources, token_count=len(indices), token_sample_seed=2709,
                           input="native", phase=15, sha256=sha256_file(args.output / "projection.npz"))
    print(json.dumps(dict(event="projection_ready", seconds=time.monotonic() - started)), flush=True)
    for row in source["files"]:
        with np.load(args.dataset / row["file"]) as data:
            clips = data["rgb"]
        if args.limit_clips:
            clips = clips[:args.limit_clips]
        features = {}
        for input_size in ("native", "rgb64"):
            for phase in (15, 0):
                for start in range(0, len(clips), args.num_streams):
                    tokens = token_batch(encoder, clips[start:start+args.num_streams], input_size, phase=phase, parity=parity)
                    for projection, centered, matrix64, matrix16 in (
                        ("random", tokens, random64, random16),
                        ("pca", tokens - center, pca, pca[:, :16]),
                    ):
                        for pooling, matrix in (("mean", matrix64), ("space_to_depth", matrix16)):
                            key = f"{input_size}/phase{phase}/{projection}/{pooling}"
                            if key not in features:
                                features[key] = np.empty((len(clips), 7, 7, 64), dtype=np.float32)
                            features[key][start:start+len(tokens)] = spatial_features(tokens, matrix, pooling=pooling)
                manifest["memory"].append(memory(encoder))
        path = args.output / row["file"]
        with path.open("xb") as stream:
            np.savez_compressed(stream, **features)
        record = {k: row[k] for k in ("game", "split", "seed", "file")}
        record.update(sha256=sha256_file(path), count=len(clips), variants=list(features))
        manifest["files"].append(record)
        print(json.dumps(dict(event="recording_complete", **record, seconds=time.monotonic() - started)), flush=True)
    manifest.update(status="complete", seconds=time.monotonic() - started,
                    native_projection_max_relative_l2=max(parity, default=None))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
