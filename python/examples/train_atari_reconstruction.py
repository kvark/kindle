"""Offline, native-GPU patch-CNN reconstruction control. No RAM/action labels.

Fixed training budget; validation reconstruction chooses the checkpoint. Test
trajectories are never loaded here. Run under the ordinary GPU host guard.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from atari import sha256_file
from fit_atari_probes import checked_memory
from kindle import _native
from kindle._representation_probe import PROTOCOL


def load_frames(dataset, manifest, split, arrivals):
    records = [row for row in manifest["files"] if row["split"] == split]
    # Preallocate so unpacking one compressed recording never doubles the full
    # 1.24GB training array. Four fixed arrivals per clip; no label selection.
    count = len(records) * manifest["clips_per_seed"] * len(arrivals)
    frames = np.empty((count, 210, 160, 3), dtype=np.uint8)
    offset = 0
    for row in records:
        if sha256_file(dataset / row["file"]) != row["sha256"]:
            raise ValueError(f"dataset changed: {row['file']}")
        with np.load(dataset / row["file"]) as data:
            selected = data["rgb"][:, arrivals].reshape(-1, 210, 160, 3)
        frames[offset:offset+len(selected)] = selected
        offset += len(selected)
    return frames


def train(model, training, validation, output, *, steps, seed, batch):
    rng = np.random.default_rng(seed)
    curve, best = [], float("inf")

    def validation_loss():
        if len(validation) % batch:
            raise ValueError("validation must form complete batches")
        return float(np.mean([model.process(list(validation[start:start+batch]))
                              for start in range(0, len(validation), batch)]))

    initial = validation_loss()
    if not np.isfinite(initial):
        raise RuntimeError("nonfinite initial reconstruction")
    model.save(str(output / "initial.safetensors"))
    best, selected_step = initial, 0
    model.save(str(output / "encoder.safetensors"))
    for step in range(1, steps+1):
        indices = rng.integers(len(training), size=batch)
        loss = model.process(list(training[indices]), learning_rate=3e-4)
        if step % 512 == 0 or step == steps:
            score = validation_loss()
            if not np.isfinite(loss) or not np.isfinite(score):
                raise RuntimeError("nonfinite reconstruction objective")
            row = dict(step=step, training_mse=float(loss), validation_mse=score,
                       memory=checked_memory(model))
            curve.append(row)
            print(json.dumps(row), flush=True)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if score < best:
                best, selected_step = score, step
                model.save(str(output / "encoder.safetensors"))
    return dict(initial_validation_mse=initial, selected_validation_mse=best,
                selected_step=selected_step, curve=curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2709)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch <= 0:
        parser.error("positive step/batch budget required")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL or manifest["status"] != "complete":
        parser.error("complete matching corpus required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = load_frames(args.dataset, manifest, "train", [3, 7, 11, 15])
    validation = load_frames(args.dataset, manifest, "validation", [15])
    model = _native.ReconstructionEncoder(batch=args.batch, seed=args.seed)
    memory = checked_memory(model)
    result = dict(protocol=PROTOCOL, architecture="patch_cnn_172864", status="running",
                  seed=args.seed, steps=args.steps, batch=args.batch, learning_rate=3e-4,
                  native_sha256=sha256_file(_native.__file__), device=model.gpu_device, memory=memory,
                  dataset_manifest_sha256=sha256_file(args.dataset / "manifest.json"),
                  training_frames=len(training), validation_frames=len(validation),
                  training_arrivals=[3, 7, 11, 15], validation_arrivals=[15],
                  selection="minimum validation reconstruction MSE, including initialization",
                  objective="ImageNet-normalized RGB reconstruction MSE; Adam .9/.999/1e-8",
                  input="native RGB210x160 -> GPU letterbox224; no intermediate RGB64",
                  labels="none; no RAM, rewards or actions", temporal_history=False)
    path = args.output / "result.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    result.update(train(model, training, validation, args.output, steps=args.steps, seed=args.seed, batch=args.batch))
    result.update(status="complete", seconds=time.monotonic()-started,
                  encoder_sha256=sha256_file(args.output / "encoder.safetensors"))
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
