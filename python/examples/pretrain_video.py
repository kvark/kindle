"""Video-only Tiny pretraining through Rust/Meganeura/Blade.

Run the direct Python process under gpu_host_guard.py after numerical
qualification. Configuration and corpus are explicit; output is exclusive.
Restore uses a new output directory and the same budget, corpus and sampler.
No automatic successor, evaluation control or gameplay claim is produced.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import PIL

from kindle import _video_pretrain as video


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--expected-device", required=True)
    parser.add_argument("--expected-driver", required=True)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        parser.error("checkpoint interval must be positive")
    config = json.loads(args.config.read_text())
    corpus = video.Corpus(args.corpus)
    # Importing the extension does not create a GPU context. The constructor
    # validates the complete native configuration before initializing one.
    from kindle import _native
    identity = {
        "format": 1, "config": config, "corpus_sha256": corpus.identity,
        "sampler": video.SAMPLER, "sampler_sha256": video.sha256_file(video.__file__),
        "runner_sha256": video.sha256_file(__file__),
        "native_sha256": video.sha256_file(_native.__file__),
        "numpy": np.__version__, "pillow": PIL.__version__,
        "kindle": importlib.metadata.version("kindle"),
        "expected_device": args.expected_device, "expected_driver": args.expected_driver,
    }
    if args.restore:
        retained = json.loads((args.restore / "adapter.json").read_text())
        if retained != identity:
            raise ValueError("restore corpus, sampler, package or configuration differs")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "declaration.json", dict(
        identity, corpus=str(args.corpus.resolve()),
        restore=str(args.restore.resolve()) if args.restore else None,
        checkpoint_every=args.checkpoint_every, declared_unix_time=time.time()))
    trainer = (_native.LeVJepaTrainer.restore(str(args.restore)) if args.restore else
               _native.LeVJepaTrainer(json.dumps(config, allow_nan=False)))
    device = trainer.gpu_device
    if (device["device_name"] != args.expected_device
            or device["driver_info"] != args.expected_driver
            or device["is_software_emulated"]):
        raise RuntimeError("pretraining device or driver differs from declared fixture")
    starting_step = trainer.completed_steps
    if starting_step >= config["steps"]:
        raise ValueError("restored pretraining budget is already complete")
    write_json(args.output / "native.json", dict(config=trainer.config, device=device,
                                               starting_step=starting_step))

    def checkpoint():
        destination = args.output / f"step{trainer.completed_steps:06d}"
        trainer.save(str(destination))
        write_json(destination / "adapter.json", identity)
        return destination

    if not args.restore:
        checkpoint()
        trainer.export_encoder(str(args.output / "untrained.safetensors"))
    started = time.monotonic()
    with (args.output / "steps.jsonl").open("x") as stream:
        while trainer.completed_steps < config["steps"]:
            step = trainer.completed_steps
            before_batch = time.monotonic()
            arrays, examples = video.batch(corpus, config["model"], config["seed"], step)
            before_native = time.monotonic()
            metrics = trainer.step(**arrays)
            after_native = time.monotonic()
            if metrics["step"] != step + 1 or trainer.completed_steps != step + 1:
                raise RuntimeError("native pretraining step did not advance exactly once")
            record = dict(metrics, batch_seconds=before_native - before_batch,
                          native_seconds=after_native - before_native,
                          elapsed_seconds=after_native - started, examples=examples)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if trainer.completed_steps % args.checkpoint_every == 0:
                checkpoint()
    if trainer.completed_steps % args.checkpoint_every:
        checkpoint()
    encoder = args.output / "encoder.safetensors"
    trainer.export_encoder(str(encoder))
    write_json(args.output / "result.json", dict(
        complete=True, starting_step=starting_step, completed_steps=trainer.completed_steps,
        encoder_sha256=video.sha256_file(encoder), elapsed_seconds=time.monotonic() - started,
        corpus_sha256=corpus.identity, gameplay_evaluation=False, automatic_successor=False))


if __name__ == "__main__":
    main()
