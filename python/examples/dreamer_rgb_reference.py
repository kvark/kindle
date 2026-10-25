#!/usr/bin/env python3
"""Small RGB value/gradient/LaProp fixture using the pinned upstream modules.

Run under the GPU host guard, then run the native ignored RGB reference test
and dreamer_optimizer_reference.py on its exported raw gradients/checkpoints.
This is four synthetic updates, not a gameplay training run or full-agent parity.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"


def native_parameter(name, value):
    """Only parameter naming/storage changes; upstream computes all references."""
    import numpy as np

    module, layer, leaf = name.split("/")
    prefix = {"enc": "world.representation.encoder", "dec": "world.decoder"}[module]
    if layer.endswith("norm"):
        assert leaf == "scale", name
        layer = "spatial" if layer == "spnorm" else layer[:-4]
        suffix = f"{layer}.norm.weight"
    else:
        assert leaf in {"kernel", "bias"}, name
        suffix = f"{layer}.{'weight' if leaf == 'kernel' else 'bias'}"
    value = np.asarray(value, dtype=np.float32)
    if leaf == "kernel" and (re.fullmatch(r"cnn\d|conv\d", layer) or layer == "imgout"):
        value = value.transpose(3, 2, 0, 1)
    return f"{prefix}.{suffix}", np.ascontiguousarray(value).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"expected upstream {REVISION}, found {revision}")
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code", "HEAD", "--",
                    "dreamerv3/rssm.py", "dreamerv3/agent.py", "embodied/jax/nets.py", "embodied/jax/opt.py"], check=True)
    sys.path.insert(0, str(args.upstream))
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import optax
    import ruamel.yaml
    from safetensors.numpy import save_file
    from dreamerv3 import agent, rssm
    from embodied.jax import nets

    from upstream_matched import GpuBudget

    args.output.mkdir(parents=True, exist_ok=False)
    budget = GpuBudget(args.output / "gpu-memory.jsonl")
    budget.check("before_jax")
    jax.config.update("jax_default_matmul_precision", "highest")
    nets.COMPUTE_DTYPE = jnp.float32
    device = jax.devices()[0]
    if device.platform != "gpu" or device.device_kind != "NVIDIA GeForce RTX 5080":
        raise ValueError(f"unexpected device: {device}")
    source = subprocess.check_output(["git", "-C", str(args.upstream), "show", f"{REVISION}:dreamerv3/configs.yaml"], text=True)
    configs = ruamel.yaml.YAML(typ="safe").load(source)
    config = elements.Config(configs["defaults"]).update(configs["size1m"]).agent
    space = {"image": elements.Space(np.uint8, (64, 64, 3))}
    encoder = rssm.Encoder(space, **config.enc.simple, name="enc")
    decoder = rssm.Decoder(space, **config.dec.simple, name="dec")
    rng = np.random.default_rng(103)
    reset = jnp.zeros((2, 1), bool)

    def batch():
        pixels = rng.integers(0, 256, (2, 1, 64, 64, 3), dtype=np.uint8)
        deter = rng.uniform(-0.3, 0.3, (2, 1, 512)).astype(np.float32)
        stoch = np.eye(4, dtype=np.float32)[rng.integers(0, 4, (2, 1, 32))]
        return pixels, {"deter": deter, "stoch": stoch}

    def objective(pixels, features):
        _, _, tokens = encoder({}, {"image": pixels}, reset, training=True)
        _, _, decoded = decoder({}, features, reset, training=True)
        reconstruction = decoded["image"]
        loss = reconstruction.loss(jnp.float32(pixels) / 255).mean() + 0.01 * jnp.square(tokens).mean()
        return loss, (tokens, reconstruction.pred())

    pure = nj.pure(objective)
    pixels, features = batch()
    params = nj.init(pure)({}, pixels, features, seed=103)
    loss_and_grad = jax.jit(jax.value_and_grad(lambda p, x, f: pure(p, x, f, seed=103)[1], has_aux=True))
    optimizer = agent.Agent._make_opt(None, lr=4e-5, warmup=2)
    state = optimizer.init(params)
    update = jax.jit(optimizer.update)
    tensors = {}

    def save_parameters(prefix, values):
        for name, value in values.items():
            native_name, array = native_parameter(name, value)
            tensors[f"{prefix}/{native_name}"] = array

    save_parameters("initial", params)
    for step in range(4):
        if step:
            pixels, features = batch()
        (loss, (encoded, decoded)), gradient = loss_and_grad(params, pixels, features)
        updates, state = update(gradient, state, params)
        params = optax.apply_updates(params, updates)
        prefix = f"step{step}"
        centered = pixels.astype(np.float32) / 255 - 0.5
        tensors[f"{prefix}/pixels"] = centered[:, 0].transpose(0, 3, 1, 2).copy()
        tensors[f"{prefix}/features"] = np.concatenate([features["deter"].reshape(2, -1), features["stoch"].reshape(2, -1)], -1)
        tensors[f"{prefix}/loss"] = np.asarray([loss], np.float32)
        tensors[f"{prefix}/encoded"] = np.asarray(encoded).reshape(2, -1)
        tensors[f"{prefix}/decoded"] = (np.asarray(decoded)[:, 0].transpose(0, 3, 1, 2) - 0.5).copy()
        save_parameters(f"{prefix}/gradient", gradient)
        save_parameters(f"{prefix}/parameter", params)
        save_parameters(f"{prefix}/momentum", state[2][1])
        save_parameters(f"{prefix}/variance", state[1][1])
        budget.check(prefix)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, args.output / "reference.safetensors")
    result = dict(upstream=revision, device=device.device_kind, preset="size1m", dtype="float32",
                  batch=2, updates=4, warmup=2, learning_rate=4e-5, agc=0.3,
                  parameters={k: int(v.size) for k, v in tensors.items() if k.startswith("initial/")},
                  limits=["synthetic RGB pair only; not RSSM, behavior, EMA or gameplay parity"])
    (args.output / "manifest.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    budget.close()
    print(json.dumps(dict(status="complete", parameters=sum(result["parameters"].values()), updates=4)))


if __name__ == "__main__":
    main()
