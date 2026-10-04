#!/usr/bin/env python3
"""Check four native fixed-batch learner steps against pinned upstream Agent.loss.

The native ignored test exports actual production posterior/imagination/targets,
gradients and checkpoints. This verifier uses common initial weights and draws,
checks raw model gradients at each native pre-step checkpoint, then supplies
identical raw gradients to upstream's own optimizer and EMA. The optimizer runs
continuously; it does not resynchronize its weights or moments. This separates
gradient arithmetic from optimizer roundoff and LaProp sign normalization.
Scan unrolling, random-number supply and the documented CDP construction fixes
adapt the model reference; loss math is upstream.
Replay sampling, environment collection and gameplay competence are out of scope.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from dreamer_rgb_reference import REVISION, native_parameter

CDP_REVISION = "a851fa3e3d70b624b094ee1810ad4bb602346092"


def parameter_layout(name, value):
    """Map upstream leaves into native checkpoint names and storage."""
    import numpy as np

    if name.startswith(("enc/", "dec/")):
        return native_parameter(name, value)
    parts = name.split("/")
    module, leaf = parts[0], parts[-1]
    if module == "dyn":
        layer = parts[1]
        if layer.startswith("pred"):
            prefix = "world.future_predictor"
            suffix = ("layer0.norm.weight" if layer == "pred0norm" else
                      f"{'out' if layer == 'pred_out' else 'layer0'}.{'weight' if leaf == 'kernel' else 'bias'}")
            return f"{prefix}.{suffix}", np.asarray(value, np.float32).reshape(-1)
        prefix = ("world.dynamics.core" if layer.startswith("dyn") else
                  "world.representation.posterior" if layer.startswith("obs") else "world.dynamics")
        if leaf == "scale":
            assert layer.endswith("norm"), name
            suffix = layer[:-4] + ".norm.weight"
        else:
            suffix = layer + (".weight" if leaf == "kernel" else ".bias")
    else:
        prefix = {"rew": "world.reward", "con": "world.continuation", "pol": "behavior.actor",
                  "val": "behavior.value", "slowval": "behavior.value"}[module]
        if parts[1] == "mlp":
            layer = parts[2]
            if layer.startswith("norm"):
                assert leaf == "scale", name
                suffix = f"layer{layer[4:]}.norm.weight"
            else:
                assert layer.startswith("linear"), name
                suffix = f"layer{layer[6:]}.{'weight' if leaf == 'kernel' else 'bias'}"
        else:
            assert parts[1] == "head" and leaf in {"kernel", "bias"}, name
            suffix = "out.weight" if leaf == "kernel" else "out.bias"
    return f"{prefix}.{suffix}", np.asarray(value, np.float32).reshape(-1)


def upstream_parameter(name, value, shape):
    """Undo the native OIHW convolution storage; other leaves are unchanged."""
    import numpy as np

    value = np.asarray(value, np.float32)
    if name.startswith(("enc/", "dec/")) and name.endswith("/kernel") and len(shape) == 4:
        height, width, inputs, outputs = shape
        return value.reshape(outputs, inputs, height, width).transpose(2, 3, 1, 0)
    return value.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--cdp", action="store_true", help="use pinned CDP reference; omit its detached visualization decoder")
    parser.add_argument("--output", type=Path, help="fresh result directory; native fixture remains read-only")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    expected_revision = CDP_REVISION if args.cdp else REVISION
    if revision != expected_revision:
        raise ValueError(f"expected upstream {expected_revision}, found {revision}")
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code", "HEAD", "--",
                    "dreamerv3/rssm.py", "dreamerv3/agent.py", "embodied/jax"], check=True)
    sys.path.insert(0, str(args.upstream))
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    import optax
    import ruamel.yaml
    from safetensors.numpy import load_file
    from dreamerv3 import agent, rssm
    from embodied.jax import nets, outs
    from upstream_matched import GpuBudget

    native_config = json.loads((args.root / "config.json").read_text())
    assert native_config["model_size"] == "size1_m" or native_config["model_size"] == "size1m"
    assert native_config["observation_kind"] == "rgb64"
    assert native_config["actor_unimix"] == 0
    batch, length, horizon = 2, 4, 3
    assert (native_config["batch_size"], native_config["batch_length"], native_config["imagination_length"]) == (batch, length, horizon)
    output = args.output or args.root
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    budget = GpuBudget(output / "upstream-gpu-memory.jsonl")
    budget.check("before_jax")
    jax.config.update("jax_default_matmul_precision", "highest")
    nets.COMPUTE_DTYPE = jnp.float32
    device = jax.devices()[0]
    if device.platform != "gpu" or device.device_kind != "NVIDIA GeForce RTX 5080":
        raise ValueError(f"unexpected JAX device: {device}")
    source = subprocess.check_output(["git", "-C", str(args.upstream), "show", f"{revision}:dreamerv3/configs.yaml"], text=True)
    configs = ruamel.yaml.YAML(typ="safe").load(source)
    config = elements.Config(configs["defaults"]).update(configs["size1m"]).agent.update({
        "imag_length": horizon, "ac_grads": native_config.get("actor_critic_gradient", False)})
    obs_space = {"image": elements.Space(np.uint8, (64, 64, 3)),
                 "reward": elements.Space(np.float32),
                 **{key: elements.Space(bool) for key in ("is_first", "is_last", "is_terminal")}}
    act_space = {"action": elements.Space(np.int32, (), 0, 18)}
    # Bypass only the device/sharding/runner wrapper, not the agent definition.
    model = object.__new__(agent.Agent)
    if args.cdp:
        # Upstream passes the outer {typ, simple} config to a helper expecting
        # {depth, mults}; its default depth64 accidentally only fits Size200M.
        calculate_width = rssm.Encoder.calculate_encoder_output_dim
        dimension_fix = patch.object(rssm.Encoder, 'calculate_encoder_output_dim',
            lambda self, spaces, cfg: calculate_width(self, spaces, cfg[cfg.typ]))
    else:
        dimension_fix = nullcontext()
    with dimension_fix:
        agent.Agent.__init__(model, obs_space, act_space, config)
    if args.cdp:
        assert native_config['loss_scales']['reconstruction'] == 0
        assert native_config['loss_scales']['future_prediction'] == 500
        assert native_config['encoder_learning_rate'] == config.enc_lr
        assert native_config['dynamics_learning_rate'] == config.dyn_lr
        assert model.enc_output_dim == 256
        # The released decoder is detached from the world/encoder and used
        # only for visualization. Excluding it cannot change their gradients.
        model.dec = lambda carry, *a, **kw: (carry, {}, {})
        model.scales.pop('image')
    current = {}
    observe = rssm.RSSM.observe
    imagine = rssm.RSSM.imagine

    def sequence_observe(self, carry, tokens, actions, reset, training, single=False):
        if single:
            return observe(self, carry, tokens, actions, reset, training, single=True)
        outputs = []
        for time in range(length):
            current.update(phase="posterior", time=time)
            carry, *parts = observe(self, carry, tokens[:, time],
                jax.tree.map(lambda x: x[:, time], actions), reset[:, time], training, single=True)
            outputs.append(tuple(parts))
        stack = lambda *xs: jnp.stack(xs, 1)
        return (carry, *jax.tree.map(stack, *outputs))

    def sequence_imagine(self, carry, policy, steps, training, single=False):
        if single:
            return imagine(self, carry, policy, steps, training, single=True)
        features, actions = [], []
        for time in range(steps):
            current.update(phase="imagination", time=time)
            carry, (feature, action) = imagine(self, carry, policy, 1, training, single=True)
            features.append(feature)
            actions.append(action)
        stack = lambda *xs: jnp.stack(xs, 1)
        return carry, jax.tree.map(stack, *features), jax.tree.map(stack, *actions)

    def categorical_sample(self, seed, shape=()):
        del seed
        assert not shape
        time, data = current["time"], current["data"]
        if current["phase"] == "posterior":
            uniforms = data["posterior_uniforms"][time].reshape(batch, 32, 4)
        else:
            key = "action_uniforms" if self.logits.shape[-1] == 18 else "latent_uniforms"
            # Native starts are time-major; upstream starts are batch-major.
            uniforms = data[key][time].reshape(length, batch, -1).swapaxes(0, 1).reshape(self.logits.shape)
        uniforms = jnp.clip(uniforms, jnp.finfo(jnp.float32).tiny, 1 - jnp.finfo(jnp.float32).eps)
        return jnp.argmax(self.logits - jnp.log(-jnp.log(uniforms)), -1)

    def read_batch(step):
        data = json.loads((args.root / f"step{step}/batch.json").read_text())
        values = {k: jnp.asarray(v, jnp.float32) for k, v in data.items() if k != "flags"}
        for key in ("is_first", "is_last", "is_terminal"):
            values[key] = jnp.asarray([[f[key] for f in row] for row in data["flags"]], bool).T
        return values

    def objective(data):
        current["data"] = data
        try:
            pixels = data["observations"].reshape(length, batch, 3, 64, 64).transpose(1, 0, 3, 4, 2)
            image = jnp.rint((pixels + 0.5) * 255).astype(jnp.uint8)
            obs = dict(image=image, reward=data["rewards"].T,
                       **{k: data[k] for k in ("is_first", "is_last", "is_terminal")})
            actions = {"action": data["actions"].reshape(length, batch, 18).argmax(-1).T.astype(jnp.int32)}
            carry = ({}, dict(deter=data["initial_deter"].reshape(batch, 512),
                              stoch=data["initial_stoch"].reshape(batch, 32, 4)), {})
            loss, auxiliary = model.loss(carry, obs, actions, training=True)
            _, _, output, metrics, *_ = auxiliary
            return loss, (output["repfeat"], metrics)
        finally:
            current.clear()

    pure = nj.pure(objective)
    parameter_prefixes = ("enc/", "dyn/", "dec/", "rew/", "con/", "pol/", "val/")
    reports = []

    def compare(name, actual, expected, atol=2e-4, rtol=2e-3, gradient=False):
        actual, expected = np.asarray(actual, np.float64).reshape(-1), np.asarray(expected, np.float64).reshape(-1)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"{name}: nonfinite values or mismatched shapes")
        error = np.abs(actual - expected)
        relative = float(np.linalg.norm(error) / max(1e-30, np.linalg.norm(expected)))
        record = dict(name=name, maximum=float(error.max(initial=0)), relative_l2=relative)
        reports.append(record)
        if np.any(error > atol + rtol * np.abs(expected)):
            worst = int(np.argmax(error / (atol + rtol * np.abs(expected) + 1e-30)))
            record.update(worst_index=worst, actual=float(actual[worst]), expected=float(expected[worst]),
                          expected_rms=float(np.sqrt(np.mean(expected ** 2))))
            np.savez_compressed(output / 'failed-comparison.npz', actual=actual, expected=expected)
            raise AssertionError(record)
        if gradient and np.linalg.norm(error) > rtol * np.linalg.norm(expected) + 1e-7 * np.sqrt(len(actual)):
            raise AssertionError(record)

    def native_checkpoint(path):
        world = load_file(path / "world.safetensors")
        behavior = load_file(path / "behavior.safetensors")
        # The world's critic is a frozen copy, owned/updated by behavior.
        return {**world, **behavior}

    try:
        with patch.object(rssm.RSSM, "observe", sequence_observe), patch.object(rssm.RSSM, "imagine", sequence_imagine), patch.object(outs.Categorical, "sample", categorical_sample):
            data = read_batch(0)
            def initialize(data):
                result = objective(data)
                model.slowval.count.read()
                return result
            state = nj.init(nj.pure(initialize))({}, data, seed=103)
            initial = native_checkpoint(args.root / "initial")
            initial_slow = load_file(args.root / "initial/slow.safetensors")
            mapped = set()
            for name, value in state.items():
                if name.startswith(parameter_prefixes + ("slowval/",)):
                    native_name, _ = parameter_layout(name, value)
                    source = initial_slow if name.startswith("slowval/") else initial
                    state[name] = jnp.asarray(upstream_parameter(name, source[native_name], value.shape))
                    if name.startswith(parameter_prefixes):
                        mapped.add(native_name)
                else:
                    state[name] = jnp.zeros_like(value)
            native_names = {k for k in initial if k.startswith(("world.", "behavior."))}
            assert mapped == native_names, (mapped ^ native_names)
            params = {k: v for k, v in state.items() if k.startswith(parameter_prefixes)}
            optimizer = model._make_opt(lr=native_config["learning_rate"], warmup=native_config["learning_rate_warmup"])
            labels = {name: 'enc' if name.startswith('enc/') else 'dyn' if name.startswith('dyn/') else 'other'
                      for name in params}
            if args.cdp:
                optimizer = optax.multi_transform({
                    group: model._make_opt(lr=rate, warmup=native_config['learning_rate_warmup'])
                    for group, rate in [('enc', config.enc_lr), ('dyn', config.dyn_lr),
                                        ('other', native_config['learning_rate'])]}, labels)
            optimizer_state = optimizer.init(params)

            def loss_fn(params, extras, data):
                updated, (loss, aux) = pure({**extras, **params}, data, seed=103)
                return loss, (updated, aux)

            evaluate = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
            update = jax.jit(optimizer.update)
            ema = jax.jit(nj.pure(model.slowval.update))
            for step in range(4):
                data = read_batch(step)
                path = args.root / f"step{step}"
                native = json.loads((path / "outputs.json").read_text())
                extras = {k: v for k, v in state.items() if not k.startswith(parameter_prefixes)}
                before = native_checkpoint(args.root / (f'step{step - 1}' if step else 'initial'))
                gradient_params = {name: jnp.asarray(upstream_parameter(name,
                    before[parameter_layout(name, value)[0]], value.shape)) for name, value in params.items()}
                (_, (state, (feat, metrics))), grads = evaluate(gradient_params, extras, data)
                for key in ("deter", "stoch"):
                    compare(f"step{step}/{key}", native[key], np.asarray(feat[key]).swapaxes(0, 1),
                            atol=0 if key == "stoch" else 2e-4, rtol=0 if key == "stoch" else 2e-3)
                for key, group, field in [("image", "world", "reconstruction_loss"), ("dyn", "world", "dynamics_kl"),
                                          ("rep", "world", "representation_kl"), ("rew", "world", "reward_loss"),
                                          ("con", "world", "continuation_loss"), ("repval", "world", "replay_value_loss"),
                                          ("policy", "behavior", "policy_loss"), ("value", "behavior", "value_loss")]:
                    if args.cdp and key == 'image':
                        key, field = 'dyn_deter', 'future_prediction_loss'
                    compare(f"step{step}/loss/{key}", native[group][field], metrics[f"loss/{key}"])
                native_grads = json.loads((path / "gradients.json").read_text())
                common_grads = {}
                for name, value in grads.items():
                    native_name, expected = parameter_layout(name, value)
                    compare(f"step{step}/gradient/{name}", native_grads[native_name], expected, atol=3e-3, rtol=3e-3, gradient=True)
                    common_grads[name] = jnp.asarray(upstream_parameter(name, native_grads[native_name], value.shape))
                updates, optimizer_state = update(common_grads, optimizer_state, params)
                params = optax.apply_updates(params, updates)
                state.update(params)
                state, _ = ema(state)
                after = native_checkpoint(path)
                def moments(index):
                    if not args.cdp:
                        return optimizer_state[index][1]
                    return {name: optimizer_state.inner_states[labels[name]].inner_state[index][1][name]
                            for name in params}
                for kind, values, prefix in [("parameter", params, ""), ("momentum", moments(2), "adam_m."),
                                             ("variance", moments(1), "adam_v.")]:
                    for name, value in values.items():
                        native_name, expected = parameter_layout(name, value)
                        compare(f"step{step}/{kind}/{name}", after[prefix + native_name], expected,
                                atol=3e-6 if kind != "momentum" else 3e-3, rtol=3e-3)
                slow = load_file(path / "slow.safetensors")
                for name, value in state.items():
                    if name.startswith("slowval/"):
                        native_name, expected = parameter_layout(name, value)
                        compare(f"step{step}/ema/{name}", slow[native_name], expected, atol=3e-6, rtol=3e-5)
                budget.check(f"step{step}")
        result = dict(status="passed", upstream=revision, updates=4, comparisons=reports,
                      cdp=args.cdp, detached_visualization_decoder_omitted=args.cdp,
                      cdp_encoder_config_nesting_fixed=args.cdp,
                      actor_critic_gradient=native_config.get("actor_critic_gradient", False),
                      limits=["fixed synthetic batches; not replay sampling or gameplay competence",
                              "raw gradients use exact native pre-step weights; optimizer weights/moments remain independent",
                              "raw gradients compared first; optimizer receives identical gradients to isolate its math"])
    except Exception as error:
        result = dict(status="failed", upstream=revision, error=str(error), comparisons=reports)
        raise
    finally:
        budget.close()
        with (output / "upstream-result.json").open("x") as stream:
            json.dump(result, stream, allow_nan=False)
    print(json.dumps(dict(status="passed", comparisons=len(reports), updates=4)))


if __name__ == "__main__":
    main()
