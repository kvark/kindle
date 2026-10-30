"""Compare the pinned upstream optimizer with native updates on identical gradients.

The separate RGB check compares model values and raw gradients. Reusing those
exact native gradients here avoids magnifying near-zero cross-backend rounding
through LaProp's first-step sign normalization. No gameplay or model training.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dreamer_rgb_reference import REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == REVISION
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code", "HEAD", "--",
                    "dreamerv3/agent.py", "embodied/jax/opt.py"], check=True)
    sys.path.insert(0, str(args.upstream))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from safetensors.numpy import load_file
    from dreamerv3.agent import Agent
    from upstream_matched import GpuBudget

    args.output.mkdir(parents=True, exist_ok=False)
    budget = GpuBudget(args.output / "gpu-memory.jsonl")
    result = dict(status="failed", upstream=REVISION, updates=4, comparisons=[],
                  limits=["optimizer only, with identical raw gradients; model values/gradients checked separately"])
    try:
        budget.check("before_jax")
        devices = jax.devices()
        assert len(devices) == 1 and devices[0].platform == "gpu" and devices[0].device_kind == "NVIDIA GeForce RTX 5080", devices
        tensors = load_file(args.reference / "reference.safetensors")
        params = {k.removeprefix("initial/"): jnp.asarray(v.reshape(-1)) for k, v in tensors.items() if k.startswith("initial/")}
        # These transforms operate per parameter, independent of storage layout.
        optimizer = Agent._make_opt(None, lr=4e-5, warmup=2)
        state = optimizer.init(params)
        update = jax.jit(optimizer.update)
        for step in range(4):
            raw = json.loads((args.native / f"step{step}-gradients.json").read_text())
            gradients = {k: jnp.asarray(v, jnp.float32) for k, v in raw.items()}
            assert gradients.keys() == params.keys()
            updates, state = update(gradients, state, params)
            params = optax.apply_updates(params, updates)
            actual = load_file(args.native / f"step{step}.safetensors")
            for kind, values, prefix in [("parameter", params, ""), ("momentum", state[2][1], "adam_m."),
                                         ("variance", state[1][1], "adam_v.")]:
                maximum = 0.
                for name, expected in values.items():
                    expected = np.asarray(expected)
                    value = actual[prefix + name].reshape(-1)
                    tolerance = 3e-5 if kind == "momentum" else 3e-6
                    np.testing.assert_allclose(value, expected, atol=tolerance, rtol=3e-4, err_msg=f"step{step}/{kind}/{name}")
                    maximum = max(maximum, float(np.max(np.abs(value-expected))))
                result["comparisons"].append(dict(step=step, component=kind, maximum_absolute_error=maximum))
            budget.check(f"step{step}")
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        budget.close()
        (args.output / "result.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
