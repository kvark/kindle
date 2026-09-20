"""Independent CPU numerical fixture, not a pretraining implementation.

Uses ordinary PyTorch attention/autograd with clip-major tensors; the native
graph uses token-major block products. No CUDA, native extension or NVML calls.
Run with a CPU-only device environment and one thread. Requires torch and
safetensors in an analysis environment, not Kindle's runtime environment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn import functional as F


HIDDEN, HEADS, LAYERS, FRAMES = 192, 3, 12, 16
CONFIG = dict(batch=3, local_views=2, projector_hidden=1024,
              projector_output=128, directions=7)


def parameters(generator):
    weights = {}

    def parameter(name, shape, kind="normal"):
        if kind == "one":
            value = torch.ones(shape, device="cpu")
        elif kind == "zero":
            value = torch.zeros(shape, device="cpu")
        else:
            value = torch.randn(shape, generator=generator, device="cpu") * 0.02
        weights[name] = value.requires_grad_()

    def linear(name, inputs, outputs):
        parameter(name + ".weight", (outputs, inputs))
        parameter(name + ".bias", (outputs,), "zero")

    def norm(name, width):
        parameter(name + ".weight", (width,), "one")
        parameter(name + ".bias", (width,), "zero")

    parameter("encoder.cls_token", (1, 1, HIDDEN))
    linear("encoder.patch_embed.proj", 768, HIDDEN)
    for layer in range(LAYERS):
        prefix = f"encoder.blocks.{layer}"
        norm(prefix + ".norm1", HIDDEN)
        linear(prefix + ".attn.qkv", HIDDEN, 3 * HIDDEN)
        linear(prefix + ".attn.proj", HIDDEN, HIDDEN)
        norm(prefix + ".norm2", HIDDEN)
        linear(prefix + ".mlp.fc1", HIDDEN, 4 * HIDDEN)
        linear(prefix + ".mlp.fc2", 4 * HIDDEN, HIDDEN)
    norm("encoder.norm", HIDDEN)
    linear("projector.fc1", HIDDEN, CONFIG["projector_hidden"])
    norm("projector.norm", CONFIG["projector_hidden"])
    linear("projector.fc2", CONFIG["projector_hidden"], CONFIG["projector_output"])
    assert len(weights) == 155
    assert sum(p.numel() for p in weights.values()) == 5_817_472
    return weights


def linear(x, weights, name):
    return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])


def norm(x, weights, name):
    return F.layer_norm(x, (HIDDEN,), weights[name + ".weight"],
                        weights[name + ".bias"], eps=1e-6)


def rotary(q, ids, grid):
    # Original positions survive dropping. Each axis repeats its ten
    # frequencies in halves, while the rotation itself uses adjacent pairs.
    positions = (ids // (grid * grid), (ids // grid) % grid, ids % grid)
    frequency = 10000.0 ** (-torch.arange(10, device="cpu") / 10.0)
    angles = torch.cat([(p[..., None] * frequency).repeat(1, 1, 2)
                        for p in positions], dim=-1)
    angles = F.pad(angles, (0, 4))[:, None]
    angles = F.pad(angles, (0, 0, 1, 0))  # CLS uses identity rotation.
    paired = q.reshape(*q.shape[:-1], 32, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)
    return q * angles.cos() + rotated * angles.sin()


def allowed_attention(ids, grid):
    frames = ids // (grid * grid)
    allowed = frames[:, :, None] >= frames[:, None, :]
    allowed = F.pad(allowed, (1, 0, 1, 0), value=False)
    allowed[:, 0, :] = True
    return allowed


def encode(patches, ids, grid, weights):
    clips, kept, _ = patches.shape
    cls = weights["encoder.cls_token"].expand(clips, -1, -1)
    x = torch.cat((cls, linear(patches, weights, "encoder.patch_embed.proj")), dim=1)
    allowed = allowed_attention(ids, grid)[:, None]
    for layer in range(LAYERS):
        prefix = f"encoder.blocks.{layer}"
        qkv = linear(norm(x, weights, prefix + ".norm1"), weights, prefix + ".attn.qkv")
        q, k, v = qkv.reshape(clips, kept + 1, 3, HEADS, 64).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, ids, grid), rotary(k, ids, grid)
        scores = (q @ k.transpose(-1, -2)) / 8.0
        probability = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        attention = (probability @ v).transpose(1, 2).reshape(clips, kept + 1, HIDDEN)
        x = x + linear(attention, weights, prefix + ".attn.proj")
        mlp = linear(norm(x, weights, prefix + ".norm2"), weights, prefix + ".mlp.fc1")
        x = x + linear(F.gelu(mlp, approximate="none"), weights, prefix + ".mlp.fc2")
    return norm(x, weights, "encoder.norm")


def objective(global_cls, local_cls, directions, weights):
    x = linear(torch.cat((global_cls, local_cls)), weights, "projector.fc1")
    x = F.batch_norm(x, None, None, weights["projector.norm.weight"],
                     weights["projector.norm.bias"], training=True, eps=1e-5)
    z = linear(F.gelu(x, approximate="none"), weights, "projector.fc2")
    views = CONFIG["local_views"] + 1
    z_views = z.reshape(views, CONFIG["batch"], -1)
    invariance = (z_views - z_views[:1]).square().mean()
    knots = torch.linspace(0, 3, 17, device="cpu")
    phi = torch.exp(-knots.square() / 2)
    quadrature = torch.full((17,), 2 * 3 / 16, device="cpu")
    quadrature[[0, -1]] *= 0.5
    angles = (z_views @ directions)[..., None] * knots
    ecf_real, ecf_imag = angles.cos().mean(1), angles.sin().mean(1)
    errors = (ecf_real - phi).square() + ecf_imag.square()
    sigreg = (errors * phi * quadrature).sum(-1).mean() * CONFIG["batch"]
    return invariance + 0.02 * sigreg, invariance, sigreg, z


def native_layout(value):
    if value.ndim == 2:
        return value.T.contiguous()
    if value.ndim == 3:  # CLS parameter
        return value.reshape(1, HIDDEN)
    return value.contiguous()


def make_fixture(output):
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(7301)
    weights = parameters(generator)
    tensors, manifest, views = {}, dict(config=CONFIG, seed=7301, patch_ids={}), {}
    for name, clips, grid in [("global", CONFIG["batch"], 14),
                              ("local", CONFIG["batch"] * CONFIG["local_views"], 6)]:
        full = FRAMES * grid * grid
        kept = round(full * 0.05)
        ids = torch.stack([torch.randperm(full, generator=generator, device="cpu")[:kept]
                           for _ in range(clips)])
        patches = torch.randn((clips, kept, 768), generator=generator, device="cpu")
        tensors[name + ".patches"] = patches.transpose(0, 1).contiguous().reshape(-1, 768)
        manifest["patch_ids"][name] = ids.T.flatten().tolist()
        allowed = allowed_attention(ids, grid)
        mask = torch.zeros(allowed.shape, device="cpu").masked_fill(~allowed, -torch.inf)
        tensors[name + ".mask"] = mask.permute(1, 0, 2)[:, :, None, :].expand(
            -1, -1, HEADS, -1).contiguous().reshape(kept + 1, -1)
        views[name] = patches, ids, grid

    # Check causality, independent clips and batch layout in the reference
    # itself before producing any native golden values.
    with torch.no_grad():
        patches, ids, grid = views["global"]
        baseline = encode(patches, ids, grid, weights)
        independent = torch.cat([encode(p[None], i[None], grid, weights)
                                 for p, i in zip(patches, ids)])
        torch.testing.assert_close(baseline, independent, atol=1e-5, rtol=1e-5)
        changed = patches.clone()
        frames = ids // (grid * grid)
        changed[frames >= 8] += 2.0
        future = encode(changed, ids, grid, weights)
        torch.testing.assert_close(baseline[:, 1:][frames < 8],
                                   future[:, 1:][frames < 8], atol=0, rtol=0)
        assert (baseline[:, 0] - future[:, 0]).abs().max() > 0.01

    directions = torch.randn((128, CONFIG["directions"]), generator=generator, device="cpu")
    directions = F.normalize(directions, dim=0)
    tensors["sigreg.directions"] = directions
    global_cls = encode(*views["global"], weights)[:, 0]
    local_cls = encode(*views["local"], weights)[:, 0]
    result = objective(global_cls, local_cls, directions, weights)
    result[0].backward()
    for name, value in weights.items():
        assert value.grad is not None and value.grad.isfinite().all(), name
        tensors["weight." + name] = native_layout(value.detach())
        tensors["gradient." + name] = native_layout(value.grad)
    for name, value in zip(("loss", "invariance", "sigreg", "embeddings"), result):
        tensors["expected." + name] = value.detach().reshape(-1).contiguous()
    for name, value in tensors.items():
        if not name.endswith(".mask"):
            assert value.isfinite().all(), name
        assert value.device.type == "cpu", name
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "reference.safetensors"
    save_file(tensors, str(fixture))
    manifest.update(torch_version=torch.__version__,
                    tensors_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    losses=[float(v.detach()) for v in result[:3]],
                    checks=["independent clip parity", "exact patch causality", "CLS sees future",
                            "all 155 finite gradients"],
                    tolerance=dict(relative_l2=0.003, absolute_rms=0.00001,
                                   relative_max=0.01, absolute_max=0.00003))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "patch_ids"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory; refuses overwrite")
    make_fixture(parser.parse_args().output)
