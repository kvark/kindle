"""Independent F64/PyTorch CPU oracle for dense causal Tiny and its gradients.

Uses the existing independent encoder reference, not Kindle's graph builder.
This is numerical analysis, never a CPU learner or gameplay result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from levjepa_tiny_reference import encode, native_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    weights = {name: value.double().requires_grad_()
               for name, value in load_file(args.checkpoint, device="cpu").items()}
    patch_weight = weights["encoder.patch_embed.proj.weight"]
    weights["encoder.patch_embed.proj.weight"] = patch_weight.detach().reshape(192, 768).requires_grad_()
    batch, length = 2, 2
    patches = torch.randn(batch, length * 196, 768, device="cpu", dtype=torch.float64,
                          generator=torch.Generator(device="cpu").manual_seed(712)) * .3
    ids = torch.arange(length * 196, device="cpu")[None].repeat(batch, 1)
    tokens = encode(patches, ids, 14, weights)[:, 1:]
    state, projection = 0xD1_30_00_03_00_00_00_01, []
    for _ in range(192 * 64):
        state ^= state >> 12
        state ^= (state << 25) & ((1 << 64) - 1)
        state ^= state >> 27
        bit = ((state * 0x2545_F491_4F6C_DD1D) & ((1 << 64) - 1)) >> 63
        projection.append(.125 if bit else -.125)
    grid = (tokens @ torch.tensor(projection, dtype=torch.float64).reshape(192, 64))
    grid = grid.reshape(batch, length, 14, 14, 64)
    pooled = (grid[:, :, 0::2, 0::2] + grid[:, :, 0::2, 1::2]
              + grid[:, :, 1::2, 0::2] + grid[:, :, 1::2, 1::2]) / 4
    coefficients = (torch.arange(pooled.numel(), dtype=torch.float64).reshape_as(pooled) % 19 + 1) / 19
    loss = (pooled.square() * coefficients).mean() * 17
    loss.backward()
    tensors = {"loss": loss.detach().reshape(1).float()}
    for time in range(length):
        tensors[f"pixels_{time}"] = patches[:, time * 196:(time + 1) * 196].reshape(batch, -1).float().contiguous()
        tensors[f"features_{time}"] = pooled[:, time].reshape(batch, -1).detach().float().contiguous()
        tensors[f"coefficients_{time}"] = coefficients[:, time].reshape(batch, -1).float().contiguous()
    for name, weight in weights.items():
        if name == "encoder.cls_token":
            continue
        assert weight.grad is not None and weight.grad.isfinite().all(), name
        tensors[f"gradient.{name}"] = native_layout(weight.grad.detach()).float()
    # Independent regularizer value/derivative reference, including its scale.
    features = pooled.detach().requires_grad_()
    z = features.mean(dim=(2, 3)).reshape(batch * length, 64)
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / (batch * length - 1)
    std = (z.var(dim=0, unbiased=True) + 1e-4).sqrt()
    off_diagonal = covariance[~torch.eye(64, dtype=torch.bool)]
    penalty = (1 - std).relu().mean() + .04 * off_diagonal.square().sum() / 64
    (5 * penalty).backward()
    tensors["regularization.loss"] = (5 * penalty).detach().reshape(1).float()
    tensors["regularization.spread"] = std.detach().mean().reshape(1).float()
    for time in range(length):
        tensors[f"regularization.gradient_{time}"] = features.grad[:, time].reshape(batch, -1).float().contiguous()
    save_file(tensors, args.output / "reference.safetensors")
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(batch=batch, length=length, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=checkpoint_sha, torch=torch.__version__,
                  reference_precision="float64", gpu_execution=False, loss=loss.item())
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
