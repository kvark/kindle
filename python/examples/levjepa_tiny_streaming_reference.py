"""CPU-only dense Tiny fixture for streaming, projection and reset checks.

Uses the independent attention reference and its original untrained weights.
This is numerical test data, not a pretrained checkpoint or gameplay result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import levjepa_tiny_reference as reference
from levjepa_reference import letterbox


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = args.training_reference / "reference.safetensors"
    manifest = json.loads((args.training_reference / "manifest.json").read_text())
    if sha(source) != manifest["tensors_sha256"] or sha(reference.__file__) != manifest["generator_sha256"]:
        raise ValueError("independent training reference identity differs")
    native = load_file(source, device="cpu")
    weights = {}
    for key, value in native.items():
        if key.startswith("weight.encoder."):
            name = key.removeprefix("weight.")
            weights[name] = (value.reshape(1, 1, 192) if name == "encoder.cls_token" else
                             value.T.contiguous() if value.ndim == 2 else value)
    del native
    assert len(weights) == 149 and sum(v.numel() for v in weights.values()) == 5_486_592
    exported = dict(weights)
    exported["encoder.patch_embed.proj.weight"] = weights["encoder.patch_embed.proj.weight"].reshape(192, 3, 1, 16, 16)
    save_file(exported, args.output / "encoder.safetensors")

    index = np.arange(64 * 80 * 3).reshape(64, 80, 3)
    rgb = np.stack([np.stack([((index * 37 + frame * 19 + clip * 53 + 13) % 251).astype(np.uint8)
                              for frame in range(16)]) for clip in range(2)])
    resized = np.stack([np.stack([letterbox(frame) for frame in clip]) for clip in rgb])
    pixels = torch.from_numpy(resized).permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    pixels = ((pixels - mean) / std).contiguous()
    patches = pixels.reshape(2, 16, 3, 14, 16, 14, 16).permute(0, 1, 3, 5, 2, 4, 6).reshape(2, 3136, 768)
    ids = torch.arange(3136, device="cpu")[None]
    with torch.inference_mode():
        tokens = torch.stack([reference.encode(clip[None], ids, 14, weights)[0, 1:].reshape(16, 196, 192)
                              for clip in patches])
        changed = patches[0].clone()
        changed[8 * 196:] = patches[1, 8 * 196:]
        future = reference.encode(changed[None], ids, 14, weights)[0, 1:].reshape(16, 196, 192)
        torch.testing.assert_close(future[:8], tokens[0, :8], atol=0, rtol=0)
    state, projection = 0xD1_30_00_03_00_00_00_01, []
    for _ in range(192 * 64):
        state ^= state >> 12
        state ^= (state << 25) & ((1 << 64) - 1)
        state ^= state >> 27
        bit = ((state * 0x2545_F491_4F6C_DD1D) & ((1 << 64) - 1)) >> 63
        projection.append(0.125 if bit else -0.125)
    projected = tokens @ torch.tensor(projection).reshape(192, 64)
    grid = projected.reshape(2, 16, 14, 14, 64)
    pooled = (grid[:, :, 0::2, 0::2] + grid[:, :, 0::2, 1::2]
              + grid[:, :, 1::2, 0::2] + grid[:, :, 1::2, 1::2]) / 4
    tensors = dict(rgb=torch.from_numpy(rgb).float(), pixels=pixels, tokens=tokens,
                   projected=projected, pooled=pooled.reshape(2, 16, 49, 64).contiguous())
    for value in tensors.values():
        assert value.device.type == "cpu" and value.isfinite().all()
    save_file(tensors, args.output / "reference.safetensors")
    result = dict(format=1, architecture="tiny", torch=torch.__version__, causal_max_abs=0,
                  training_reference_sha256=sha(source),
                  sources={name: sha(Path(__file__).with_name(name)) for name in (
                      "levjepa_tiny_streaming_reference.py", "levjepa_tiny_reference.py", "levjepa_reference.py")},
                  files={name: sha(args.output / name) for name in ("encoder.safetensors", "reference.safetensors")},
                  tolerance=dict(relative_l2=1e-4, max_abs=0.005), gpu_execution=False)
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
