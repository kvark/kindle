"""Generate native parity fixtures from the inspected, pinned LeVJEPA release.

Run in a separate PyTorch/Transformers environment. Production Kindle does not
depend on this stack or execute downloaded Python. The reference directory must
contain the four files from the exact snapshot below; nothing is fetched here.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file, save_file


REVISION = "e831a0347737fcaa660b39c57d41c109de399845"
HASHES = {
    "config.json": "4116cab6f850ac22791ee4c58537c4679fec399f9642564c4492d834f4b6a91f",
    "configuration_levjepa.py": "3c8756c0afd578610c9ce7e4874e3cd8326b8a69a4767ad4ff82f0210a9e5df6",
    "modeling_levjepa.py": "764b1f208732e42889d717e2427295d1efd2f6e6c2b59a321671703572390efc",
    "model.safetensors": "da8bd836ce6532e1b0074ee5a6a46c65b67103f96323529ec4195be1538edc7d",
}


def letterbox(rgb: np.ndarray, target: int = 224) -> np.ndarray:
    """Independent float32 transcription of Kindle's RGB8 preprocessing."""
    height, width, _ = rgb.shape
    scale = min(np.float32(target) / width, np.float32(target) / height)
    sw, sh = (max(1, min(target, int(np.floor(np.float32(size) * scale + np.float32(.5)))))
              for size in (width, height))
    x = np.clip((np.arange(sw, dtype=np.float32) + .5) * np.float32(width) / np.float32(sw) - .5, 0, width - 1)
    y = np.clip((np.arange(sh, dtype=np.float32) + .5) * np.float32(height) / np.float32(sh) - .5, 0, height - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    mx, my = (x - x0.astype(np.float32))[None, :, None], (y - y0.astype(np.float32))[:, None, None]
    top = rgb[y0[:, None], x0].astype(np.float32) * (1 - mx) + rgb[y0[:, None], x1].astype(np.float32) * mx
    bottom = rgb[y1[:, None], x0].astype(np.float32) * (1 - mx) + rgb[y1[:, None], x1].astype(np.float32) * mx
    resized = np.floor(top * (1 - my) + bottom * my + .5).astype(np.uint8)
    output = np.empty((target, target, 3), dtype=np.uint8)
    output[:] = [124, 116, 104]
    oy, ox = (target - sh) // 2, (target - sw) // 2
    output[oy:oy + sh, ox:ox + sw] = resized
    return output


def load_reference(directory: Path):
    for filename, expected in HASHES.items():
        with (directory / filename).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"unrecognized reference file {filename}: {actual}")
    package = types.ModuleType("_kindle_levjepa_reference")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    modules = {}
    for name in ("configuration_levjepa", "modeling_levjepa"):
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", directory / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    config = modules["configuration_levjepa"].LeVJEPAConfig(
        **json.loads((directory / "config.json").read_text())
    )
    model = modules["modeling_levjepa"].LeVJEPAModel(config)
    model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rgb-clips", type=Path, help="optional uint8 .npy [2,16,H,W,3]")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_reference(args.reference_dir).cuda()
    if args.rgb_clips:
        rgb = np.load(args.rgb_clips, allow_pickle=False)
        if rgb.ndim != 5 or rgb.shape[:2] != (2, 16) or rgb.shape[-1] != 3 or min(rgb.shape) < 1 or rgb.dtype != np.uint8:
            raise ValueError("RGB fixture must have shape [2,16,H,W,3], dtype uint8")
    else:
        index = np.arange(224 * 224 * 3).reshape(224, 224, 3)
        rgb = np.stack([
            np.stack([((index * 37 + frame * 19 + clip * 53 + 13) % 251).astype(np.uint8)
                      for frame in range(16)])
            for clip in range(2)
        ])
    raw_rgb = torch.from_numpy(rgb).float()
    rgb = np.stack([np.stack([letterbox(frame) for frame in clip]) for clip in rgb])
    pixels = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    pixels = ((pixels - mean) / std).contiguous()
    tokens = []
    with torch.inference_mode():
        for clip in pixels:
            video = clip.permute(1, 0, 2, 3).unsqueeze(0).cuda()
            output = model(video).last_hidden_state[:, 1:].reshape(16, 196, 1024)
            tokens.append(output.cpu())
        changed = pixels[0].clone()
        changed[8:] = pixels[1, 8:]
        output = model(changed.permute(1, 0, 2, 3).unsqueeze(0).cuda()).last_hidden_state
        causal_error = (output[:, 1:1 + 8 * 196].cpu().reshape(8, 196, 1024) - tokens[0][:8]).abs().max().item()
        if causal_error > 1e-5:
            raise AssertionError(f"reference leaks future frames: {causal_error}")
    state = 0xD1_30_00_03_00_00_00_01
    projection = []
    for _ in range(1024 * 64):
        state ^= state >> 12
        state ^= (state << 25) & ((1 << 64) - 1)
        state ^= state >> 27
        bit = ((state * 0x2545_F491_4F6C_DD1D) & ((1 << 64) - 1)) >> 63
        projection.append(0.125 if bit else -0.125)
    tokens = torch.stack(tokens)
    projected = tokens @ torch.tensor(projection).reshape(1024, 64)
    grid = projected.reshape(2, 16, 14, 14, 64)
    pooled = (grid[:, :, 0::2, 0::2] + grid[:, :, 0::2, 1::2]
              + grid[:, :, 1::2, 0::2] + grid[:, :, 1::2, 1::2]) / 4
    metadata = {"revision": REVISION, "torch": torch.__version__, "causal_max_abs": str(causal_error)}
    save_file({"rgb": raw_rgb, "pixels": pixels, "tokens": tokens, "projected": projected,
               "pooled": pooled.reshape(2, 16, 49, 64).contiguous()}, args.output, metadata=metadata)
    print(json.dumps({**metadata, "fixture": str(args.output), "token_min": min(t.min().item() for t in tokens), "token_max": max(t.max().item() for t in tokens)}, sort_keys=True))


if __name__ == "__main__":
    main()
