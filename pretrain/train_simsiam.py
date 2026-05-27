"""SimSiam pretraining for kindle's CNN encoder.

Self-supervised contrastive learning without negatives (Chen & He 2020).
Two augmented views of the same image → encoder → projector → predictor;
loss = - cos_sim(predictor(view1), stop_grad(projector(view2))).

Better than DAE for preserving semantic structure because no decoder
collapse target — encoder is free to learn what distinguishes images
rather than what reconstructs them.

Same Encoder class as train_dae.py; same export format (partial
wm.safetensors with rescaled conv/fc weights).

Augmentations chosen for cartoon-grid input (0..15 integer cells):
  - Random crop + resize back to 64×64
  - Color-permutation (remap 0..15 → random permutation 0..15)
  - Random spatial flips (h, v)

Color-permutation is key: it preserves SHAPE/STRUCTURE but changes the
literal pixel values. The encoder learns to be invariant to color
identity → focuses on geometric features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    """Same as train_dae.py — matches kindle CnnEncoderDqn topology."""
    def __init__(self, in_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, latent_dim, bias=False)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class Projector(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.BatchNorm1d(dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def augment_batch(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """x: (B, 1, 64, 64) float in [0, 1] (already / 15).

    Returns augmented copy via random crop + flips + color permutation.
    """
    B, C, H, W = x.shape
    out = x.clone()

    # Random horizontal flip per-sample
    for b in range(B):
        if rng.random() < 0.5:
            out[b] = torch.flip(out[b], dims=[-1])
        if rng.random() < 0.5:
            out[b] = torch.flip(out[b], dims=[-2])

    # Random crop + resize (per-batch, simpler)
    crop_h = int(H * (0.7 + 0.3 * rng.random()))
    crop_w = int(W * (0.7 + 0.3 * rng.random()))
    y0 = int(rng.integers(0, max(1, H - crop_h)))
    x0 = int(rng.integers(0, max(1, W - crop_w)))
    out = out[:, :, y0:y0+crop_h, x0:x0+crop_w]
    out = F.interpolate(out, size=(H, W), mode='nearest')

    # Color permutation: map 0..15 → random perm; equivalent in [0,1]
    # space to multiplying integer cells by some perm. We approximate by
    # linear remap with random gain+bias per-sample, clamped to [0,1].
    for b in range(B):
        g = float(0.5 + rng.random())  # 0.5..1.5
        bias = float((rng.random() - 0.5) * 0.3)  # -0.15..0.15
        out[b] = torch.clamp(out[b] * g + bias, 0.0, 1.0)

    return out


def neg_cos_sim(p, z):
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


def export_partial_safetensors(encoder: Encoder, out_path: Path,
                                in_channels: int) -> None:
    """Same as train_dae's export — rescales to kindle xavier magnitude."""
    import math
    import safetensors.torch as st_torch

    def rescale_to_kindle(t):
        n = t.numel()
        fan = max(1, int(math.sqrt(n)))
        kindle_std = math.sqrt(3.0 / fan)
        cur_std = float(t.std()) or 1.0
        return (t.detach() * (kindle_std / cur_std)).flatten().contiguous()

    out = {}
    out["encoder.conv1.weight"] = rescale_to_kindle(encoder.conv1.weight)
    out["encoder.conv2.weight"] = rescale_to_kindle(encoder.conv2.weight)
    out["encoder.conv3.weight"] = rescale_to_kindle(encoder.conv3.weight)
    out["encoder.fc1.weight"] = rescale_to_kindle(encoder.fc1.weight.t().contiguous())
    out["encoder.fc1.bias"] = encoder.fc1.bias.detach().flatten().contiguous()
    out["encoder.fc2.weight"] = rescale_to_kindle(encoder.fc2.weight.t().contiguous())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    st_torch.save_file(out, str(out_path))
    print(f"wrote partial encoder weights → {out_path}")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)} float32 std={v.std():.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/aff_runs/pretrain_grids_50k.npz")
    p.add_argument("--out", default="/tmp/aff_runs/pretrain_simsiam_ckpt/wm.safetensors")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    print(f"loading {a.data}")
    data = np.load(a.data)
    frames = data["frames"]
    print(f"  {frames.shape} {frames.dtype}")

    x = torch.from_numpy(frames.astype(np.float32) / 15.0).unsqueeze(1)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0)

    device = torch.device(a.device)
    enc = Encoder(in_channels=a.in_channels, latent_dim=a.latent_dim).to(device)
    proj = Projector(dim=a.latent_dim).to(device)
    pred = Predictor(dim=a.latent_dim).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(proj.parameters()) + list(pred.parameters()),
        lr=a.lr,
    )
    rng = np.random.default_rng(a.seed)

    print(f"training {a.epochs} epochs on {len(ds)} samples")
    for ep in range(a.epochs):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            v1 = augment_batch(batch, rng)
            v2 = augment_batch(batch, rng)

            z1 = proj(enc(v1))
            z2 = proj(enc(v2))
            p1 = pred(z1)
            p2 = pred(z2)
            loss = 0.5 * (neg_cos_sim(p1, z2) + neg_cos_sim(p2, z1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep+1}/{a.epochs} loss={np.mean(losses):.4f}")

    export_partial_safetensors(enc, Path(a.out), in_channels=a.in_channels)


if __name__ == "__main__":
    main()
