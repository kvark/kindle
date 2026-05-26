"""Train a denoising autoencoder on procedural grids.

Architecture mirrors kindle's CnnDqn encoder + small decoder. After
training, the encoder conv1/conv2/conv3/fc1 weights are exported as a
partial wm.safetensors that meganeura can load (load_checkpoint is
permissive — missing params keep xavier init).

Topology (matches kindle/encoder.rs::CnnEncoderDqn for 64x64 input):
  conv1: in_ch=C, 32 filters, k=8, stride=4 → (32, 15, 15)
  conv2: 32 → 64, k=4, stride=2 → (64, 6, 6)
  conv3: 64 → 64, k=3, stride=1 → (64, 4, 4)
  fc1: 1024 → 512
  fc2: 512 → latent_dim

Decoder: mirror with ConvTranspose + Linear.
Objective: MSE reconstruction of clean grid from noisy input.
Noise: drop 20% of cells to random colors.
"""
from __future__ import annotations

import argparse
import struct
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        # Match kindle CnnEncoderDqn (no bias on convs to match meganeura).
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        # For 64x64 input → conv3 output = (64, 4, 4) → flat 1024.
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, latent_dim, bias=False)

    def forward_features(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h  # (B, 64, 4, 4)

    def forward(self, x):
        h = self.forward_features(x)
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        z = self.fc2(h)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        # Mirror: 1024 → (64, 4, 4) → upsample to 64x64
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0)
        # k=12 stride=4 gets us back to 64x64: (14-1)*4 + 12 = 64.
        self.deconv1 = nn.ConvTranspose2d(32, out_channels, kernel_size=12, stride=4, padding=0)

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv2(h))
        x = self.deconv1(h)  # → (B, C, 64, 64)
        return x


def add_noise(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    mask = (torch.rand_like(x[:, :1]) < p).float()  # same noise across channels
    noise = torch.rand_like(x)  # uniform in [0, 1]
    return x * (1 - mask) + noise * mask


def export_partial_safetensors(encoder: Encoder, out_path: Path,
                               in_channels: int) -> None:
    """Write a wm.safetensors that contains only encoder.conv{1,2,3} +
    encoder.fc1 weights, flat float32, in the format meganeura expects.
    """
    import safetensors.torch as st_torch
    # Build flat-float32 tensors keyed by meganeura parameter names.
    out = {}
    # Conv weights: meganeura stores flat (out * in * kH * kW), same as
    # PyTorch (out, in, kH, kW) flattened row-major.
    # Conv weights: meganeura Conv2d stores flat (out * in * kH * kW),
    # which matches PyTorch (out, in, kH, kW) flattened row-major.
    out["encoder.conv1.weight"] = encoder.conv1.weight.detach().flatten().contiguous()
    out["encoder.conv2.weight"] = encoder.conv2.weight.detach().flatten().contiguous()
    out["encoder.conv3.weight"] = encoder.conv3.weight.detach().flatten().contiguous()
    # fc weights: meganeura nn::Linear is (in_features, out_features); PyTorch
    # nn.Linear.weight is (out, in). Transpose before flatten.
    out["encoder.fc1.weight"] = encoder.fc1.weight.detach().t().contiguous().flatten()
    out["encoder.fc1.bias"] = encoder.fc1.bias.detach().flatten().contiguous()
    out["encoder.fc2.weight"] = encoder.fc2.weight.detach().t().contiguous().flatten()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st_torch.save_file(out, str(out_path))
    print(f"wrote partial encoder weights → {out_path}")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)} float32")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/aff_runs/pretrain_grids.npz")
    p.add_argument("--out", default="/tmp/aff_runs/pretrain_ckpt/wm.safetensors")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--noise-p", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    print(f"loading {a.data}")
    data = np.load(a.data)
    frames = data["frames"]
    print(f"  {frames.shape} {frames.dtype}")

    # Scale 0..15 → 0..1 to match kindle's frame normalization.
    x = torch.from_numpy(frames.astype(np.float32) / 15.0).unsqueeze(1)  # (N,1,64,64)
    if a.in_channels == 2:
        # Frame-diff mode: synthesize a "previous-frame" channel via
        # randomly shifted-and-noised version of x. Forces encoder to
        # care about per-pixel deltas.
        prev = torch.roll(x, shifts=1, dims=2)  # shift down
        # Channel 0 = current frame; channel 1 = delta = current - prev.
        x = torch.cat([x, x - prev], dim=1)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0)

    device = torch.device(a.device)
    enc = Encoder(in_channels=a.in_channels, latent_dim=a.latent_dim).to(device)
    dec = Decoder(latent_dim=a.latent_dim, out_channels=a.in_channels).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)

    print(f"training {a.epochs} epochs on {len(ds)} samples")
    for ep in range(a.epochs):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            noisy = add_noise(batch, p=a.noise_p)
            z = enc(noisy)
            recon = dec(z)
            # Target is the CLEAN image at original spatial size.
            # Note deconv may produce slightly different size; trim/pad.
            if recon.shape[-2:] != batch.shape[-2:]:
                recon = recon[..., :batch.shape[-2], :batch.shape[-1]]
            loss = F.mse_loss(recon, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep+1}/{a.epochs} mse={np.mean(losses):.4f}")

    export_partial_safetensors(enc, Path(a.out), in_channels=a.in_channels)


if __name__ == "__main__":
    main()
