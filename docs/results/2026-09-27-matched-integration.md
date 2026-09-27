# Matched integration smokes

All four runners complete **6,144 actual actions and 1,186 updates**, with the
first update at action 1,404 and zero remaining update debt. This is one smoke
seed (7301), **not the three-seed learning comparison or a competence result**.
[Configuration, all episodes/tails, curves, finite checkpoints and device records](2026-09-27-matched-integration.json).

| Method | Run seconds | Construction seconds | Actions/s | Minimum estimated GPU headroom |
| --- | ---: | ---: | ---: | ---: |
| Upstream Dreamer12M | 245.643 | 52.018 | 25.01 | 2.37 GiB |
| Pretrained Tiny | 239.594 | 11.423 | 25.64 | 10.73 GiB |
| Initial Tiny | 240.014 | 11.505 | 25.60 | 10.73 GiB |
| Large | 362.681 | 17.177 | 16.94 | 5.40 GiB |

These timings include prefill, initial policy compilation and final checkpoint
writes, not agent construction. They are **not steady-state throughput estimates**.
There are too few updates to infer learning quality or claim a speed advantage
from the small upstream/Tiny difference. The historical unreconciled upstream
59-actions/s figure is not a comparison to this protocol.

The common recipe is F32/B16/T64/H15/R256/N6, full18 actions, sticky .25, repeat4,
no reset no-ops, and no exploration/reward aid. All final tensors are finite;
native world/behavior optimizer moments are nonzero, and upstream RSSM, encoder
and actor parameters change. All four direct-child guards pass and reap their
workers, with no kernel GPU fault. No application NVML telemetry is collected.
Headroom is sampled Vulkan budget minus usage, not physical free or peak VRAM.

Kindle uses frozen native-RGB features and latent prediction; upstream jointly
learns an RGB64 encoder and reconstruction. Their training/input differences
remain explicit. Full matched learning curves are still required for Phase 2.
