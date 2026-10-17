# Grouped imagination cuts full network-update time by 8.1%

**Retain the grouped path.** The single [declared trial](../experiments/2026-10-06-cdp-imagination-throughput.md)
passes its numerical and >=5% speed gate. Extend the existing grouped block
matrix product from B2–16 through B128; no custom shader or new model setting.
Batch-one and >128 remain unchanged. Meganeura is refreshed to upstream
`c6376542`; Blade remains `e349cddf`. Both timing arms use that same backend.

| Ordinary synthetic full update | Serial B128 control | Grouped B128 |
| --- | ---: | ---: |
| Mean | 30.487 ms | 28.011 ms |
| Median | 30.549 ms | 27.894 ms |
| p90 | 30.988 ms | 28.705 ms |
| Imagination dispatches | 3,100 | 1,210 |

This is **1.088x throughput /8.12% less update time**, not an 8.12% measured
whole-game speedup. One pair,16 warmup plus256 measured updates per executable,
identical centered1009 checkpoint, fixture RNG701, no timestamp instrumentation
or overlapping builds. Posterior, imagination/targets, both training stages,
weight synchronization and slow EMA are timed; fixture generation, replay
bookkeeping, exports and environment acting are not. GPU utilization remains
unmeasured. No optimization sweep follows this success.

All346 restored tensors match the source and each other exactly. The first
posterior, sampled actions, continuation weights and value/slow targets match
exactly, as do all272 recorded world/behavior metric reports after removing
timing fields. This is evidence for this fixed synthetic sequence, not a
guarantee of bitwise equality for arbitrary learning trajectories.

## Qualification and cost

- Existing small-batch production outputs and all gradients remain bitwise
  exact against serial block operations. CPU tests cover parameter schemas,
  dispatch reduction and unchanged B1/B129/B1024 boundaries.
- Six independent F64 forward cases at B17/64/128 and small-RSSM block shapes
 256x64 /64x192 cover428,032 outputs. Maximum absolute error9.56e-7 and
 relative L2<2.84e-7 pass the unchanged2e-6*(1+abs(reference))/2e-5 gates.
- Original and centered cosine values/derivatives pass; full upstream CDP/RGB
 checks pass1,300/1,524 comparisons, including raw gradients and optimizer/EMA.
- Raw and centered real Pong smokes each complete1,024 actions/195 finite
 updates with zero debt, followed by1,024 frozen actions/zero updates.
 All346 saved tensors per frozen model are exactly unchanged.
- All14 guarded jobs and both independent CPU audits pass. No new allocation
 or validation warning, NVML polling or recovery. Sampled Vulkan estimated
 headroom is at least2GiB; this is not physical free or peak VRAM.
- 115 CPU Rust tests and1,172 Python tests pass, as do formatting and strict
 Clippy. Preparation uses one CPU,2GiB and zero swap.

The trial adds544 synthetic timing updates, eight exported reference updates
and2,048 smoke training plus2,048 frozen actions, not competence evidence.
Native:`6d38eea2ebe8f1ae81cea20e138a10b525ec371af465989071b731d12f897223`.

[Compact audit](2026-10-06-cdp-imagination-throughput.json).
Full reports, binaries and checkpoints:
`runs/cdp-imagination-20261006.e8agRpD0`.
The new attention-backward upstream changes are not an identified CDP fix;
this qualification does not retroactively change older results or qualify
unexercised Tiny attention paths.

Next: [the finite centered five-game500k allocation](../experiments/2026-10-06-cdp-centered-five-game-budget.md).
All five DreamerV3 quality targets remain open.
