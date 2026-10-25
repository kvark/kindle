# Backend performance: retained result, work parked

Training is the priority. This file is an archive pointer, not an optimization
queue. Use the [current roadmap](kindle_single_life_dreamer_plan.md) and
[handover](handover.md). The former detailed work ledger remains in Git at8d464e0.

The preserved c8aeba28 implementation combines reduced-precision kernels,
resident targets, GPU sampling and convolution reuse. Complete synthetic RGB
updates last measured39.740ms and CDP39.398ms; the retained inclusive RGB/JAX
ratio is1.525×. Those are workload-specific timings, not GPU utilization,
learning superiority, cross-device performance or strict JAX implementation parity.

Important evidence:

- [Combined backend integration](results/2026-10-12-backend-integration.md)
- [Split-convolution tiles](results/2026-10-12-split-convolution-tiles.md)
- [Resident targets](results/2026-10-16-resident-targets-qualified.md)
- [Same-driver convolution counters](results/2026-10-21-jax-current-convolution.md)
- [Convolution reuse integration](results/2026-10-21-convolution-reuse-native.md)
- [Raw gradients and complete-update cost](results/2026-10-21-convolution-reuse-cost.md)
- [Training return and B32 failure](results/2026-10-21-cdp-training-return.md)

The counter comparison did **not** establish identical native/JAX kernels,
tiling or machine code. Operand feeding/reuse and control flow explained useful
parts of the gap, and qualified changes improved full-update cost. Rejected
variants and failed captures remain in their dated reports. Do not restart the
old geometry, compiler or profiling queues from their historical next steps.

The handover branches preserve the exact qualified library source with portable
Git dependencies. They are archival implementations, not a fresh upstream port
or a claim that every experimental feature belongs upstream. Only the user
merges PRs. Rebuild and check the target machine before relying on historical
hardware results; do not block learning on another JAX parity campaign.
