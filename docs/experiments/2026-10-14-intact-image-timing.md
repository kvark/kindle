# Sparse timing of intact cached image-training execution

The [image-region experiment](../results/2026-10-14-image-training-region.md)
leaves complete RGB12M/B32 at63.640ms,2.39× retained original JAX. Its masked
attribution strongly interacts and cannot rank component latency. Measure the
real execution without suppressing producers or changing operands.
Artifacts: `runs/intact-image-timing-20261014.rQVbtlH0`.

Keep qualified e1a2407c arithmetic/model/precision/dependencies. A diagnostic
marks the31 existing dependency groups containing the16 image kernels and30
parameter-gradient reduction tails. Some groups also contain independent
heavy/other operations; retain their membership and report mixed envelopes,
not pure image-kernel durations. No group is split or reordered. Adjacent
unmarked groups remain combined; original four-chunk and staged boundaries
remain. Cache each resulting section, using Blade's existing primary-boundary
timestamp mechanism with distinct labels. No per-dispatch timing, skipped work,
secondary-query-pool rewrite, Nsight/driver change or new GPU kernel.

CPU checks must prove each original group executes exactly once and in order
under empty, adjacent, edge and chunk-clipped selections. First one short
ordinary/traced pair,16 warmup+8 measured updates/arm,300 seconds/arm. Require
exact five plans/pipeline lists,272-or-short metric reports excluding timings,
first outputs and initial/first/final292 tensors; same device, finite values,
headroom and cache replay. Preserve failures; no blind retry.

After smoke review, one ordinary/traced complete-update pair,16 warmup+256
measured updates/arm,300 seconds/arm. The traced arm records sparse scopes in
the real full learner. Require exact results/states and no more than3% mean
whole-update overhead (also disclose speedup/noise and median). If the gate
fails, do not use its timings as ordinary component costs or repeat timing
until a nicer result. Actual named regions, other graph spans, optimizer/copy
passes and uncovered host time must reconcile without double counting.
Sparse timing does not measure SM utilization or promise additive savings.

Existing persistent guarded systemd rules apply: Restart=no,KillMode=control-group,
bounded deadlines, no CPU quota on GPU work; expected5080,at least2GiB sampled
Vulkan estimated headroom,record allocation warnings. CPU preparation/review
uses1CPU/2GiB/zero swap. No NVML polling, production repin, Atari restart or
goal closure. Upstream remains Meganeura99e3de27/Bladee7c3f7b at this recheck;
no newly identified fix. Keep all old rejected candidates separate.
