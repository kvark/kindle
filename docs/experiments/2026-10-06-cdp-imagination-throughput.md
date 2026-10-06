# One bounded imagination-throughput trial before longer CDP training

The [centered learning result](../results/2026-10-06-cdp-centered-learning.md)
and [paired world probes](../results/2026-10-06-cdp-centered-world.md) justify
keeping centering. None of the five DreamerV3 targets is reached. Improve the
measured bottleneck before a larger finite learning allocation; no new loss,
capacity, replay ratio, horizon, precision setting or exploration coefficient.

The completed profile at `runs/cdp-centered-profile-20261006.EvJjduZZ` finds
3,100 imagination dispatches, including660 SplitA,660 SplitB and360 Concat.
Ordinary imagination GPU/wall medians are10.91/13.28ms; the production
imagination/target stage averages16.18ms out of32.13ms/update. Per-dispatch
instrumentation inflates wall time3.97x; family percentages are not ordinary
GPU utilization or whole-agent shares. The guard, unchanged profiling tensors
and source checkpoint all pass.

## Candidate and fixed comparison

`BlockLinear` currently uses one native grouped matrix product for B2–16, but
falls back to per-block slices/matmuls/concatenations for B128 imagination.
Try extending that existing path through B128. Preserve parameter leaves,
initialization, optimizer, equations and the B1/>128 paths. This uses the current
Meganeura primitive, not a resurrected quarantined binary or a custom shader.

First adopt upstreamc6376542 (attention-backward/layout-search update) on top of
qualifiedb684ffd9; Bladee349cddf stays unchanged. Preserve a newly built
same-backend serial control, then build the grouped candidate. Do not conflate
an upstream refresh with the grouped-path comparison or repin historical runs.

CPU graph/schema tests cover the expanded range and unchanged outside range;
formatting and strict Clippy pass. Keep the existing exact small-batch GPU
output/gradient checks. Add independent F64 forward checks for B17/64/128 at
the small RSSM's block shapes, with maximum error<=2e-6*(1+abs(reference)) and
relative L2<=2e-5. No tolerance relaxation after seeing results. Preserve
upstream CDP/RGB checks for complete learning targets/gradients and both raw /
centered production/frozen smokes for a changed native build.

One paired full-update timing, same centered1009 source checkpoint and synthetic
batch generator, on the same upstream backend. Fixed warmup and measured update
counts declared in the runner before launch; save every report and final
checkpoint. Compare numerical outputs before using speed, and keep profiled
session/kernel time separate from ordinary full-update timing. No overlapping
build or GPU work. One short real Pong learning/frozen check suffices; no new
multi-seed speed campaign.

Retain the grouped path only if qualification passes and ordinary full updates
are at least5% faster without lost updates/debt or a numerical failure. Otherwise
discard it and return to the qualified centered recipe; no optimization sweep.
This is not a gate that can indefinitely postpone learning.

GPU jobs remain serialized under the host guard, persistent Restart=no /
KillMode=control-group services, bounded processes, expected RTX5080/580.178.04
and>=2GiB sampled Vulkan estimated headroom. Record allocation warnings; actual
API/numerical/hard faults/deadlines stop the job for review. No NVML polling,
recovery or blind retries. Heavy CPU preparation uses one CPU,2GiB, zero swap.

After reviewing this single trial, declare the next centered-CDP learning budget
toward **all five original games**, retaining three learner seeds, actual
initial/frozen controls, curves, videos and the unchanged DreamerV3 references.
Neither faster updates nor another early Pong screen completes the objective.
