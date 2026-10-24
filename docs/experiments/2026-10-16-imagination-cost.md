# Valid-value attribution of the native imagination recurrence

Declared October16,2026 before source changes, build or GPU work. Follow the
[CPU boundary audit](../results/2026-10-16-imagination-boundaries.md), not the
unqualified prior-batching candidate. Keep qualified834241b6 RGB12M/B32/T16,
BF16, original seed/checkpoint, word staging/image views/device means and cached
recording. No production or learning change.

The actual imagination graph has802 dispatches. Select its722 recurrent/
individual-feature dispatches:180 matrix products and542 other operations.
Leave20 parameter conversions,43 rollout-assembly copies and17 batched-head
dispatches unchanged. Four conditions: ordinary, recurrent matrices substituted,
recurrent non-matrices substituted, both substituted. Preserve original shaders,
allocations, groups/barriers and all externally consumed/protected values.
The copied values come from an original fixed-input execution. This diagnoses
conditional execution costs, not pure kernel latency or achievable update gains.

Reuse the retained integer-copy diagnostic and version verifier. Before build,
independently prove selected products, every hidden read, protected buffer and
escaping version from current saved plans. Snapshot every selected output;
require total snapshot size<=4GiB. Copy only escaping values during closed
substitution. Poison discarded snapshot values with0x7fc17fc1, execute the
all-output substitution and verify live boundaries/finite complete outputs.
Restore snapshots by an ordinary execution and verify exact restoration before
closed-substitution validation. No invalid-input zero-work mask.

An isolated observer build may add only the existing diagnostic hooks and this
test to the qualified source. Keep a distinct Cargo target/source copy. Check
CPU verifier tests, formatting and Clippy, the integer-copy shader, full emitted
plans/pipelines/groups, exact292 initial/first/before-after tensors, first
reports/targets and all six full imagination outputs. Retain failures and tails.

Two serialized guarded GPU jobs: the unchanged five-size integer-copy smoke
(3,906 exact words,35 standalone executions), then one initializer update and
fixed imagination attribution. No game actions or replay resume. Attribution
uses16 warm+64 measured pre-snapshot ordinary replays, then eight16 warm+64
measured blocks in order0,1,2,3,3,2,1,0:576 measured/720 warm+measured replays,
plus13 validation/capture/restoration replays,733 total fixed imagination replays.
The initializer's imagination execution is separate. Count actual work in the
receipt; no optimizer runs during replay.

Retain every timing sample. Normal-with-snapshots mean must stay within5% of
the fresh pre-snapshot mean and within10% of retained8.740ms; otherwise cost
qualification fails without retiming. Sample GPU-pass durations and queued wall
time separately. Original allocations/groups must remain unchanged; do not
compare a new heavily instrumented schedule with the old ordinary boundary.
JAX's2.907ms activity/3.147ms span excludes native's separate heads/assembly;
no direct ratio or additive savings budget follows.

GPU jobs120s/600s, service900s, Restart=no,KillMode=control-group, no CPU quota.
Host guard expects5080/580.178.04/boot3e89d55c-a9e5-472f-a18a-06508c5bafa7 and
>=2GiB sampled Vulkan budget headroom. Record allocation warnings; other API,
numerical, identity, deadline or hard faults stop for review. No separate NVML
polling or recovery action. CPU setup/build/review use one CPU,2GiB,zero swap,
bounded persistent services; none runs during GPU timings. Rechecked upstream
f648a453/4845d652 is unchanged. No automatic follow-up allocation, geometry/
precision sweep, numerical relaxation, prior-candidate timing or Atari restart.
