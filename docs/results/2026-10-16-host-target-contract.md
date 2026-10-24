# Host-target contract: a concrete residency gap, not the whole JAX gap

October16,2026. [Declaration](../experiments/2026-10-16-host-target-contract.md)
· [Compact evidence](2026-10-16-host-target-contract.json).

**Kindle constructs and writes22.1472MB of behavior inputs on CPU per update;16.6464MB
is dense two-hot targets expanded from65,280bytes of scalar values.** Both current
RGB and CDP recipes have this path. JAX constructs its targets inside the compiled
device update. Its source also constructs dense one-hot arrays; this is not proof
that JAX avoids all dense materialization or uses a special compact loss.

No implementation or new speed measurement yet. Keep qualified native58aff452,
RGB46.199/JAX27.223ms,1.70×. Convolution feeding remains the larger unresolved
issue; production and Atari stay unchanged.

## Actual interfaces

| Behavior input group | Logical bytes/update | Current producer |
|---|---:|---|
| Four value/slow two-hot targets |16,646,400|CPU returns/encoding|
| Replay feature |4,915,200|CPU join of downloaded posterior state|
| Weighted action target and two weight arrays |585,600|CPU target construction|
| Total host-produced behavior inputs |22,147,200|Eight input writes|

The imagined feature is already copied on the GPU and is excluded from that
total. Each world program additionally has80 late-target inputs totaling
1,570,816bytes: reward, continuation, replay value/slow targets and replay
weights across16 time steps. Some reuse the same logical targets, so these are
input-write extents, not unique data or measured PCIe/DRAM traffic.

Posterior readback is5,242,880bytes. After existing GPU categorical means,
imagination readback is684,032bytes for RGB and716,800 for CDP, including actions,
continuations and CDP bonuses. The earlier25.1MB categorical-logit readbacks have
already been removed; do not optimize them again.

Both current programs use B32/T16/H15,255 bins and18 actions. All18 behavior
interfaces and160 world-target interfaces reconcile. The stochastic initial
input is physically[1024,16], not[32,512]; its joined feature is[rows,2560].
The original reader's flat-shape assertion fails and remains retained. Its
separate exact-physical-shape correction passes without changing runtime or
numerical gates.

## Dependency and numerical contract

- Targets use pre-update world/critic parameters and stay detached. The target
  normalizer updates exactly once, before actor advantages are formed.
- Imagined returns use arrival reward/termination at t+1 and the current value
  bootstrap. Predicted continuation already includes discount; do not apply it
  twice. The slow critic supplies the regularization target, not the bootstrap.
- Replay `last` cuts the trace but preserves bootstrap; terminal removes it.
  World replay weights retain the T/(T−1) correction and zero final row.
- Normalization uses F32 linear5/95 percentiles, EMA.01, no debias and minimum
  scale1. Value/advantage normalization are disabled in this pinned recipe.
- CDP's bonus belongs to the arrival reward. Preserve its scale and replay bonus,
  not just the RGB extrinsic-only path.

The seven existing Rust CPU distribution tests pass.36 extracted pinned-JAX
lambda-return cases match6,696 explicit-F32 scalar results exactly; six malformed
interface mutations reject. These are CPU source/alignment checks, not bit-exact
XLA/Vulkan arithmetic or full-model numerical qualification. GPU qualification
must also cover soft continuations, bin/percentile boundaries and changed state.

The actual JAX executable has302 entry parameter nodes with no corresponding
dense target arguments. Its reachable work includes target construction and a
normalizer sort;50 normalizer nodes and542 source-attributed two-hot-loss nodes
reconcile across1,595 reachable computations. Those are static graph counts, not
launch counts, memory traffic or measured cost.

## Time boundaries and retained responsibilities

The independent review replays the retained272-update trace, checking2,448
selected phase intersections. Target/report construction occupies1.437ms and
behavior-input preparation1.012ms in that instrumented run. Its8.086ms imagination
readback interval overlaps7.913ms of GPU execution: most is not transfer overhead.
The full trace adds2.315% overhead and changes intermediate synchronization.
Neither these intervals nor7.528ms total uncovered time are promised savings.

Production still refreshes CPU replay context from posterior states; the fixed
synthetic benchmark excludes replay sampling/context refresh. Removing that
responsibility would change production semantics. Keep it separate from the
unnecessary CPU feature join/target round trip and disclose the timing boundary.

Existing `DeviceCopies::copy_regions` already accepts checked raw buffer regions
and retains a completion fence. Reuse it; no parallel handoff framework is needed.
Device means, imagined features and slow-critic EMA already have GPU paths.

## Decision

Implement one resident target-generation path: returns/weights, percentile/EMA,
weighted actions and two-hot encoding, device replay-feature handoff and staged
world target writes. Preserve existing graph losses first. Keep normalizer state
resident with explicit checkpoint/diagnostic boundaries; do not introduce a CPU
normalizer workaround. Qualify binding extents, producer visibility, buffer reuse,
pre-update ordering and both RGB/CDP before one complete-update comparison.

CPU review finishes21:13:05UTC.
33 source seals pass; no GPU calls/builds/actions/updates or remaining worker.
No new GPU allocation, convolution/geometry sweep, prior-batching timing, numerical
relaxation, production promotion or learning restart follows from this audit.
Artifacts: `runs/host-target-contract-20261016.8BBXEPyc`.
