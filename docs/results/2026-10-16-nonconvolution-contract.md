# The training prior is unnecessarily time-unrolled

October16,2026. [Declaration](../experiments/2026-10-16-nonconvolution-contract.md)
· [Compact evidence](2026-10-16-nonconvolution-contract.json).

The remaining gap includes a concrete graph-construction mismatch: native
training evaluates the RSSM prior independently at each of16 timesteps;
upstream evaluates it once over the complete posterior sequence. The same
1.107G nominal MACs become **144 native matrix products versus9 JAX products**,
including input and weight gradients. This is unnecessary fragmentation, not
extra model capacity or a need to parallelize causal histories.

This CPU-only audit does not measure its latency or qualify a replacement.
Keep isolated834241b6: complete RGB12M/B32/T16 remains **48.068ms versus
inclusive JAX27.223ms,1.77×**. CDP45.505ms is a different workload.
Production and Atari remain unchanged; no GPU allocation occurs here.

## What the execution graphs establish

The current imagination, slow-value and behavior plans/pipeline descriptors
match the [older coarse capture](2026-10-15-whole-update-attribution.md) exactly,
including constants and ordered dispatches. Its cached GPU-pass intervals are
8.740ms, .504ms and3.011ms respectively. The world program has changed since
that capture; its old chunk times are not relabeled as current measurements.
Serialized plan groups are empty because runtime owns them separately; they
do not imply zero execution groups.

All28,064 kernels in the retained eight-update JAX capture map to their CPU
launch thunk and executable-HLO context. Counts and summed time reconcile
independently for each update. The capture also matches the current JAX control's
940 static fusion bodies,112 custom-call configurations and reachable heavy
work. The three real loops are16 posterior steps,15 imagination steps and16
RSSM backward steps; later root work is not part of the preceding loop.

| JAX execution context | Kernels/update | Summed kernel time |
| --- | ---: | ---: |
| Posterior recurrence |1,440|1.530ms|
| Imagination recurrence |630|2.852ms|
| RSSM backward recurrence |544|1.391ms|
| Mixed batched heads/targets/gradients, `call.15` |317|2.342ms|

All other kernels remain in the full22.573ms kernel sum. `call.15` crosses
model components, so its2.342ms is not a policy-only or imagination-only time.
Library kernel names inside a command-buffer thunk can cover several products;
the audit does not invent individual-operation timing from those names.
Native GPU-pass intervals include dependency/issue gaps; JAX kernel sums do not.
**Do not subtract these columns into a full-update savings budget.**

The imagined recurrence itself is closely matched: both execute180 matrix
products,12 per step. Native nominal work is44.866G MACs versus JAX44.890G;
the entire difference is JAX padding the18-action projections to24. JAX's
recurrence kernel sum comprises2.106ms matrix kernels and.747ms other kernels.
Native's8.740ms imagination pass also includes the batched reward/continuation/
value heads and output assembly, so it is not a measured3× recurrence ratio.
This gives a specific common region for future native attribution; it does not
yet identify whether its matrix implementations or surrounding materialization
dominate the native cost.

## The prior mismatch is proven, not inferred from names

Native training calls `Prior::forward` inside the observation loop. Its three
layers operate on32 rows, with16 forward,16 input-gradient and16 weight-gradient
products per layer. Upstream `RSSM.loss` calls `_prior(feat['deter'])` after
`observe`; the compiled graph contains three forward and six derivative
products over512 rows. Nominal arithmetic agrees exactly.

The native proof follows versioned buffers and gradient accumulation chains,
including in-place addends and hidden fusion reads. None of these144 products
or their prior parameters occurs in either exported posterior-state dependency
closure. All occur after the1,043-dispatch posterior prefix. The prior is
required to sample the next imagined state, but **not to advance observation**;
only training-prior work can be time-batched this way.

Across the complete update, native has716 matrix products versus557 in JAX.
This one difference accounts for135 of the159 extra products, not135 units of
equal latency. Total non-convolution nominal work differs only4.33%
(112.530G versus107.864G MACs), and total heavy work still differs about1.40%.
The prior has only1.107G MACs; these counts cannot explain away the whole gap.

## Shortcuts that are already present, and boundaries that remain

- Native reward/continuation/value heads already batch all16 imagined states;
  world image/reward/value heads already batch the16 observed timesteps.
  Grouped matrices already use cooperative hardware. Basic head batching or
  enabling tensor products is not a new fix.
- Native behavior training recomputes the value head after target generation;
  JAX's monolithic differentiation can retain that forward tape and combine
  replay/imagined head work. Both implementations also evaluate policy heads
  for sampling and training. Do not count all actor recomputation as a native-
  only defect. Cross-program tape reuse needs parameter-version, rounding,
  feature-order and gradient-boundary proof before removing work.
- Target construction, percentile normalizers, replay refresh and relevant
  state transfers remain required work. The older native capture measures
  about1.439ms host targets and.120ms host decoding. Its9.849ms readback span
  contains9.669ms GPU-pass overlap: that span is not9.849ms of copy overhead.
  No fully GPU-resident learner claim follows.

## Next decision

Time-batch the independent **training prior and KL**, using the existing world
head grouping pattern. Keep the causal core/posterior and imagined prior
sequential. Retain masks, free-nat reductions, stop-gradients, stochastic input
ownership, visible rounding, and staged late-target isolation. A longer weight-
gradient reduction can change numerical order and kernel selection: independent
forward/raw-gradient checks and a whole/staged boundary check precede one matched
complete-update comparison. This audit authorizes no implicit GPU queue.

This is one justified graph repair, not a parity plan by itself. Native
convolution feeding still has [measured instruction overhead](2026-10-16-word-staging.md),
and matched imagination arithmetic now gives a narrow region to attribute next.
Host residency and tape reuse are separate hypotheses, not additive promised
savings. No packet/tile sweep or compiler feature project is reopened.

## Audit and retained failure

Six bounded CPU services use one CPU,2GiB and no swap. Review finishes
07:13:16UTC;51 source/input seals pass. No native build, device initialization,
GPU dispatch, synthetic update or game action occurs. Upstream remains
Meganeura f648a453/Blade01be8446; production pins do not change.

The first observer conserved kernels/times but leaked loop scope into later
root nodes, producing invalid phase labels. Its source and result remain.
The separate repair passes18 regression checks, reproduces six original
failures, and passes an independent body-membership and per-update SQL audit.
No GPU retry or numerical gate relaxation was needed. Raw artifacts:
`runs/nonconvolution-contract-20261016.Z0XELHL5`.
