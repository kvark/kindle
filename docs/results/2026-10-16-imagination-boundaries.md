# Imagination has a concrete data-movement mismatch

October16,2026. [Declaration](../experiments/2026-10-16-imagination-boundaries.md)
· [Compact evidence](2026-10-16-imagination-boundaries.json).

Native imagination uses **seven dispatches per step to assemble one grouped
RSSM input; JAX uses one fusion**. Across15 steps, these native chains write
747.110MB logically, versus125.829MB for the final packed inputs. All62,914,560
source-element addresses agree with the direct mapping. The six intermediate
outputs per chain are private, so this is a concrete composition opportunity,
not missing model work. It is **not a measured621MB DRAM reduction or speedup**.

The complete qualified update remains **48.068ms versus inclusive JAX27.223ms,
1.77×**. Keep isolated834241b6. This CPU-only study changes no backend, runtime,
production setting or Atari state. The remaining gap is not yet fully explained.

## The boundaries now reconcile

Current native non-world plans and pipeline descriptors still match the older
coarse capture exactly. Its8.740ms imagination pass covers all of the following:

| Native imagination region | Dispatches | Matrix products | Nominal MACs |
| --- | ---: | ---: | ---: |
| Parameter conversion |20|0|0|
| Recurrence and individual-state features |722|180|44.866G|
| Three rollout-output assembly trees |43|0|0|
| Batched reward, continuation and value heads |17|8|18.252G|
| Total |802|188|63.117G|

The180 recurrent products match JAX's twelve products per step. The already
audited small arithmetic difference is JAX's18→24 action-channel padding.
Head batching is already present; inventing it again is not an optimization.
Slow-value and behavior training remain separate native programs.

JAX's previously reported2.852ms is a **kernel sum**, not its complete recurrent
execution interval. The executable loop also performs45 indexed device-to-device
copies: one deterministic state, stochastic state and integer action per step.
All offsets, sizes, ownership and eight captured updates reconcile:

| Retained JAX recurrence boundary | Mean |
| --- | ---: |
|630 kernel durations |2.852145ms|
|45 copy durations,39.352MB |.055252ms|
|Union of kernel and copy activity |2.907397ms|
|First-to-last activity span |3.146679ms|

These are historical capture measurements, not a new JAX run. None measures SM
utilization. Native8.740ms still includes heads, conversions and different output
assembly; **do not divide it by any row of this table**. Native uniforms are
prepared outside its imagination graph, while JAX generates random samples in
the loop. Native exports time-major F32 features/one-hot actions; JAX's loop
exports BF16 states/integer actions. Shared post-loop JAX heads/targets/gradients
remain mixed in`call.15`; its2.342ms cannot all be assigned to imagination.

## The missing composition is exact and narrow

Each native input chain:

1. Concatenates three256-wide normalized inputs using two copies.
2. Repeats the768-wide result across eight RSSM groups using three copies.
3. Concatenates each group's256 recurrent-state values with that shared input.
4. Converts the resulting`[512,8,1024]`array to BF16.

JAX's actual`fused_concatenate.4`contains the same two logical concatenations,
one broadcast and a view, writing only the packed final array. No reduction,
normalization or activation needs to change to compose the native copies.
The proof checks all15 actual producer chains, exact source order, unique logical
writers,182 protected buffers and every consumer. A direct implementation must
still preserve the existing final BF16 conversion and qualify on GPU.

Rollout stacking is a separate issue: **it already uses balanced trees**, with
maximum depth four, not quadratic prefix concatenation. However, the15-state
and16-state outputs build separate overlapping trees. Together with the action
tree,43 copies write647.049MB logically. The15-state tree's leaves equal the
16-state tree's prefix exactly. Sharing the prefix or writing final slices
directly is feasible to investigate, but no shader/address implementation or
latency benefit is qualified here. Keep this distinct from the RSSM input chain.

## Next measurement, not an assumed fix

Use the now-defined recurrence region for one separately declared fixed-input,
valid-value native attribution: distinguish its180 matrix products from the
surrounding recurrence/materialization, retaining heads, allocations, barriers,
exact escaping values and complete outputs. Include an ordinary control and
do not interpret conditional substitution differences as pure kernel costs or
attainable full-update savings. Existing exact-copy diagnostic machinery can
be reused; no new convolution family or geometry sweep is justified.

This establishes whether to prioritize the proven copy composition or recurrent
matrix execution. Do not assume747MB of logical writes explains six milliseconds,
or count the same allocation twice. Convolution feeding, host targets and the
full parity goal remain open. Prior batching remains numerically unqualified
and must not be timed or promoted as a consequence of this study.

## Audit and failures

Four bounded CPU services use one CPU,2GiB and no swap; maximum observed RSS
is309.4MiB. All63 seals,62.915M address checks, six independent structural
mutations and three NVTX-label regressions pass. Independent raw SQL confirms
all360 retained recurrence copies and their times/bytes. Review finishes before
10:38:10UTC. There is no build, GPU call, synthetic update, action or worker.

The first independent SQL reviewer selected inline NVTX labels only and failed
its30-thunk count. The retained capture uses registered`textId`labels. Its source
and failure remain; the separate repair honors inline-text precedence and joins
registered strings. No captured data, numerical gate or GPU work changes.
Retain the initial source-path miss and accidentally truncated single-line plan
query; subsequent readers use bounded fields. The quadratic-stack hypothesis was
rejected by source inspection, before implementation.

Latest upstream remains Meganeura`f648a453`/Blade`4845d652`; no silent refresh.
Raw artifacts:`runs/imagination-boundaries-20261016.W7Up85O0`.
