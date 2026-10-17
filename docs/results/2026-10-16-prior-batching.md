# Prior batching is implemented, but not qualified

October16,2026. [Declaration](../experiments/2026-10-16-prior-batching.md) ·
[Evidence and all failures](2026-10-16-prior-batching.json).

The isolated12M training-prior boundary compiles to **9 matrix products instead
of144**, and168 dispatches instead of1,181. All eight serial/batched output and
raw-gradient comparisons pass the declared limits. The causal posterior and
imagined recurrence are unchanged. Boundary-program counts are not full-update
counts or measured speedups.

**Independent numerical qualification fails in both control and candidate.**
Actual whole/staged-world qualification and all full-update timings are unrun.
Do not promote, relax the gates or infer a batching defect from this result.
Keep834241b6: complete RGB48.068ms versus inclusive JAX27.223ms,1.77×;
CDP45.505ms is a different workload. Production/Atari stay unchanged.

## Numerical result

One bounded GPU process performs16 raw backwards: four size/free-nat cases,
two deterministic input variants, serial and batched graphs. Parameters remain
unchanged; zero optimizer updates/actions. One guard passes, no allocation
warnings/API failures/hard faults. Sampled Vulkan estimated headroom is
16,475,947,008B, not physical free/peak VRAM. No worker remains.

The F64 forward/analytic-gradient review checks27,564,960 values, including
pairs. Tiny B3/T3 free-nats0 and B3/T4 free-nats1 pass completely; the latter
has six clamped/six active states per variant. Fully clamped12M gradients are
exactly zero. Every paired comparison passes, but some scalar reductions and
accumulated parameter gradients differ: no bitwise trajectory-parity claim.

Fourteen independent-reference array checks fail symmetrically in both arms.
The second12M variant has748 logits outside oneBF16ULP+1e-5, in ordinary and
fully clamped cases; maximum error.00390625. Ordinary prior0/prior1 weight
gradients and one prior0 bias also exceed individual2e-5+.02*abs(reference)
envelopes. All their relativeL2 errors stay below.000350, versus the separate
.01 gate. A passing norm gate does not override failed pointwise checks.

## Precision diagnosis, not an exoneration

The oracle rounds BF16 matrix operands/results but computes RMSNorm/SiLU in
F64; native keeps those operations in F32. Its analytic derivatives pass40
finite-difference checks accounting for detached branches; four BF16 tie cases
pass. Correct derivatives do not establish correct finite-precision tapes.

A separate CPU diagnosis uses F32 normalization/activation on the saved data.
Differences below5e-7 at the first activation move four values across BF16
rounding boundaries and affect four rows downstream. Failed final-logit
envelopes fall748→408, not to zero. Both references reproduce all262,144 logits
of the first variant exactly. This establishes rounding sensitivity, not that
every native discrepancy is harmless or that the remaining408 are explained.
Original references, tolerances and failures remain; no GPU retry or replacement
input cohort. Resolve this precision-reference discrepancy before progressing
this candidate. The broader convolution-feeding/parity goal stays open.

## Source and upstream

Four isolated Kindle files change: world graph, two test modules and module
registration. `KINDLE_BATCH_TRAINING_PRIOR=1` uses the existing head grouping,
per-state free-nat clamp and asymmetric stop-gradients; disabled retains the old
graph. No backend kernel/model/precision/exploration/checkpoint change.
Native`aca80566` remains unqualified; retain834241b6.

Formatting, release Clippy/build and CPU interface/posterior-dependency tests
pass for lengths1/3/4/16. Preserve the initial nonexistent`read_parameter`
method error and invalid length-one training-config test. Repairs use
`read_param` and a legal two-step configuration for the one-step chunk.
A separate Cargo cache/manifests establish source identity;2,271 seals pass.
CPU work uses1CPU/2GiB/no swap; GPU has no CPU quota. Full sources/logs:
`runs/prior-batching-20261016.w4yGvAmq`.

Blade upstream4845d652 now includes workgroup cooperative matrices and BF16
capability alignment; Meganeura f648a453 is unchanged. Review that existing
support for the separate feeding problem before proposing new compiler work.
No silent repin, Naga project, convolution sweep or learning restart follows.
