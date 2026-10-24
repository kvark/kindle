# Prior batching: real-input forward attribution

October16,2026. The [actual-world qualification](../results/2026-10-16-prior-world.md)
remains failed. Its first prior product changes eight F32 partial reductions to
one tiled accumulation; both arms retain BF16 rounding. A CPU simulation using
the saved inputs changes15 RGB/17 CDP BF16 values, but is not a Vulkan simulator.

Diagnose this frozen forward boundary only. Use the unchanged, sealed standalone
serial/batched B32/T16 plans and qualified834241b6 libraries. Export all eight
actual prior parameters and16 deterministic states from each retained RGB/CDP
world checkpoint. Posterior-logit placeholders are zero and must never contribute
to an executed dispatch. Prove the exact versioned prior-logit dependency closure,
time order, BF16 boundaries and corresponding actual-world shader contracts on
CPU before execution. Keep original allocations/pipelines/group boundaries;
execute only closed forward cuts within those groups, not unrelated backward
seeds or KL work. This is not an ordinary full-world execution.

One bounded process runs three forward conditions for each workload: serial,
batched, and batched with its first BF16 product replaced by the concatenated
serial products. Capture all three layers' matrix/bias/normalization/activation
outputs, serial first-layer partials/reduction and final logits at their live
boundaries. Replacement uses an exact-byte GPU copy after producer completion,
before any consumer, with original automatic barriers and immutable parameters.
Record raw ordinary and substituted values separately; never silently overwrite
the evidence. No backward pass, optimizer update, timing or game action.

Verify plan/pipeline identity, all parameter bytes, output finiteness, headroom,
captured byte extents and exact copied product. Independently compare saved
first-product operands with F64 arithmetic/explicit BF16 and the fixed CPU
reduction hypothesis; retain all residuals rather than changing tolerances.
Measure whether substitution removes downstream differences, without requiring
that hypothesis to succeed. A forward attribution is not qualification of all
world gradients, learning parity or a replacement for the failed cohort.

CPU preparation/review:1CPU/2GiB/no swap, bounded persistent services. GPU host
guard:180s process/240s service, no CPU quota, expected5080/580.178.04 and>=2GiB
sampled Vulkan budget-minus-usage. Record allocation warnings; stop API, identity,
nonfinite, hard-fault or deadline failures for review. No retry, compiler/backend
rebuild, geometry/split/precision sweep, production promotion or Atari restart.
Upstream remainsf648a453/4845d652. Preserve the initial CPU `inspect.py` stdlib
shadowing failure and filename repair. Artifacts:
`runs/prior-real-inputs-20261016.At0oVqjy`.
