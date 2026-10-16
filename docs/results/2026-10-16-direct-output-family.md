# Direct output saves too little to qualify

October 16, 2026. [CPU proof](2026-10-16-direct-output-contract.md) ·
[Fixed allocation](../experiments/2026-10-16-direct-output-family.md) ·
[Compact evidence](2026-10-16-direct-output-family.json).

**Reject this candidate.** Complete convolution operations improve 3.60% GPU
time and 3.65% queued wall time, below the declared 5% gates. Numerical checks
pass. No selector, retime, integration or full-update trial follows.

| All 23 convolution products, occurrence-weighted | GPU time | Queued wall |
| --- | ---: | ---: |
| Original native execution | 19.465 ms | 19.529 ms |
| Observer, original shaders | 19.120 ms | 19.189 ms |
| Observer, direct-output candidate | 18.432 ms | 18.489 ms |

The observer calibrates −1.77% GPU/−1.74% queued wall against native, within
the 5% aggregate gate. This is not per-kernel calibration. All transforms,
reductions, unchanged operations and 7,680 timing samples remain included.
Ten of 20 contracts regress. The 0.688 ms difference is a complete-operation
intervention, not pure epilogue cost or an additive full-update saving.

Keep isolated **e45b9c43**, complete RGB **43.197 ms** versus inclusive JAX
**27.223 ms**, **1.59×**. That complete-update baseline was not remeasured.
Production and Atari remain unchanged; the root-cause/parity goal remains open.

## What this tests

The existing matrix-store operation writes F32 accumulators directly when a
whole workgroup tile fits and its destination/stride are aligned. Seven actual
occurrences contain eligible tiles, covering 104.595M of 280.986M output values.
Tails, reduced outputs and packed dW's channel/tap permutation retain the original
scratch/scatter path. All 23 products are evaluated, not just eligible cases.

K loops, input staging, compensation, output rounding, geometry and shared-memory
allocation stay unchanged. This narrow output intervention does not supply the
library's operand-feeding implementation. The measured benefit does not justify
another output/tile search or explain the remaining 15.97 ms learner gap.

## Evidence and retained variation

- All 20 numerical pairs pass their original independent F64 gates: 2,690,432
  comparisons, exact zero interiors, changed-input and restoration checks.
  Independent full-output review verifies 774,861,888 bit-exact value pairs
  across numerical and timing arms, including signed-zero representations.
- Every original plan, pipeline, replacement source, binding, parameter and
  execution segment reconciles. Output bindings satisfy 16-byte alignment and
  allocation bounds. All 2,020 source receipts pass. Backend libraries are
  unchanged; the standalone observer was rebuilt, not the learner/backend.
- All 100 GPU guards pass; 15 allocation warnings are retained. There are 8,880
  standalone operation calls, zero learner updates and zero game actions.
  Minimum sampled Vulkan estimated-budget headroom is 15,828,123,648 bytes,
  not physical free memory or peak usage. No recovery or separate NVML polling.
- Native dX `1-32-3-64` shifts from **2.035327 to 0.457215 ms** between its first
  and last 32 samples. All samples stay; no cause, replacement steady-state
  estimate or explanation of the whole JAX gap is asserted. The largest other
  shift is 4.18%. The candidate/control decision still uses the original full
  cohort, not trimmed timings.
- The initial CPU reader's unmatched-parenthesis failure remains with its
  syntax-only repair. The final independent audit succeeds without another GPU
  run, widened numerical gates or changed performance thresholds.

GPU work ends **23:28:37 UTC**; terminal audit ends **23:29:00 UTC**. Both services
are inactive and successful. Artifacts: `runs/convolution-direct-output-20261016.93l7AixM`.

## Next root-cause boundary

Convolution's measured instruction/data-feeding overhead remains unresolved.
Do not assume the same diagnosis explains recurrent matrices: their retained
[complete counters](2026-10-16-jax-matrix-counters.md) show fewer native warp
instructions, similar tensor work, but 21,840 native versus 66,480 JAX workgroups.
These counts are not a causal occupancy or latency result. Existing standalone
matrix observations and full-JAX kernel times have different boundaries.

Before another kernel proposal, distinguish actual operand-transfer mechanisms
and work distribution from instruction count. Review the current generated code
and retained controls, not another tile, split-K or precision sweep. The failed
output candidate, prior split-K result and rejected feeding families stay closed.

Source review also rules out a purportedly new global tensor-load experiment:
the October15 rejected family's `spatial.comp` already calls
`coopMatLoadTensorNV` on its storage-buffer image and weight operands, not on
shader-managed shared arrays. Its prepared layouts and geometry differ from
the current native path, but merely exposing the same API is not a new fix.

The next useful measurement boundary is the **current native model execution**.
Existing native convolution/matrix counters use standalone synthetic operands;
JAX's corresponding counters come from its complete update. The earlier captured
five-operation group ruled out input values for that historical group, not for
every current kernel or execution context. First prove complete native range
membership and an observer that preserves original commands/dependencies. Do
not infer a latency ratio from the existing counters or repeat the old group.
