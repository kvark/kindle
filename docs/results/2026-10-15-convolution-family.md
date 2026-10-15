# The fixed workgroup convolution does not generalize

October15,2026. [Declaration](../experiments/2026-10-15-convolution-family.md) ·
[Data and every shape](2026-10-15-convolution-family.json) ·
[Backend plan](../backend_training_performance_plan.md).

**Reject this family implementation.** Across every actual forward/input-gradient
contract in RGB12M/B32, the fixed shared lowering is41.29% slower than native,
despite passing numerical checks. The earlier40% single-derivative win does
not justify Naga/Blade integration or a complete-update trial.

## Complete-operation result

Thirteen distinct contracts represent fifteen graph occurrences: the HW8/64→64
forward and dX each occur twice. Posterior/world are one staged plan, not two
executions. The native heavy shader/parameters/grid/pipeline key for every case
matches the actual current world plan. Each operation includes all GPU packs,
dependencies and, for the candidate's forward pass, BF16 output conversion.

| Family, weighted by actual occurrences | Native | Candidate |
| --- | ---: | ---: |
| Eight forward convolutions |7.824ms|10.896ms|
| Seven input gradients |5.009ms|7.235ms|
| Total |12.833ms|18.131ms|

GPU and corrected queued-wall totals both regress41.29%, failing the20% reduction
gate. These are standalone operation sums, not complete-update latency, additive
savings or the eight unchanged weight gradients. Native and candidate retain
their own external NCHW/BF16→BF16-forward/F32-gradient interfaces. No pooling,
bias/activation fusion or learning workload is silently removed.

Ten of thirteen distinct contracts regress. The first RGB forward and final
decoder dX take5.61×/4.42× as long. Their three input channels are padded per
filter tap to32 in this proof, wasting contraction work. But several aligned
cases also regress by~71–90%: this is not only an RGB boundary problem. Shared
typed arithmetic remains appropriate; copying one operation's workgroup schedule
across the family does not reproduce the library's multistage data feeding.
No post-hoc faster-shape selector or geometry sweep follows.

The native CNN1 dX mean is2.512ms rather than the earlier2.149ms. Its128 samples
start around6.13ms and end around2.15ms; every sample is retained. Code-generation
sources match the previous control, but no clock/cache/driver cause is established.
Do not claim a new steady-state48.6% improvement from this noisy reference or
replace its mean with a convenient tail. This variation does not rescue the
family gate; no timing repeat was used to select a result.

## Qualification and limits

The same typed source is compiled for BF16/F16; only BF16 runs on GPU here.
All503,808 complete numerical outputs pass the predeclared F64/rounded-BF16
gates. Worst relativeL2 is1.731e−6 for dX and1.197e−4 for rounded forward;
zero pointwise failures. All uploaded bits are independently regenerated,
changed-input/restores pass,257,949,696 native/candidate output pairs are
byte-identical on the dyadic fixture, and416 independent F64 probes pass.
CPU coverage checks33,177,600 addresses/503,808 output locations;78 shader
variants compile. This is not qualification of arbitrary shapes/F32/F16 training.

All27 guards pass; five standalone allocation warnings remain.3,835 operation
calls/11,088 shader dispatches, zero game actions/learner updates/recovery.
The initial GLSL template-expression compilation failure remains with its source;
using the same precomputed integer dimension fixes it before any GPU job.
The timing reporter initially included command recording in the native field
named queued wall time. A separate CPU audit subtracts the saved recording time
from native samples; both original reports remain, with no GPU rerun or changed
gate. The preceding qualification's corrected queued-wall gain is39.71%, still
passing. Independent publication/source audit passes; no worker remains.

Artifacts: `runs/convolution-family-20261015.ZI0VhKI6`. Keep current isolated
native28864210; full B32 remains60.041/26.632ms,2.25×, not remeasured/promoted.
Next address measured device-side categorical decode/target dataflow, preserving
symmetry, terminal alignment and old-weight consumers. Kernel feeding and device
dependency costs remain open; host work alone does not explain the remaining
gap. Production, Atari and the original root-cause goal remain unchanged/open.
