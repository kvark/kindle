# Extra shared reads follow operand preparation, not repeated multiply-adds

October21,2026. [Evidence](2026-10-21-load-mma-isolation.json) ·
[Preceding null result](2026-10-21-shared-word-view.md).

**A calibrated single-shader diagnostic separates operand loading/preparation
from repeated matrix multiply-add work.** Doubling multiply-adds doubles tensor
instructions but changes neither LSU instructions nor shared-read wavefronts
minus conflicts. Doubling loading iterations increases the derived shared reads
as predicted. This narrows the investigation; it is not a kernel fix or timing
result. Retain79bc7cec/RGB42.808ms versus inclusive JAX26.066ms,1.642×.

## Why another ordinary model capture could not answer this

The nine actual typed convolution modules have four cooperative loads and two
multiply-add sites per inner iteration. Their only ordinary shared read is in
the output loop; the cooperative accumulator store is also outside reduction.
Source-derived logical work, including actual split intervals and output masks,
exactly fits every saved historical/current occurrence:

- Historical derived shared reads: `4L + O` =425,266,688.
- Current derived shared reads: `6L + O` =634,981,888.
- `L` is104,857,600 logical matrix loads; `O` is5,836,288 logical32-element
  output row groups. These logical groups do not assert local-ID/warp mapping.

But all nine also have `L = 2M`, where `M` counts logical multiply-adds.
Therefore `6L + O` and `4L + 4M + O` explain the existing measurements equally
well. Exact rational rank analysis confirms the ambiguity. Repeating those
captures could not resolve it. Thirty-six incorrect count models reject.

## Fixed diagnostic, one shader binary

Use the actual16×16 BF16 operand/F32 accumulator contract, padded shared stride,
128-invocation workgroups and four logical output tiles. The original four load
expressions and accumulator-store expression reconcile verbatim. Global
convolution gathers are replaced by known input matrices; this is not an actual
convolution performance test. Grid stays[1,1,256], inline uniforms/default subgroup
settings stay fixed, and the same submitted module serves all settings.

Three predetermined settings independently vary loading iterations and repeated
multiply-adds on already loaded operands. No tile, dtype or compiler sweep.

| Loading iterations / MMA repetitions | Logical loads | Logical MMA | Tensor instructions | Shared reads minus conflicts | LSU instructions |
| --- | ---: | ---: | ---: | ---: | ---: |
|1 /1|4,096|2,048|4,096|32,768|64,512|
|2 /1|8,192|4,096|8,192|57,344|104,448|
|1 /2|4,096|4,096|8,192|32,768|64,512|

The first setting passes its predeclared model-pattern gate,32,768=`6L+O`.
The three-setting system has full rank and gives local coefficients
`6L + 0M + O` in both repeated capture ranges. Thus the additional repeated
multiply-adds do not cause additional shared reads in this diagnostic. The
once-per-output term also remains consistent with the source accounting.

This supports investigating operand loading/preparation, including possible
first-use operand conversion. It does **not** identify a particular SASS
instruction or distinguish A from B, staging from cooperative-load lowering,
or first-use preparation from the load itself. Coefficients are not measured
per-source-instruction costs. The old/current-driver comparison remains
confounded; no driver-only causality or native/JAX machine-code identity claim.
cuDNN's larger reuse and multistage feeding remain a distinct unresolved issue.

## Qualification and next step

Offline Naga/SPIR-V checks, independent integer/F64 and ordered-F32 CPU oracles
pass. All three numerical settings pass before capture. Every GPU output equals
its reference exactly, including a−0.5× input change/restoration, immutable
inputs/parameters and256-byte tails. Recorded membership verifies four32-lane
subgroups without assuming their mapping to local IDs. Both paired pipelines
contain the same module: they check repeatability, not old/new performance.

Six guards pass with zero allocation warnings;3,145,728 F64 comparisons,
1,572,864 paired output values and196,608 membership records include repeated
qualification during captures. All522 metrics reevaluate offline across150
passes. There are336 primitive dispatches, zero learner updates/game actions/
ordinary timing samples. Two diagnostic helpers build; no backend/compiler edit.
GPU ends11:42:55UTC; independent141-receipt audit ends11:44:43UTC. All workers
are terminal, with no failed/repeated GPU cohort or relaxed gate.

Next design a role-asymmetric known-value probe: vary A and B operand reuse
independently while retaining the validated matrix contract and recording.
First prove its outputs and independent count dimensions on CPU. This is
source attribution, not permission for another storage-format/geometry sweep
or a Naga feature project. Production/Atari remain unchanged; five-game mastery
and complete backend parity are still open.

Raw source, declarations, source accounting, CPU proofs, submitted modules,
guards and counters: `runs/shared-read-accounting-20261021.boSjzSV5`.
