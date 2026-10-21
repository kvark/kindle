# Raw-word shared storage leaves the instruction deficit unchanged

October 21, 2026. [Compact evidence](2026-10-21-shared-word-view.json) ·
[Preceding contract](2026-10-21-shared-read-residual.md).

**Reject the representation-only fix.** All nine actual typed convolution
occurrences pass numerical qualification, but changing shared storage from
`vec4<bf16>` to `vec2<u32>` changes none of their warp, ALU, LSU or tensor
instruction counts. The extra shared-read relation survives exactly. Do not
integrate this view into Naga, time it, or search neighboring packet formats.

This is a useful negative result, not parity: the retained isolated complete
RGB update remains42.808ms versus inclusive JAX26.066ms,1.642×. Production and
Atari are unchanged; the five-game objective remains unfinished.

## Matched controls and measured outcome

The observer uses the qualified predicate candidate79bc7cec's actual modules,
binding numbers, inline64-byte uniforms, default subgroup settings, full B512
grids and original split partitions. It saves the actual words submitted to
`vkCreateShaderModule`. The candidate is only the reversible, offline-validated
shared-storage view; tiling, matrix types, indices, stride operands, arithmetic,
barriers and global gathers stay unchanged. No Naga/Blade/backend rebuild.

After all nine numerical checks pass, nine separately guarded paired captures
complete. Standalone control sums reproduce **all35 CTA/warp/ALU/LSU/tensor
anchors across the seven retained model regions**. Two regions contain adjacent
dX/dW dispatches; reconciliation uses their full membership, not invented
per-kernel counts. Both arms use the same observer and execution boundaries.

| Nine typed occurrences | Control | Raw-word view |
| --- | ---: | ---: |
| Warp instructions |8,760,672,768|8,760,672,768|
| ALU-pipe instructions |5,520,477,696|5,520,477,696|
| LSU-pipe instructions |1,017,805,824|1,017,805,824|
| Tensor-pipe instructions |104,857,600|104,857,600|
| Shared-read wavefronts |635,331,212|635,329,154|
| Reported read-bank conflicts |349,324|347,266|
| Wavefronts minus conflicts |634,981,888|634,981,888|

Each occurrence independently retains its instruction anchors and derived
shared-read count. The small wavefront difference equals the conflict-counter
difference exactly; it does not remove the preceding209,715,200 excess relation.
All1,566 metric values reproduce offline from the nine saved counter images.
SDK/bridge loaded-library paths reconcile; no ranges or trace bytes are dropped.

Equal aggregate counters do **not** prove identical machine instructions or
physical shared-memory layout. Native SASS remains unavailable. This rejects
the specific prediction that the raw-word view removes the observed overhead;
it does not explain the historical/current difference or establish driver-only
causality. Neither profiler replay duration nor instruction counts are ordinary
latency. No timing or speedup claim follows.

## Numerical scope and accounting

The existing independent F64 fixtures retain two active images in the full
B512 grids, including dW partials and exact empty-partition zeros. Both processes
check full paired outputs, a −0.5× input change, restoration, immutable inputs/
parameters and256-byte buffer tails. Gates and BF16 rounding are unchanged.
These structured fixtures are not dense real-model gradients or learning parity.

Across numerical and capture processes:3,372,544 F64 comparisons,
742,033,920 empty-output zero checks and372,703,232 independently paired output
values pass. Counts include repeated qualification in the capture processes;
they are not that many distinct test inputs. There are18 passing guards,
zero new allocation warnings,450 counter passes and1,008 main-kernel dispatches;
zero learner updates, game actions or ordinary timing samples.

GPU work finishes11:27:20UTC; independent CPU publication checks281 receipts
at11:29:31UTC. All workers are terminal. CPU preparation/review retains one CPU,
2GiB and no swap; GPU controllers retain12GiB/no swap without a CPU quota.
Upstream main remainsa75c38e; no newer fix was omitted.

Before any capture, inspection corrects an unexecuted audit's assumed SDK
library names to the actual graphics library. Original scripts/receipts remain;
versioned capture readers change no observer binary or completed numerical
cohort. The earlier absent-ArrayStride/duplicate-type failures remain in the
CPU contract. No GPU retry or tolerance relaxation occurs here.

## Next decision

Keep79bc7cec and the existing production state. Do not turn this null result
into a compiler project or a format/geometry sweep. The next diagnostic design
should separate cooperative operand loads from accumulator/shared-output
handling, with independently known values and the actual matrix contract,
before allocating another GPU experiment. The saved aggregate counters alone
cannot identify which source operation acquires the extra reads.

That source-level question remains distinct from cuDNN's larger pass-specific
reuse and multistage feeding, which this experiment did not attempt to match.
Neither is a reason to claim that native and JAX execute the same kernels.

Raw declaration, observer, fixtures, guards, submitted modules, counters and
reviews: `runs/shared-word-gpu-20261021.xjH2gChb`.
