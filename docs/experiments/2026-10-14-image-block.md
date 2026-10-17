# Compare the complete first image-training block

October14,2026. Follow the [B32 attribution](../results/2026-10-14-batch-residual.md),
not another shape/precision/row-only sweep. Artifacts:
`runs/image-block-20261014.vPwv5OlK`. Production, defaults and Atari stay stopped.

Question: how much of the connected first CNN block's cost comes from its
heavy kernels versus the surrounding storage, layout, pooling and normalization
pipeline? The actual B32 learner has512 images,3→32 channels,64×64,5×5 stride1
padding2, then2×2/stride2 pooling, channel RMSNorm epsilon1e-4 and SiLU. Its
backward computes weight, bias and norm-scale gradients; pixels have no input
gradient. Retain that contract and a late incoming activation gradient.

First build a diagnostic-only public-API graph against the immutable qualified
cached backend. Explicit derivatives reproduce the existing autodiff expressions;
do not add a scalar loss/reduction or optimizer. Compare actual forward/backward
shader descriptors, layouts, rounding and gradient partitions to the retained
full learner before GPU work. Small CPU reference cases cover pooling ties,
padding and all parameter derivatives. Distinguish unavoidable benchmark
boundary/output differences rather than claiming exact subgraph equivalence.

Use a separately compiled JAX counterpart with the native numerical contract,
including BF16 convolution boundaries and F32 non-heavy work/parameter gradients.
Trace forward and late-gradient backward separately so the compiler cannot
invent a single-pass training lifetime. Compare reachable compiled HLO and
live tapes; count any extra fixture work explicitly. Do not substitute a
forward-only graph, omit backward outputs or assume ordinary JAX autodiff
preserves native BF16-gradient semantics. Any comparison with production JAX
must retain its different rounding/fusion contract separately.

CPU preparation is one CPU/2GiB/zero swap. No GPU allocation is started by
this declaration. Review the source/plan/reference contract first, then declare
exact binaries, numerical gates, repetitions and commands before one bounded
native/JAX block allocation. Restore unchanged full outputs after any diagnostic
work masks. Unknown or changed contracts stop for review; retain failures.
No new kernel implementation follows from logical bytes or dispatch counts.

Native stages use the existing command cache and preparation lifetime rules;
JAX uses its ordinary compiled device execution. Report complete and separated
stage timings with all live outputs. No unsafe Vulkan profiler path; a CUDA
capture, if justified, needs its own concrete question. A block result is not
a complete learner speedup, and no candidate is promoted without a subsequent
ordinary full-B32 comparison. No automatic batch/layout/tile/dtype sweep.

GPU work, when declared, is serialized under the persistent host guard with
expected5080, driver580.178.04,>=2GiB sampled Vulkan estimated headroom,
Restart=no, KillMode=control-group and bounded deadlines; no CPU quota on GPU
workers. Retain standalone allocation warnings; numerical/API/hard faults stop
the affected job, no blind retry. No recovery/NVML polling. Latest Meganeura
99e3de2 adds Xe2 rectangular BF16 support while preferring this5080's existing
16×16×16 product; no identified image-block fix. Bladee7c3f7b unchanged.

The first CPU build rejects two fixture API mistakes: the convolution call
omitted its separate kernel-width argument, and the compiler entry point is
`compile_with`. Preserve source/log and correct those calls before proceeding.
No GPU allocation or contract/tolerance change.

The first CPU plan also catches a genuine fixture-contract mismatch before
GPU use: explicit gradient nodes do not undergo autodiff's BF16 operand
conversion merely by enabling its option. Retain the21-dispatch plan/source;
insert the same explicit dy/input casts as production autodiff. The JAX/F64
contract already uses those rounded operands, so this corrects the native
fixture rather than changing the declared math. Re-audit its typed gradient
and512 reduction partitions before execution.

## Reviewed block allocation

CPU preflight passes: all23 final dispatch descriptors (shader, parameters,
grid, generated kernel, storage and precision flags) occur in the actual
learner; first-convolution output is BF16 and weight-gradient output F32 with
512 partitions. This is descriptor containment, not whole-subgraph/alias
identity. The block ends at F32 NCHW activation: next-convolution BF16 cast/
packing lies outside both arms. Incoming late dy is explicitly F32 NCHW.
The independent F64 versus CPU-JAX reference passes5,104 values in three
small cases, including ties. Source/backend/lock audits pass. A third CPU
failure is retained: Clippy requests fixed-size `as_chunks` readers.

Declare one serial queue, native then JAX, each<=300seconds, service<=660seconds.
Native binary351f4347 runs the three small cases (ordinary and late-dy stages),
then512×3×64×64→32-channel full block, same fixed modular inputs as JAX.
Small forward/bias/scale outputs must satisfy2e-5 absolute+2e-5 relative;
weight gradients use5e-4+2e-3 for rounded BF16 gradient operands. These bounds
are fixed before GPU work. Native staged/ordinary outputs must be bit-exact,
with NaN late inputs excluded from preparation. No optimizer is installed.

Native: eight warmups+64 measured complete calls, then eight+64 staged calls;
152 total block calls including qualification,128 measured. Every timed
cache reuse and unchanged full output is checked. JAX: the same three small
F64 cases, one actual-size qualification and eight warmups+64 measured staged
calls,76 total/64 measured. Its automatic VJP uses an explicit convolution
rule to retain BF16 dy/products and F32 master-weight gradients; pooling
gathers the first maximum rather than silently splitting tied gradients.
Large JAX/native outputs use the same declared bounds; all outputs must stay
finite and unchanged across timing. Both sides include forward tapes and
all three parameter gradients. HLO, plans, pipeline keys, full outputs and
timing samples are retained. No profiler instrumentation or per-dispatch timing.

Review native qualification before JAX: the queue stops on any failed guard,
result or output check, with no automatic retry. Any actual-plan discrepancy
invalidates the comparison. This measures a native-contract block, not the
different production-JAX rounding/fusion recipe or a whole-learner speedup.

The first GPU allocation stops during shader construction, before block
execution: the fixture's default context selected the AMD integrated adapter,
which rejects BF16. Device assertions had been placed after construction.
This is a fixture-selection failure, not an observed5080 failure. Preserve
binary/source/guard and its output-limited kernel-snapshot receipt; JAX never
starts. Review finds zero new kernel records/warnings in the failure interval.
The corrected fixture applies explicit env-based context selection and checks
5080/BF16/headroom before any shader construction. Its CPU plan must be exactly
unchanged. A new queue with fresh output directories may run the same bounded
qualification/timing after source review; no graph, shader, precision or gate
change, no completed work retried, no host/GPU recovery.

Corrected native0d2ab53f completes: all5,104 small F64 values pass, exact
ordinary/staged outputs with poisoned late inputs,152 block calls/128 measured,
ordinary6.218ms/staged6.379ms. Retain that completed arm without a retry.
JAX then fails on Python import: the local declaration file named `queue.py`
shadows the standard library. It has not run a block or initialized CUDA.
Preserve that source/guard; rename the declaration helper and verify the
standard-library import on CPU. Re-declare only JAX with unchanged script,
math, bounds and76-call budget in fresh output/guard directories. No native
rebuild or rerun, GPU recovery, tolerance relaxation or speed claim yet.

JAX now reaches its first small GPU numerical case and fails the unchanged
forward gate: maximum error.002734 versus F64; no timing or actual-size block
runs. CPU and native pass the same case. Do not waive the gate or repeat the
timing queue. The earlier precision study already encountered conversion
elision; inspect the actual compiled HLO rather than assuming casts preserve
rounding. One separately bounded120-second JAX numerical diagnostic runs this
same small block once unchanged and once with the previously used explicit
`reduce_precision(8,7)` at the forward convolution boundary. Save both
forward/backward HLOs and all outputs/errors; two block calls, no timing,
actions or optimizer updates. This tests one rounding hypothesis, not a
precision sweep or successful qualification by definition. All failures stay.

That diagnostic completes: original244/256 forward values fail, whereas
explicit rounding passes all872 forward/gradient values; weight gradients
are exact F64 in this case. Optimized HLO confirms the original round-trip
conversion vanished; the fenced control retains reduce-precision before bias.
Both custom calls use F32 convolution outputs; this is not the production-JAX
BF16 island or a pure storage-cost ablation. Re-declare only the uncompleted
JAX timing arm with that exact rounding correction, unchanged three-case and
actual-size gates,76-call budget and300-second bound. Retain original sources,
all failures and completed native timing. No precision relaxation, native
rerun, new shader or learning update. Check the reachable actual-size HLO too.

The fenced JAX control passes the three small cases and actual-size forward,
but fails the actual-size weight-gradient pointwise gate (max difference
.030655); no timing samples run. Keep both compiled HLOs, memory analysis and
the numerical failure. No cross-stack block speed ratio is established, and
no further JAX retry/tolerance change follows automatically. Review large
gradient accumulation/rounding separately before any further JAX allocation.

## Finish the native heavy-versus-dataflow attribution

The completed native block is independently qualified and its ordinary timing
is retained. One separately bounded180-second job answers the still-open
heavy-versus-other question on exactly that block, using existing diagnostic
masks. No shader/graph/math changes: add only a test command, require exact
full plan/pipeline keys and every saved output against the completed native
before and after. Optimizers remain absent. All real dependencies/bindings
and the command cache stay intact.

Two convolution operations versus21 other dispatches, four subsets, three
rotated orders, two warm+16 measured calls/subset;40 bracketing calls each
side with the first eight excluded. Including full restorations and initial
qualification,312 block calls/192 measured subset calls. Every measured cache
reuse must pass. Masked outputs are invalid; only full restorations are checked.
Report interacting allocations, not pure kernel duration, removable savings,
SM utilization or a JAX comparison. Source/CPU-plan identity review precedes
launch; same host guard/headroom/failure rules. No automatic optimization.

The native attribution completes with exact full restores:78.4% other work.
The separate actual-JAX B32 capture also completes with exact272 reports and
final tensors; its reachable image fusions take1.556ms encoder/.904ms decoder
in the instrumented trace. These are not native-equivalent block timings.
Before choosing a rewrite, declare one180-second unchanged-native follow-up
to rank the23 block operations by leave-one-out cost, plus a full control,
three rotated orders, two warm+16 measured calls/condition. Retain real
bindings/dependencies/cache and restore full outputs each round. This is
1,452 block calls/1,152 measured, zero updates/actions, with80 bracketing calls
included. No additive-savings or pure-shader-time claim. Source/CPU-plan/
actual-plan/pipeline/output identities must pass. Artifacts:
`runs/image-operations-20261014.cRZo7KQC`; service<=240seconds, same guard rules.
The suspected pooling/norm shaders match latest upstream99e3de2 byte-for-byte.
The GPU job passes exact runtime plans/outputs and all cache checks. Its first
CPU reporter incorrectly labels runtime dispatch indices from the differently
ordered CPU schedule. Preserve that report/source and relabel from the exact
saved runtime plan; no timing, output, shader or GPU rerun changes.
