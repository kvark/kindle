# Explain the full native/JAX gap

The active objective is to root-cause the full wall-clock gap, not merely find
another useful optimization. The preceding split-tile change is progress, not
completion. No game learning or old queue restart.

Compare RGB12M/B8/T16/H15 on the RTX5080 first; CDP/B32 remains a transfer check,
not an equal-work JAX comparison. Current native Meganeura5a49332 includes
published main20af998; Blade663afd2 is based on main7909fc1. Both upstream heads
are rechecked October12 and unchanged. Preserve all historical pins, captures,
failed profiler exits and user work.

## Evidence required

1. Audit actual model shapes, losses, sequence/context and imagination work,
   optimizer, precision, input residency, synchronization and timed boundaries.
   Identify extra/repeated native work rather than calling every difference a
   backend defect. Export JAX's compiled graph and native selected pipelines.
2. Reconcile complete wall time into non-overlapping GPU execution and host /
   submission / synchronization intervals, retaining instrumentation effects.
   A sum of kernel families from different clocks or workloads is not a full
   explanation. Invalid Systems Vulkan timestamps stay invalid.
3. Attribute expensive GPU work to concrete shapes, storage/arithmetic and
   selected kernels; compare JAX fusion/library execution. Test the outstanding
   causal hypotheses with bounded, independently checked controls rather than
   a tile/batch/precision sweep. Quantify duplicate computation, layout/casting,
   transfers and target/EMA work as well as matrix/convolution cost.
4. Publish an accounting of the complete measured difference, with explicit
   residual and uncertainty. Do not close the goal while a material bucket is
   merely named or inferred. Root-cause completion does not require implementing
   every remedy or pretending the algorithms are bitwise-identical.

## First bounded checks

Reuse the existing exact-coverage Nsight captures for a CPU-only stage/kernel
inventory and API-wait analysis. Those Graphics captures have failed shutdown
guards and remain diagnostic evidence; no profiler crash suppression or blind
capture retry. Use event streaming for large plans, one CPU/2GiB/zero swap for
analysis and compilation.

One fresh JAX BF16 resident control and one host-refreshed-input control may run
after the source audit, each16 warmup+256 measured updates,600-second bound.
Keep actual input values, initialization and seeds equal between those controls;
export compiled HLO, parameter shapes and metrics outside the timed region.
Measure the public wrapper, explicit completion and metric materialization
separately so excluded asynchronous work is visible. Require the expected GPU
and >=2GiB sampled budget headroom. No driver changes or NVML polling.

Native follow-ups will be declared here after reviewing the existing traces;
prefer minimally perturbing CPU/submission instrumentation on the current
qualified source. Retain controls and exact native metric/tensor comparisons.
All GPU checks use the existing host guard, serialized persistent services,
Restart=no, KillMode=control-group and bounded deadlines; no CPU quota on GPU
workers. Standalone allocation warnings are recorded, other faults reviewed.

Artifacts: `runs/jax-gap-accounting-20261012.ji9zF4mj`.

## Preparation review

The first JAX control stops before the update loop: diagnostic fingerprinting
attempted an implicit host transfer, rejected by upstream's transfer guard.
Retain its failed guard/source and cancelled unstarted host-input arm. Use the
existing public checkpoint export for fingerprints and explicit `device_get`
for the final pending drain, outside timing. Source inspection also confirms
the wrapper already precompiles training; export that executable instead of
lowering another one. Declare fresh `v2` output paths, same budgets and unchanged
timed learner code. This is an instrumentation error, not a GPU wedge.

## Native coarse accounting

The corrected JAX pair completes. Before interpreting it, audit exact initial /
final fingerprints and all reported metrics. Next build an isolated Kindle
diagnostic binary on the unchanged qualified backend, with the existing CPU
profiler feature enabled. Add CPU phase markers and harvest the two existing
transfer encoders' timestamp queries after their already-required waits. No
new kernels, transfers or waits from those markers; existing Meganeura timed
chunking still adds its disclosed inter-chunk waits.

Run one ordinary RGB control and one coarse-timestamp RGB control,16+256 updates
each,600-second bounds. This does not inject Nsight or use per-dispatch timers.
Verify all272 numerical reports, first targets and292 initial/first/final
tensors against the retained ordinary fixture. Audit timestamp coverage and
measure perturbation rather than subtracting profiled times from ordinary
timings. Require the full four world chunks and explicitly count uncovered
transfer/host intervals. The instrumentation remains isolated, not a production
runtime update or a clean-exit claim for the old Graphics captures.

The JAX CPU audit confirms exact initial/final state and272 metric reports, but
its raw HLO equality check fails: independent XLA compilations reorder/rename
fusions and differ in code allocation. Keep both programs and the failed audit;
do not call the12.39/12.68ms difference an isolated upload cost. One additional
600-second process may compare resident then host-refreshed inputs using the
same compiled executable. Restore the public initial checkpoint/counters and
fresh initial carry between arms, clear only drained deferred outputs, and
retain16+256 updates per arm. Require identical executable identity/HLO and
exact state/metric parity. This adds544 synthetic updates, zero game actions;
it is not a learning resume or permission to run repeated timing campaigns.

The coarse native capture preserves every numerical report/tensor. It covers
all22 GPU passes in255 measured updates; the last update lacks late-harvested
posterior/imagination/copy timestamps and stays explicitly incomplete, not
silently labelled idle. Its additional timed-chunk waits raise56.27ms to60.51ms.
Do not subtract this perturbed CPU/GPU split from ordinary latency.

Declare one ordinary/CPU-only profiling pair on the same isolated source,
16+256 updates each,600-second bounds. Enable only the existing CPU subscriber
and phase markers, with GPU timestamp collection off: no added waits, barriers,
queries or submissions. Compare exact metrics/checkpoints and ordinary timing;
use the coarse capture for calibrated pass overlap and this new capture for
unmodified scheduling. This distinguishes real host work from profiler-created
bubbles without another Nsight teardown trial.

## Matched hot-matrix controls

The CPU-only pair passes exact state/metrics at56.46/56.28ms; no measurable
marker overhead. It locates2.33ms imagination preparation,1.61ms CPU targets
and1.04ms behavior synchronization, but these do not explain the GPU gap.
An ordered plan/Nsight join identifies the following eight fixed matrix cases
(M,N,K,G): grouped forward(128,256,1024,8),(128,768,256,8),
(8,256,1024,8); grouped BT(8,256,768,8),(8,1024,256,8);
BF16 dense forward(8,256,2048,1),(128,256,2560,1); BF16 dense
AT(2560,256,1920,1). No shape/tile sweep.

Run each current native graph with resident inputs,16 warmups and one128-call
batch recorded into a single submission, followed by one bounded timestamp
batch. The second batch is instrumentation evidence, not ordinary latency.
Compare the same operands/shapes in JAX: grouped F32/highest and BF16 variants,
and the matching dense BF16 output/accumulation convention. Inputs are exactly
representable bounded binary fractions, with independently computed F64 outputs;
record any operand/output casts and layouts. Save every output/selected kernel
and HLO. One CUDA-only Nsight Systems pass may supply kernel durations separately
from public dispatch/wait time; no native profiler injection or driver changes.
Use600-second per-process bounds, current guards/headroom requirements and no
learning actions/optimizer updates. A failed numerical/guard check stops the
affected queue for review. This diagnoses implementation/precision coverage;
isolated ratios are not additive whole-update speedups or learning qualification.

The first matrix service invocation uses named arguments against the controller's
positional CLI and exits before creating a queue or GPU child. Preserve that
service log; correct only the invocation in a fresh service. The declared jobs,
outputs, source and numerical requirements are unchanged and still unstarted.

Both native matrix arms pass128 sampled F64 elements/case, and the independent
full-output audit in the JAX process passes the first six native shapes. The
JAX dense-small result fails exact BF16-output comparison by.0300293. Its saved
compiled HLO has BF16 operands but returns the F32 GEMM directly: the requested
output narrowing/widening is absent. Grouped cases retain that rounding and pass.
Keep the failure/source/HLO and incomplete outputs; this is not relaxed into
a successful comparison. Declare one corrected JAX-only process, fresh `v2`
paths and unchanged call budget: return the BF16 array itself across the compiled
boundary and widen only its explicit host export. Save outputs before checking.
All full F64 requirements remain exact; completed native timings are not rerun.

## Submission-accounting control

The corrected13 JAX matrix cases and all eight full native outputs pass the
independent F64 products. Native F32 grouped products are slower even than
JAX F32; dense BF16 also lags, so missing BF16 coverage is not the whole cause.
Keep their isolated input/output-layout and dispatch-boundary limitations.

To close the wall-time accounting without the coarse profiler's extra chunk
waits, declare one native RGB ordinary/coarse pair with the world session's
existing submission count set to1 in the diagnostic harness only. Everything
else stays fixed;16+256 updates/arm,600-second bounds, exact272 reports/first
targets/292 tensors. This explicitly changes scheduling, not a same-runtime
speedup or production recommendation. Compare its ordinary wall time with the
retained four-chunk control; collect all GPU pass intervals without intervening
query-pool-reuse waits and quantify remaining timestamp overhead. Retain any
late-harvested final sample as incomplete. No repeated timing sweep, model change
or learning allocation.

## Matched convolution control

Use one retained hot shape only: N128,Cin32,Cout48,H=W32,kernel5,stride1,
padding2. Time native BF16 forward and existing F32 input/weight derivatives
separately, including the current split-K weight-gradient selection. Compare
JAX F32/highest and BF16 versions of those exact logical NCHW/OIHW operations;
save chosen HLO/library kernels, casts and layout transforms.16 warmups+128 calls
per case and one128-call timestamp/CUDA-only capture,600-second process bounds.
Binary-fraction inputs allow exact independent sampled F64 forward/derivative
checks; compare complete outputs with matching explicit output precision.
No optimizer, tolerance relaxation, kernel replacement or shape sweep.

Native convolution arms pass. The JAX F32 input derivative differs from the
direct-product reference by5.59e-9 under cuDNN algorithm5; the declared exact
comparison fails and remains failed. Its full output/HLO/source are saved.
Do not treat this tiny algorithmic discrepancy as a native/GPU fault, or repeat
F32 attempts to seek a passing library algorithm. The bounded follow-up is
BF16-only forward/input/weight comparison, with fresh paths and the original
exact BF16-output checks. It adds no native rerun, does not qualify the failed
F32 arm and makes no F32-versus-BF16 convolution speed attribution.

The single-submission native ordinary/coarse pair passes at57.974/57.971ms,
with255 complete19-pass samples. No measurable timestamp overhead; GPU pass
union45.282ms, uncovered host/scheduling12.689ms. Keep the slower scheduling
diagnostic-only. A final short current JAX CUDA-only capture runs16 warmups+8
measured updates (not another256-update timing campaign),600-second bound,
using the corrected public-wrapper/accounting harness and pinned recipe.
Compare its first24 metric reports and initial state with retained controls;
account CUDA kernels/copies and wrapper/seed/metric boundaries on one clock.
No Nsight Graphics injection, native kernel change or learning allocation.
