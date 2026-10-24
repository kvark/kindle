# Bound the cost of compensated weight-gradient accumulation

October14,2026. **Complete:** [result](../results/2026-10-14-accumulation-cost.md).
Only.89% lower fixed-world GPU time, below10%; retain compensation and close
the conditional counter/sweep path. Original pointwise failures remain.
Started after04:52 UTC in
`runs/accumulation-cost-20261014.JNqzeTou`. Follow-up to the
[weighted residual diagnosis](../results/2026-10-14-residual-reassessment.md).
The remaining gap is not just host transfer: current GPU-pass union22.144ms
versus retained JAX10.295ms. Convolution allocation6.888ms remains significant.

Test one structural hypothesis before further optimization: the native shared
weight-gradient loops carry extra total/error accumulator matrices and flush
256-position blocks. All eight RGB weight-gradient dispatches have4,096
positions/partition,16 blocks. These were introduced after real cancellation-
sensitive F64 failures. Nominal BF16 coverage hides that extra arithmetic.
No actual register pressure/spill or causal slowdown is yet demonstrated.
Do not assume cuDNN uses or omits a particular compensation algorithm.

1. Recheck published upstream; preserve production and the qualified isolated
   control. Inspect the generated checked/tiled paths and actual learner
   descriptors. Keep initial independent failures and exact source/binaries.
2. Use a separate diagnostic-only build, changing only this accumulation
   mechanism in shared typed kernels: same BF16 operands/F32 accumulation,
   layouts, grids, partitions, buffers and dependencies. No dtype-specific
   shader, compiler-policy change, tile search or simultaneous optimization.
   Preserve the existing qualified implementation untouched.
3. First measure fixed optimizer-free current graph work, with exact initial/
   final292 tensors and unchanged full control outputs. Record all alternative
   outputs/errors against the control and independent F64 fixtures; a known
   failed numerical contract remains failed, not reclassified as qualification.
   Review numerical differences before any further GPU work. Never use the
   diagnostic's changed gradients for actor/optimizer updates or a learning claim.
4. Only if the cost is substantial, use the previously validated same-algorithm
   CUDA-twin/Nsight Compute method to distinguish register/spill/instruction
   pressure from geometry. One source-derived affected shape, no shape/parameter
   sweep. Preserve all conversion/packing/reduction work. No retry of invalid
   Vulkan Nsight timestamps or failed Graphics injection.
5. Publish the cost and numerical tradeoff, including a null result. This does
   not authorize weakening the existing qualification gate or adopting an
   unqualified fast path. Any new numerical policy/learning comparison requires
   a separate explicit declaration. Continue the wider execution diagnosis.

Use serial guarded persistent GPU services with expected5080,>=2GiB sampled
Vulkan estimated headroom, recording allocation warnings and no CPU quota.
Bound each initial job to600 seconds; stop/review unexpected numerical/API/
hard faults. No blind retries, NVML polling, GPU recovery or Atari allocation.
Preparation/analysis uses one CPU,2GiB/no swap. No new permission gate for
ordinary GPU work.

Implementation: preserve the complete qualified control for one preparatory
update, then disable optimizer/clipping/accumulation. Lazily compile only the
eight dW dispatches' alternative pipelines using the same shared generators;
all buffers, plans and other pipelines remain. Restore original pipelines
before exact control-output/state checks. No alternative pipeline can execute
with an optimizer enabled. Generated-source hashes distinguish the arithmetic
variants even on the same device. No global environment toggle during threaded
compilation; a scoped generator override must reset even on panic.

CPU build review: the inherited mask patch omitted18 previously added files.
Original sources/base patch remain; a complete mask snapshot restores those
files before the new diagnostic. The original failed CPU build/log remains,
with no GPU attempt. Both scoped-override/unwind and checked/packed shader
validation tests now pass, as does backend release Clippy.

First GPU preflight is the existing cancellation-sensitive3,808-position F64
fixture in checked and packed layouts, each with original versus plain F32
accumulation. Zero optimizers, unchanged pointwise/L2 gates; the original
control must pass, alternative violations are explicit diagnostic results,
never qualification. Review this before the fixed world timing. That timing
uses one unchanged preparatory control update followed by443 optimizer-free
world-graph replays: three alternating arm orders,8 warmups+64 measurements.
Record all gradients/outputs and exact restored controls/model state.
Before any GPU timing, define “substantial” as at least10% lower fixed-world
GPU interval in the matched three-order comparison. Below that, close this
hypothesis without Nsight or an accumulation/precision sweep. This is an
investigation threshold, not a numerical acceptance or production gate.
