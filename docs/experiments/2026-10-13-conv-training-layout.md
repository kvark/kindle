# Convolution training layout diagnostic

Extend efficient channel-contiguous staging and tile reuse through convolution
backpropagation, using the shared typed checked/tiled matrix families. Keep
F32 master parameters/accumulation/gradient outputs and the explicit BF16
derivative policy. No dedicated BF16 arithmetic shader, geometry sweep, CUDA
learner or Atari work. The grouped/dense/native-forward prototype is the control;
this study explicitly integrates the earlier bounded BF16 derivative prototype.

Carry derivative addressing, split boundaries, padding/stride and native output
layout through packing and shared tiled staging. Respect schedule locks and
checked fallback for small shapes/shared-memory or bounded-scratch limits.
Bound reduction lengths and all staging/partial buffers before allocation.
Existing strict F32 derivative defaults remain.

Before timing: CPU type/shader/fallback tests, independent rounded-operand F64
forward/dX/dW at irregular/aligned/long shapes, ordinary/tiny incoming gradients,
independent partial reductions and production alias/poison checks. Retain failed
source/results and unchanged format-appropriate numerical tolerances. Then one
ordinary complete RGB12M/B8/BF16 pair, 16 warmup+256 measured updates/arm; same
checkpoints/configs/host annotations, all packing/casts/reductions/host work
included. CDP mixed world arithmetic stays unchanged and receives qualification,
not another timing pair absent an affected graph. Audit exact initials, finite
reports/states and actual plan selections; changed trajectories are not learning
parity. No profiler, timestamps, compilation overlap or production pin changes.

Artifacts: `runs/conv-training-layout-20261013.B3HJ6iEd`.
Upstream2214412/fe5f5b5 rechecked unchanged; base056562b with the qualified shared
prototypes and Bladefe5f5b5/Nagaa4b4183. CPU preparation/analysis uses one CPU,
2GiB and zero swap. GPU work is serialized under host guards and bounded user
services, Restart=no/KillMode=control-group, no CPU quota or NVML polling.
Require5080 and≥2GiB sampled estimated headroom; retain standalone allocation
warnings. Other failures stop for review, without blind retries/driver recovery.

The full root-cause goal remains open. Stop kernel-layout work at this bounded
comparison; next is connected posterior→targets→backward execution and GPU
targets/EMA, preserving old-weight consumers and activation lifetimes.

The first numerical check fails before timing: aligned64-channel dW with a
3,808-position unsplit reduction exceeds the unchanged pointwise F64 gate
(index968,−.010330051 versus−.010347925). Source/binaries/failure remain.
One separately guarded numerical diagnosis compares checked and packed kernels
on exactly that fixture and records independent CPU F32 accumulation order.
This diagnoses the failed gate; it is not qualification or a tolerance change.
All unstarted qualification/timing jobs remain stopped pending that review.

The numerical-only control reproduces the same14 pointwise failures, maximum
error and3.62e-6 relative L2 in both checked and packed dW; dX passes. CPU F32
summation-order controls also show cancellation sensitivity. This does not
implicate packing. The reviewed correction uses256-position F32 matrix partials
and compensated summation in both shared checked/tiled weight-gradient loops,
without extra global buffers/dispatches or relaxed gates. Existing global split
counts/bounds stay. Rerun the declared checks with a fresh immutable candidate;
retain the failed version and diagnostic separately. No geometry sweep.
