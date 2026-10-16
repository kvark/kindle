# Complete convolution read-span qualification

Execution declarations and sealed sources are retained in
`runs/convolution-read-spans-20261016.ZN5Ij9VI/declaration.md` and
`gpu_declaration.md`. This publication summarizes them; it does not replace
their pre-execution receipts.

## Question and fixed mechanism

Can composing repeated operand addresses, masks and contiguous reads reduce
the current convolution family's excessive feeding instructions and time?
Start from isolated58aff452, RGB46.199ms versus inclusive JAX27.223ms. No model,
precision, production or learning change.

One shared typed forward/dX/dW generator combines valid four-element spans.
Aligned spans use vector reads; unaligned spans retain scalar reads under the
common predicate. Tails/discontinuities retain scalar behavior. Weight-gradient
A retains one cursor and derives crossings instead of advancing four cursors.
Require aligned bindings, divisible logical extents and sufficient allocation
capacity; otherwise retain the original source. Never read neighboring tensors
or allocation padding. Preserve tiles, splits, K order, compensation, barriers,
rounding and output handling. All14 packed occurrences remain unchanged.

CPU qualification covers all152 shader variants, actual typed addresses, small
odd/strided/tail cases, binding negatives and preserved generated operations.
Only the generator and observer are built against retained runtime libraries;
there is no backend or compiler rebuild.

## Fixed GPU comparison

All20 contracts /23 occurrences are included, with original buffers, transforms,
scratch and reductions. The observer replaces only the convolution dispatch.
Original Session exports and the actual replacement source/bindings/segments
are audited separately.

1. Forty numerical jobs: control/candidate, original independent F64 gates,
   exact complete outputs, changed-input and restoration checks.
2. Forty counter jobs: control/candidate, retained41-metric SDK configuration,
   expected34 and at most64 passes. Retain all heavy/helper/empty ranges,
   reevaluate offline and require original control instruction counts.
3. Sixty ordinary jobs: fresh native/control/candidate,16 warmup and128 retained
   samples each. Include all operation costs; do not use profiled timings.

Independent CPU review follows every pair/trio; failures stop the controller.
Require at least5% aggregate GPU **and** queued-wall benefit, with absolute
control/native calibration within5%, before considering integration. No
favorable subset, retiming, geometry/width/precision sweep or automatic full-
update test follows. A family improvement would not establish learning parity.

Persistent serialized service:3-hour deadline,12GiB/no swap, no GPU-worker CPU
quota. Each process has a120-second host guard. CPU preparation/review uses
one CPU,2GiB/no swap and720 seconds. Require5080/580.178.04, the declared boot
and at least2GiB sampled Vulkan estimated budget headroom. Record allocation
warnings; API/numerical/hard failures remain fatal. No NVML polling or recovery.
Zero learner updates/actions; Atari remains stopped.

The original CPU compilation, alignment-audit and observer binding-index
failures remain. Five runtime libraries match historical hashes; all137 are
sealed now. The earlier complete timing queue did not seal those137 libraries,
so its failed membership assertion is retained without inventing old receipts.
