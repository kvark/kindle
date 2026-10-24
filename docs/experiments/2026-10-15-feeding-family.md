# Actual convolution feeding: fixed complete-family comparison

Declared October15 after the [CPU feasibility proof](../results/2026-10-15-feeding-pipeline.md).
The current complete RGB update is51.945ms versus inclusive JAX27.223ms. Test
whether separating memory-loading subgroups from tensor-computing subgroups is
a useful missing mechanism, before undertaking a Naga feature change.

Use the exact23 convolution occurrences from the current B32/T16 RGB world
gradient (20 distinct contracts). Retain32/64 output tiles, K32, BF16 inputs,
F32 accumulation, reduction order/compensation,512-image batches, all original
dW splits, layouts, packing, scratch and output rounding. Include RGB boundary
channels and checked-kernel selections. No geometry, precision, compiler-flag
or favorable-shape sweep. This is not the rejected NV workgroup-matrix family.

Three arms: unchanged native control linked to the image-view backend; a GLSL
serial control implementing the same arithmetic/indexing; and the same GLSL
generator using two loading/four computing subgroups and two operand slots.
Both GLSL arms require32-lane subgroups through Vulkan pipeline features.
All invocations execute every workgroup barrier. No unsafe Naga validation
bypass, compiler edit, asynchronous-copy claim or production change.

CPU preparation must reconcile every contract, validate generated SPIR-V and
resource limits, and independently check loader ownership/address/tail/output
coverage. F32/F16/BF16 share a generator; the known packed-F32 shared-memory
overflow excludes F32 GPU execution, not a silent resource-limit increase.

Numerical allocation: one three-arm cohort per distinct contract, full512-image
dispatches and original splits, validation enabled. To keep the independent
F64 oracle bounded, the first and last images contain the retained wide-BF16
fixture and all intervening images are zero. Forward/dX must reproduce both
active images and exact zero interiors; dW uses both active images. This is
explicit sparse-fixture coverage, not broad random derivative qualification.
Retain original gates: forward relativeL2<=.004 and pointwise1e-4+.008|reference|;
dX/dW relativeL2<=2e-5 and pointwise1e-4+1e-5|reference|. Also require exact
three-arm outputs, changed-input(-.5)/restored-input checks, finite values,
unaltered inputs and canaries. Any failure stops the cohort for review; no
tolerance changes or automatic retries. Earlier cancellation failures remain.

Only after CPU and numerical review: one complete-family timing cohort, same
three arms and shapes, dense retained dyadic inputs,16 warmup/128 measured
calls each. Count every transform, scratch write and final reduction. Retain
all samples and exact full outputs. GLSL serial must calibrate within10% of
native in occurrence-weighted GPU and queued-wall totals; otherwise do not
attribute the difference to the proposed mechanism. Require at least20% lower
complete-family GPU and queued-wall cost for the pipeline arm versus both
controls before considering compiler integration. Report regressions, not a
mixed best-of selector. A full-update trial would be separately declared;
standalone savings are not complete-update performance or SM utilization.

Each GPU process has120s deadline; each numerical/timing service has45min bound,
Restart=no/KillMode=control-group, serialized by gpu_host_guard.py. Expected5080,
driver580.178.04, known boot, >=2GiB sampled Vulkan budget headroom; record
allocation warnings, other API/numerical/validation faults fatal. No game actions
or learner updates. CPU preparation/review1CPU/2GiB/no swap, absent during timing.
Keep all sources/failures under `runs/feeding-family-20261015.FnlUqnyM`.
