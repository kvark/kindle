# Actual spatial-map consumers and complete-family qualification

Declarations, code and receipts:
`runs/convolution-metadata-kernels-20261016.c0XlqDMn`.
Prerequisite: [geometry proof](../results/2026-10-16-convolution-metadata.md).

One shared F32/F16/BF16 transform replaces B-operand spatial address/predicate
composition in typed/packed forward/dX/dW. Keep original A code, cursors, loads,
tiles/K order/splits, compensation, rounding, shared stages, outputs and barriers.
No rejected carry optimization, selector, width/layout/tile sweep or Naga change.

Offline:320 control/candidate modules plus a GPU initializer; exact retained
controls, actual-source integer interpretation, independent non-B/SPIR-V/binding
review and a compiled observer geometry test. Device-local immutable tables must
be initialized and verified before consumers; track the final consumer fence
before destruction. Offline success is not runtime or speed qualification.

GPU: all20 contracts/23 occurrences, numerical pairs then58-metric/50-pass
counter pairs then ordinary native/control/candidate trios. Original independent
F64 and full bit-exact output gates, changed-input/restoration checks and complete
plan/pipeline/binding/segment audit.16 warmup+128 samples/arm, no trimming/retries.
Require>=5% aggregate GPU and queued-wall benefit with<=5% absolute aggregate
control/native calibration difference. Include original transformations and
reductions; report one-time table allocation/initialization/readback separately.
No whole-world shared-cache or full-update claim from isolated processes.

140 total120-second guards, including60 candidate map initializations/readbacks;
parent3hours/12GiB/no swap/no CPU quota/Restart=no. CPU stages1CPU/2GiB/no swap/
720seconds. Expected5080/580.178.04/recorded boot and>=2GiB sampled Vulkan budget
headroom. Record allocation warnings; API/numerical/hard/deadline faults stop.
No recovery, separate NVML polling, backend rebuild, promotion or game action.

Retain the two CPU offline-review failures and repairs. The original GPU queue
stopped after two successful guards because the reader expected a string rather
than serde's Unix-tagged path. A separate CPU repair reviews those saved outputs.
`remaining_declaration.md` retains exactly that first numerical pair and declares
only the138 unstarted guards, unchanged binary/gates. No GPU cohort is repeated.
