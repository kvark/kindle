# Complete shared-address qualification

The separate execution declaration is retained at
`runs/convolution-carry-20261016.DpXsI5oj/gpu_declaration.md`.
The [CPU proof](../results/2026-10-16-convolution-carry.md) remains its prerequisite,
not a performance result.

One shared derivative-address change covers six contracts; seventeen forward/
packed occurrences stay unchanged. Keep the complete20-contract/23-occurrence
family, original tiles, K order, types/rounding, compensation, masks, layouts,
splits, output handling and barriers. No favorable selector or geometry sweep.

Fixed order:20 numerical control/candidate pairs with original F64 gates and
full exact outputs;20 counter pairs with58 metrics/50 passes;20 ordinary
native/control/candidate trios with16 warmup and all128 samples. Include all
transforms/reductions. Require at least5% aggregate GPU and queued-wall benefit
and at most5% absolute aggregate observer calibration difference. Review each
pair/trio, stop on failure; no retries, trimmed tails or replacement cohorts.

The observer is newly linked to replace its hard-coded wider-read generator.
Backend libraries are unchanged. Check actual replacement sources/bindings/
parameters/segments as well as Session exports. Sources/receipts are sealed
before collection; each queue checks all inputs it consumes.

140 serialized120-second guards, persistent3-hour12GiB/no-swap service without
a CPU quota. CPU stages use1CPU/2GiB/no swap/720 seconds. Require the declared
5080/580.178.04/boot and2GiB sampled Vulkan budget headroom. Record allocation
warnings; other API/numerical/hard/deadline failures stop for review. No separate
NVML polling, recovery, backend rebuild, production promotion or game action.
Passing alone would not authorize a full-update or learning claim.
