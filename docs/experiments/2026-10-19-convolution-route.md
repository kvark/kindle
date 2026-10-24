# Recover the retained convolution route on current upstream

The [dense-policy attribution](2026-10-19-dense-policy-attribution.md) recovers
CDP but fails RGB's5% recovery gate:46.3795 versus42.9115ms. Keep that failure.
The CPU source audit finds one actual WGSL difference after restoring dense
policy: ordinary `Conv2dBF16` now routes through upstream's scalar-staging typed
generator instead of the retained vectorized/incremental generator. Its plan
and pipeline labels did not change. All other204 RGB/224 CDP captured sources
match by content. RGB uses this route for its first encoder and two decoder
products; CDP only for its first encoder. This fits, but does not establish,
the measured3.334ms increase in RGB's world-training stage.

## One integration repair

In a separate copy of the dense-policy candidate, restore the old
`Variant::Scalar` routing to `generate_module`, including `Conv2dBF16`.
The existing typed-capability validation stays. Use its already-retained shared
generator, not a new shader, convolution selector, geometry or precision.
CPU prove this is the only backend source change, and require the generated
actual5x5/stride1/BF16-output module to match the old captured source exactly.
No dependency, flag, learning recipe, checkpoint or other policy changes.

Two24-update candidate smokes use the retained dense-policy smoke controls.
Require exact reports, targets, initial/first/final tensors, all eight program
plans/constants/pipeline identities and restoration. Then one four-process
complete comparison uses the same refreshed dependencies and dense policy,
with only this convolution routing changed: RGB/CDP,16 warmups and256 measured
updates per arm. Controls must calibrate within5% of the dense-policy timing;
both candidates must reproduce their own smoke prefixes and the original
qualified control's reports/targets/states/programs. Compare the complete
captured shader-content multisets, not their changed filename prefixes.

Retain every sample. Recovery within5% of the original fresh qualified stack
supports integrating current dependencies without losing old performance, not
closing the JAX gap. Report full RGB against inclusive JAX27.222569ms. A failure
stays failed: no retime, favorable subset, relaxed gate or automatic production
or learning promotion. The earlier comparisons keep their original totals.

## Bounds

Same exact reviewed historical-fault guard, driver580.178.04/5080/boot and>=2GiB
sampled budget headroom. Record allocation warnings; new hard faults/API failures
remain fatal. Serialized GPU processes360s/12GiB/no swap/no CPU quota; controller
<=2400s. Preparation/build/review1CPU/2GiB/no swap/720s. Distinct source tree and
Cargo cache, no builds during timing. No driver recovery, NVML polling or games.
