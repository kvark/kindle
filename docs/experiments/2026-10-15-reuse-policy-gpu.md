# Bounded complete-family compact-channel/reuse comparison

Declared October15,2026 after reviewing the [CPU policy](2026-10-15-reuse-policy.md).
Artifacts: `runs/reuse-policy-20261015.Yy1qEpm6`. All120 offline modules compile;
60 typed product variants fit45,568B maximum F32/27,136B BF16 explicit shared
memory. The corrected CPU review covers104,807,776 addresses and1,103,872
ownership positions. Original packing-count parenthesis and weak interval-check
observer defects are retained in the first preflight; v2 corrects the count,
checks both operands and explicitly compares16-wide reduction segments/256-term
reset boundaries. No shader/harness numerical change or GPU retry occurred.

This is one fixed complete policy, not a geometry sweep. It performs374.467G
padded tile MACs versus retained369.099G,+1.45%, so fewer nominal tensor products
do not establish its benefit. Keep all23 occurrences/20 contracts, including the
three-channel cases and their padding costs. Packed channels align to four;
the necessary bounded2D packing launch is part of this combined policy, not
promotion of the previously rejected packing-limit-only trial.

Three arms per contract:

1. Unchanged current native67559ce5, using the retained standalone wrapper and
   actual shader/grid selection; no native rebuild.
2. Unchanged original GLSL serial binary/shaders from the complete feeding study.
3. New shared typed compact-channel/reuse shader and complete-operation harness.

Numerical phase first: each arm uses the retained independent BF16 fixture,
full512-image grids/original dW splits, identical first/last active images and
exact-zero interiors. Three calls check original,−.5-scaled and restored input;
retain inputs, outputs, GPU canaries and all guards. Full output arrays compare
against the original F64 oracle. Forward relativeL2<=.004 and pointwise
1e−4+.008|reference|; dX/dW relativeL2<=2e−5 and pointwise1e−4+1e−5|reference|.
Original GLSL/native controls must remain exact. Candidate pair errors use those
same envelopes and are recorded even if exact: forward/dX term ordering can
change, so bitwise whole-trajectory equality is not assumed. No tolerance change;
earlier cancellation-fixture failures stay unresolved, not erased by this fixture.

Review the complete numerical allocation before any timing. One ordinary phase
then uses the retained dense dyadic fixture, changed/restored checks,16 warmups
and128 measured complete operations/arm. Include input/weight packs, output
handling and final dW reduction. Keep every sample/transient. Require aggregate
serial/native calibration within10% and at least20% less GPU and queued-wall time
against both controls for native integration consideration. No favorable subset,
retiming, independent-operation speed summed into a whole-update claim, or
automatic learning/full-update allocation.

Each process is guarded for120s; each persistent service has a1h deadline,
Restart=no,KillMode=control-group, no CPU quota on GPU workers. CPU reviewers
use1CPU/2GiB/no swap. Require5080/device0x2c02, driver580.178.04, current declared
boot and>=2GiB sampled Vulkan budget headroom. Record allocation warnings;
known10684 remains non-blocking. Other API/numerical/validation faults or deadlines
stop the affected job/allocation for review, with no automatic retry or recovery.
No NVML polling, driver changes, reset, production changes or game actions.
