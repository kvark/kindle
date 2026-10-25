# Transfer-cost diagnosis after a shared native stress failure

Declared October15,2026, after review and before this follow-up's GPU work.
The [original qualification](2026-10-15-convolution-transfer.md) is stopped,
not passed or resumed. Root`runs/convolution-transfer-20261015.IzSzHAYI` retains
its failed controller and all36 successful GPU guards/108 optimizer-free
operation calls. No ordinary timing has run.

Four forward contracts/three fixtures/three implementations produce identical
native/typed/wide outputs. The fourth contract,64→48/HW16, has40 cancellation
fixture values outside the original pointwise envelope in **all three**:
exact sum−.01629638671875, BF16-rounded reference−.016357421875,
GPU−.01611328125. Error.000244140625 versus bound.000230859375. The same40
values occur in each variant; the failed case's relativeL2 is6.47e−6.
Independent integer contraction with operands scaled by8,192 confirms every
failed F64 reference. Sum-absolute-products / absolute-sum is69,704.5; scalar
F32 accumulation does not reproduce the GPU result. This establishes a shared
native arithmetic limitation under the declared stress gate, not its exact
hardware cause, a driver failure or a learning failure.

## Distinct question and unchanged promotion prohibition

Does replacing8-byte typed operand transfer expressions with16-byte packed
expressions reduce complete-operation cost **at identical native arithmetic**?
A known shared mathematical error does not invalidate that counterfactual
timing question, but prevents calling this an absolutely qualified replacement.
No numerical tolerance changes, ignored failed values or Naga/Blade adoption.
The original absolute gate remains failed even if diagnostic timing is faster.

Reuse, do not repeat, the four completed forward numerical cohorts. Run only
the seven previously unstarted dX/dW numerical contracts, with their original
three fixtures and native/typed/wide variants. Keep every original F64 pointwise/
L2 gate, all-output bit comparison, canaries and changed/restored-input checks.
Any new numerical failure or native/candidate difference stops for review.
Only the exact reviewed40 shared forward failures above are carried as a
known limitation; no generic allowance for additional native failures.

If those checks pass, run the original, wholly unstarted ordinary diagnostic:
all11 distinct contracts/14 occurrences, native/typed/wide,16 warmup+128 measured
calls each. Include the failed forward contract and every packing/scatter/
partial-reduction cost. Keep all samples. Verify fresh native shader/geometry/
pipeline keys against the actual retained full plan and exact native outputs
for the ordinary dyadic fixture. The generic control's10% weighted native
calibration remains mandatory for native attribution; report every per-contract
difference. The10% weighted benefit remains a descriptive decision threshold,
**not a promotion gate now**. No compiler feature work follows automatically.

Sources, binaries, inputs and precision are unchanged. A separate controller
and queue declarations carry the reviewed exception explicitly; do not edit
the original sealed scripts/declaration/results. Expected36 retained guards+
63 new numerical+33 ordinary, if every remaining check passes. Zero games,
optimizer updates or full-learner speed claim.

Use the same host-only guarded serialization, expected5080/580.178.04/current
boot,>=2GiB sampled budget headroom and recorded allocation warnings. New
persistent service hasRestart=no,KillMode=control-group and45min deadline;
each GPU process<=180s. CPU review1CPU/2GiB/no swap; no build/review during
matched timing. API/hard/numerical failures stop the affected stage. No blind
retry, recovery, separate NVML query or training restart. The broader goal
remains open; this is a diagnostic continuation of the investigation, not a
relabelling of the failed qualification.
