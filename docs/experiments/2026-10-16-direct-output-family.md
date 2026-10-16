# Direct F32 output: fixed whole-family test

The [CPU contract](2026-10-16-direct-output-contract.md) passes. The reviewed
GPU declaration is retained at
`runs/convolution-direct-output-20261016.93l7AixM/gpu_declaration.md`.

Run all 20 contracts/23 occurrences: original F64/exact numerical pairs, then
one native/control/candidate complete-operation timing trio per contract.
Preserve 16 warmups/128 samples and every transform/reduction. Require 5%
aggregate GPU and queued-wall improvement, with <=5% native/control calibration.
No favorable subset, retime, counter collection, full-update or learning run.
Every pair/trio is independently reviewed before proceeding. Failures stop.

100 serialized 120-second guards; a bounded persistent 12 GiB/no-swap GPU
service without CPU quota. CPU preparation/review retains one CPU/2 GiB/no
swap/720-second limits. Ordinary healthy-GPU authorization applies. Production
and Atari remain unchanged.
