# Test the observed library K32 contraction against native K64 liveness

October15,2026. The [fixed-address proof](../results/2026-10-15-convolution-address.md)
cuts7.438→1.877ms but retains a reported255 registers and1.700ms tensor interval.
The actual full-JAX library uses256x32x32/four-stage blocks and166 registers.
This is one library-derived contraction, not an open geometry/hint sweep.

Keep the exact fixed halo pack, buffers,256x32 output block,128 threads/grid,
BF16 operands/F32 accumulators, direct output store and complete timing boundary.
Split each K64 tap into two K32 operations: the same64 channels in the same
order, no change to total nominal matrix work or zero padding. The operand
blocks are half-sized; this may reduce compiler liveness pressure. It does not
implement or prove the library's four-stage asynchronous pipeline.

Use the unchanged runner binary. Compile both typed shaders, GPU-test BF16.
CPU-check split-coordinate equivalence and exact packing/output source. One B1
validation smoke,three calls/all32,768 values/137 retained F64 checks/change/
restore. After review, one fresh K64/K32 B512 pair,each two change checks+16
warmup+128 timed complete calls; all16,777,216 values/128 F64 probes per arm.
>=3% complete-operation change is material; no parameter refinement. General
real-valued derivative and full-update checks remain prerequisites for promotion.

After successful numerical checks, one compile-only statistics process uses
the existing query binary on the exact candidate shader. Compare reported
register/binary/shared sizes with the retained K64 query; keep local-memory
values uninterpreted. No internal representations, dispatch or code-identity
claim from this query, no renewed assembly path. This fixed resource check
is allowed for either timing outcome; it is not another performance trial.

Normal expected5080/580/current-boot host guards and>=2GiB sampled headroom;
record allocation warnings, stop API/numerical/hard faults for review. Persistent
systemd,Restart=no,KillMode=control-group,120s smoke/statistics and300s ordinary
process bounds. CPU build/review1CPU/2GiB/no swap; no concurrent compile during
timing. Zero actions/updates/NVML polling/recovery/blind retries. Preserve all
sources/results. Production/Atari and the root-cause goal remain open.
