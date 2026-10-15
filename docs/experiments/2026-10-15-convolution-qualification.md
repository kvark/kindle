# Qualify the fixed-address K32 derivative before backend work

October15,2026. Follow the [1.293ms proof](../results/2026-10-15-convolution-contraction.md).
No shader, production precision or learning change. Same actual CNN1 contract:
CI32/CO48/HW32/K5/stride1/pad2, BF16 operands/F32 output, GPU halo/weight packs.

Numerics: compare the retained fixed-address K64 and K32 shaders on three
predeclared B1 input distributions. Each uses deterministic BF16 bit patterns,
not just the earlier fixed1/64 grid: signed roughly unit-scale operands;
small wide-exponent gradients with bounded weights; bounded backgrounds with
sparse larger gradient spikes. Exponent fields are respectively120..126;
gradient96..118/weight118..125; background117..124 with gradient131 every257
positions. Seven mantissa bits and signs come from the fixed integer generator.
All uploaded bits are saved and independently regenerated on CPU.

Six guarded validation processes, each three complete operations with changed
input(-0.5)/restore checks. Check every32,768 output against independent F64
spatial/weight contraction, not just selected coordinates. Require relativeL2
<=2e-5 and every absolute error<=1e-4+1e-5*abs(reference). No tolerance relaxation
or exact K64/K32 trajectory requirement. These are synthetic derivative cases,
not actual captured learning gradients or qualification of every shape/F16.

Only after numerical review passes: one fresh native/K32 complete-operation
pair on the retained B512 dyadic fixture,16 warmup+128 timed operations per arm.
Use unchanged qualified binaries/shaders; include all K32 transforms/scratch/
dependencies. Native's existing driver also measures dW; retain it without
claiming an unchanged candidate dW speedup. Verify native plans/outputs and
candidate full outputs/retained F64 probes.>=20% GPU and queued-wall reduction
over native is the proof's performance gate. No repeated timing or tuning.
This is not a full-update gate or automatic Naga/Blade feature project.

Normal host-only guards/current5080/580/boot with>=2GiB sampled Vulkan headroom;
record allocation warnings, stop numerical/API/hard faults for review. Persistent
systemd,Restart=no,KillMode=control-group;120s numerical/300s timing processes.
CPU build/review1CPU/2GiB/no swap; no concurrent build/review during matched timing.
Zero game actions/learner updates/NVML polling/recovery or blind retries. Keep
all sources/failures/results; root-cause/quality goals and production stay open.
