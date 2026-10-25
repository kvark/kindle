# Workgroup-matrix convolution proof

Follow the [current-algorithm attribution](../results/2026-10-14-convolution-backend.md),
not another tile search. The5080/580 advertises workgroup-scope BF16 matrices
and tensor addressing, neither exposed by our pinned Naga/Blade path.

Test one actual intermediate encoder input gradient: B512,CI32,CO48,HW32,
K5,stride1,padding2. Keep32x32 output tiles and128 threads. A diagnostic GLSL
shader uses workgroup matrices and tensor-layout loads/stores, with one48-wide
channel product per spatial tap. Driver-managed staging replaces explicit
shared arrays. Tensor views transpose the original weight layout; tensor clamps
handle image boundaries without an im2col allocation. Loop order becomes
tap-before-channel, so this is not an exact-arithmetic-order intervention.
No additional tile, split, precision, padding or geometry variants.

First one120-second validation-enabled B1 smoke: initial, gradient scaled−.5,
and restored inputs, three dispatches. Independent CPU F64 probes include
boundaries; all32,768 outputs must equal the retained deterministic reference
prefix, and changed/restored outputs must be exact. A failure stops for review.
No training, compiler/production edit or driver recovery. The standalone
C++/GLSL harness is a native Vulkan diagnostic, not a CUDA learner.

Only after the smoke's CPU review, one fresh ordinary native-control/candidate
comparison. Reuse the unchanged native primitive binary (it measures both
derivatives; only dX is the comparison). Candidate16 warmups+128 measured calls,
initial/change checks,146 dispatches total. Full resident-operation timestamps
and queued wall time include loads, padding and output stores; allocations,
initial upload and readback stay outside both measured intervals. All16,777,216
candidate outputs must match native; the128 retained F64 coordinates retain
their original bounds. No native learner rebuild or full-update timing.

A>=20% dX reduction supports subsequent shared typed compiler/backend feature
work, not production adoption or a whole-learner speed claim. Below that gate,
keep the result and do not tune this prototype. Dense-matrix speed, capability
support and library comparisons cannot substitute for this convolution result.

GPU queues use the existing persistent host guard, Restart=no,
KillMode=control-group,120-second smoke/300-second ordinary jobs,150/630-second
service deadlines, expected5080 and>=2GiB sampled Vulkan budget headroom.
Retain standalone allocation warnings; other API/numerical/validation failures
stop. CPU preparation/build/review1CPU,2GiB,zero swap. No separate NVML, reset,
new Nsight capture, Atari restart, old failure retry or root-cause-goal closure.
Artifacts: `runs/workgroup-convolution-20261014.9AR2SLMx`.
