# Connected forward/backward row-region control

October14,2026. Follow the completed
[compiler diagnosis](../results/2026-10-14-training-residuals.md).
No production adoption or Atari allocation.

Extend the existing generic reduction description, not a Dreamer-specific
shader: allow per-column prologue inputs and additional row/element outputs.
Use it for RMSNorm -> SiLU and its input/scale derivatives, retaining forward
inverse deviation, normalized input and sigmoid. Compare with the existing
composed operators on identical x/scale/upstream-gradient arrays. Both return
activation, input gradient and scale gradient. No optimizer or loss shortcut.

Current main7574ad1 includes BF16/shared-widening fixes; isolated worktree and
its existing Blade96c1dc7/Naga a97559b pins. Prototype arithmetic/storage is F32,
as in the retained native normalization derivatives. This is not BF16 learner
qualification. Preserve main's existing reduction code for descriptors without
the new fields, including scheduling, rounding and resource lifetimes.

CPU first: validate generated WGSL/SPIR-V, descriptor/layout/output counts,
row-output restrictions, serialization/hash changes and unchanged legacy
shader output. Use one CPU/2GiB/zero swap. Build only exact tests; no automatic
unfiltered GPU suite.

GPU numerical qualification:3x17,8x256,128x256, deterministic signed inputs,
nonconstant scales/upstream derivatives. Check every activation/dX/dW against
independent analytic F64 values with the existing reference tolerance
rtol2e-4/floor1e-3, plus relative-L2 <=2e-5 per output. Test normal aliasing with
poison and a no-alias control; parameters/inputs must remain unchanged.
Any failure stops for review; no tolerance relaxation or blind retry.

Only after numerical review, one matched8x256 fixed-input timing pair:
16 warmup and256 measured optimizer-free block replays per implementation,
recorded128 per submission through the same ordinary Session recording path.
Report complete coarse GPU span and host recording/submission time separately;
read back and recheck final outputs. Same context/device and existing reduction
geometry; no tile/arity sweep. This is the shape of96 retained backward regions,
not all135: no larger-image/weight-gradient extrapolation or additive savings.
Require >=20% lower block GPU time to justify a separately reviewed whole-graph
integration; a pass is not whole-agent throughput or parity.

Each GPU process <=120s; persistent systemd service <=600s, Restart=no,
KillMode=control-group, host guard, expected RTX5080 and >=2GiB sampled Vulkan
estimated budget headroom. Record allocation warnings; other hard/API/numerical
failures stop. No NVML polling, recovery or training. GPU workers have no CPU
quota. Artifacts: `runs/row-region-gpu-20261014.bWeLa0Dl`.
