# Typed selector: bounded GPU qualification

The CPU contract in `runs/typed-selector-20261019.MhXVhJTs` passes:587 RGB/643 CDP
typed dense incumbents preserve dispatches, pipeline keys and captured shader
sources;1,182 distinct typed variants validate offline. Grouped129/126 products
remain excluded. This is not actual alias/device-filtered search coverage or a
speedup. Preserve two stale test-API compilation failures and the SPIR-V1.3
test observer failure; the repair uses Blade's existing1.6 workgroup rule.

The local cached-command extension also needs invalidation when tuning or
swapping selected kernels, and must reject selection during a prepared step.
Old measured controls did not tune, so this is not an explanation of their gap.

After final CPU tests/build, run one process with12 fixed forward fixtures:
F16/BF16 NN,AT,BT,addend,split-K and checked tails. Inputs are nonuniform signed
binary fractions; compare every output exactly with an independent F64 dot before
and after search, and retain all reports, parameters and cache counts. No training
or optimizer. The ordinary common race selector has8s/class-program,64 classes,
64MiB scratch, its existing candidate catalog, calibration and qualification.
No selected manual tile, repeated cohort or new geometry. Numerical/API failures
stop for review; an incomplete search retains its incumbent and is reported.
Require actual command re-recording and rejection of tuning mid-prepared-step.

CPU build/review:1CPU/2GiB/no swap/720s. GPU process360s/12GiB/no swap/no CPU
quota, serialized by the reviewed historical-application-fault host guard.
Driver580.178.04, device5080/0x2c02, boot3e89d55c-a9e5-472f-a18a-06508c5bafa7,
>=2GiB sampled Vulkan budget headroom. Allocation warnings recorded; new hard
faults/API/validation/numerical errors remain fatal. No NVML polling or recovery.
This authorizes qualification, not a full-model search, timing or Atari restart.
