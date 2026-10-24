# Recheck the historical CDP Mixed policy after backend changes

Declared October15,2026, before GPU work. This is a cost/finite-execution
diagnostic, not a new kernel, broad profile, precision sweep or learning run.

The [role attribution](../results/2026-10-15-convolution-role-cost.md) assigns
about65% of its convolution intervention to RGB reconstruction, absent from CDP.
Source review confirms the current CDP path already reuses exploration state
projections and centers before its final projection. Do not revive the obsolete
18-way first-projection diagnosis. Its remaining Mixed policy was chosen on
October12, before reduced-precision convolution/grouped derivatives and staged
posterior reuse. It still selects F16-input/F32-storage world training and a
separate BF16 posterior; uniform BF16 admits the existing staged world path.
That is a materially changed precondition, not a blind repeat of the old test.

Use unchanged isolated native99f8aab0 and its sealed effective dependencies,
with command reuse and GPU categorical means enabled. Same historical12M
Breakout1009 checkpoint, centered CDP/action-effects, B32/T16/H15,18 actions,
four heads, replay/learning rates/reward scales and fixture RNG701. Change only
the existing precision configuration: Mixed versus BF16. No image change,
decoder, smaller model, disabled intrinsic reward or batch/replay-ratio change.
The resulting derivative/storage and posterior scheduling differences belong
to this policy comparison; it does not isolate one shader or dtype effect.

First one24-update smoke per arm. Require exact346 initial tensors against
each other and the retained same-checkpoint initial, finite first outputs/all
metrics/first/final tensors, actual master-weight changes, expected device and
>=2GiB sampled estimated headroom. Verify the compiled precision summaries:
Mixed has a separate posterior/no BF16 world matrices; BF16 shares a staged
posterior and has BF16 world matrices. Check four imagination ensemble heads
and both original configurations in source; do not infer workload from names.

Only after CPU review passes, run one ordinary16-warmup+256-measured pair,
Mixed then BF16. No profiler/per-dispatch timestamps or concurrent build/CPU
analysis. Initial tensors remain exact; each arm must reproduce its own smoke's
first24 metric reports, first outputs and first tensors. Preserve all timings,
trajectories and sampled memory; report posterior+world together as well as
stage boundaries because the staged labels differ. No timing retry.

Do not require Mixed/BF16 trajectory equality or silently call finite execution
numerical equivalence: precision/order change intentionally. Measure and retain
their output/state differences. This diagnostic cannot supersede any earlier
failed F64 gate, qualify the CDP loss/gradient implementation under a new policy,
or establish learned quality. A >=3% complete-update reduction makes independent
numerical qualification worth considering, not production adoption. No smaller
benefit triggers a tuning sweep. RGB/JAX54.224/26.632ms remains a separate
unchanged workload; do not compare CDP directly to that JAX number.

Four GPU processes maximum,120s each, serialized persistent guarded service
20min bound,Restart=no,KillMode=control-group; CPU reviews separately limited
to1CPU/2GiB/zero swap. GPU workers have no CPU quota. Expected5080/580.178.04/
current boot, recorded standalone allocation warnings. API/numerical/hard/
deadline failures stop the queue for review; no blind retry, separate NVML
polling or recovery. No native rebuild, game actions, replay resume, learning
allocation, source edits or production promotion. Preserve every artifact.
