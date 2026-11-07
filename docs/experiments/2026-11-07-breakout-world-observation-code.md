# Breakout: an observation-only visual code with recurrent dynamics

Select this architecture experiment after the
[posterior-correction diagnostic](../results/2026-11-07-breakout-world-posterior-correction.md).
Even privileged current images only partly repair a frozen posterior/readout
conditioned on a drifted h. Test a visual code whose image information and
reconstruction do not depend on recurrent history. This is a concrete hypothesis,
not an isolated cause established by that diagnostic.

Root `/mnt/data/kindle-breakout-world-observation-code-20261107`. Implementation
and native qualification pass; the separately declared bounded learner is active.

## Native intervention

Add a default-off, persisted plain-RGB option `rgb_latent_only`. Under it, the
categorical posterior reads the current learned CNN encoding without h, and the
RGB decoder reads only z. Keep the categorical prior, recurrent core, actions,
reward/continuation feature interface and native observation geometry. The core
still receives previous h/z and the submitted action; its memory learns temporal
dynamics. No object/RAM labels, teacher actions, pretrained encoder or emulator
state enter learning. No CPU resizing or Python model/inference path.

Removing these two recurrent inputs changes parameter shapes/count and gradient
routes together. Disclose them; do not claim isolated inference-conditioning or
capacity effects. Use the existing `motion_latent_overshooting`, H16 objective:
observed motion-weighted RGB reconstruction, one-step KL1/.1, observed external
reward/continuation1 and causal future KL against detached image posteriors. Its
future target now depends on that future image, without teacher h. Future targets
must never enter a prior transition. No new loss formula or predictive RGB term.
The target representation is still jointly learned; calibration is not implied.

Legacy/default configurations and checkpoints keep their behavior. Save the new
architecture under a distinguishable world-only format that older natives reject
before construction. Ordinary restore preserves the option/geometry and accepts
no fresh overrides. Reject incompatible CDP/pretrained/frame-pair/cropped modes;
do not broaden whole-agent learning in this experiment. Keep production defaults.

## Qualification and fixed allocation

CPU checks cover configuration/schema/defaults, changed shapes/counts, observation
posterior/decoder separation, causal target routing and the existing report/mask
contract. Run changed Rust/binding checks and strict Clippy, then package exact
native/runner/source bytes. Do not repeat unchanged primitive qualifications.

Separately declare one guarded full-shape fixture (900s): exact actual initial
checkpoint/restore tensors and18 prediction arrays; current-image posterior logits
and RGB readout invariant under h perturbation at fixed image/z; finite reports,
total/mask identities; expected trainable families change; inference invariance;
legacy control restore unchanged and old native rejection of the new format.
Record exact scalar/logical/optimizer tensor counts before any learning allocation.

Only after qualification, separately declare one fresh8,192-update learner(3600s)
and restore(600s), then new-final-only84-group frozen branches(1800s). Keep
RGB64/size12M/B8/T32/full BPTT/Auto,seed41001,rates4e-4,loss scales,original sixteen
fitting/eight development streams and the full original4/2/1/1 coverage plan.
No new data or emulator actions/frames. Retain actual initial,2048/4096/final
checkpoints,original periodic curves and256 fitting/128 development dense exports.
Verify reset/terminal/suffix exposures against the retained plan before launch.
The size label no longer implies the same parameter count or matched compute.

Shared initialization geometry changes, so record actual initial controls; do not
claim all tensors match the old architecture. Report both initial/final quality
and wall time,with reused development seed/families distinguished from independent
confirmation. Do not select a favorable intermediate endpoint.

Keep every original observed/one/eight-step,primary16/32 energy,serve/ending/
false-alarm and each-plan freeflight-presence gate. Report all horizons,actions,
reward,paddle,brick errors and regressions. Use the existing fixed oracle and
cached controls without recollection or old-model queries. No easier slice,
unchanged extension,sweep,concurrent learner or automatic successor. Use AMD/Auto
and the existing host guard; stop faults/nonfinite state,no reset/recovery/NVML.

Passing intermediate gates is not full-goal completion. Precise causal forecasts,
event/reward calibration,brick memory,late/tunnel/second-wall coverage,untouched
families and fresh confirmation remain required. Actor/critic and CDP efficiency
comparisons stay parked until the world-model requirements are established.

## Qualification and actual allocation

Native source780f00a8/SHAe6ca7ca4 and runner68e9bbb7 are separately pinned. The
new architecture has66 logical parameters,8,065,219 scalar weights and198 checkpoint
tensors; the old68/8,852,675/204 geometry remains separately qualified. Removing
the RGB deterministic branch removes two parameters and narrows the posterior
input. Full-shape CPU compilation/gradient-family checks pass; all54 affected Rust
CPU and99 packaged Python checks and strict core/binding Clippy pass.

The first guarded qualification passes268 pins in32.457s:198 exact initial tensors,
18 initial/trained restore arrays,18 exact legacy arrays,finite11-loss update,
34 invariant visual parameters,exact h-perturbation separation and legacy-v4
rejection. A separate309-pin learner/restore declaration selects fresh8192 updates
with the original full plan. Matched256/128 dense initial exports use the final
dense locations/draw; all198 post-query initial tensors must match the qualified
actual initial checkpoint. The final-only branch allocation remains unstarted.
