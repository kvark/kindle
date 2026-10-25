# Connected training regions in the full BF16 learner

October14,2026. The [block prototype](../results/2026-10-14-connected-row-region.md)
passes its20% gate at35.5%. This separately reviewed integration keeps the
complete-update objective; no new Atari allocation or small-kernel sweep.

Use the retained optimized learner/compiler source, with its complete
inert-mask snapshot, Blade96c1dc7/Naga a97559b. Current upstream7574ad1/e7c3f7b
is rechecked; the capability was independently tested on that Meganeura main.
Do not combine a full-main migration with this compiler intervention.

Automatic matching preserves F32 normalization arithmetic, BF16 activation storage
and all external outputs. Add saved row/element values to the existing reduction;
reuse sigmoid in activation; combine SiLU/RMSNorm backward and scale-gradient
contributions. No independent dtype-specific or Dreamer shader. Preserve the
existing tall-row weight-gradient strategy when its partials cover multiple
rows; do not replace that with a full-image contribution buffer. Input-only
regions may use the same backward reduction without scale output. Protect
explicit intervening rounding, extra consumers and staged preparation roots.

CPU first: compiler matching, rejected unsafe cases, generated WGSL/SPIR-V,
public output preservation, plan scheduling and source/lock checks. One CPU,
2GiB/zero swap. No unfiltered backend tests or unchanged production rebuild.

Then one guarded numerical check:3x17,8x256,128x256,8x2048,128x512 and
120x256 input-only. Control/candidate, F32/BF16 activation storage, alias/no-alias
with poison. Independent analytic F64 F32 activation/dX/dW gates remain
rtol2e-4/floor1e-3 and relative-L2<=2e-5. Narrowed activations must equal explicit
RNE of the separately verified F32 control and match control/candidate exactly;
do not compare a rounded tensor as if it were F32. Input/parameter arrays stay
exact. A staged BF16 control delays dy until after preparation; saved values,
outputs and derivatives must match whole execution with no optimizer writes.
Each numerical process<=180s; failures stop for review, no tolerance relaxation.

After numerical review, one24-update RGB smoke; review finite reports, exact
initial292 tensors, first outputs, changed derivative/trajectory evidence and
the actual transformation coverage. Same source binary with the diagnostic
connected-region option disabled/enabled; qualify the disabled control against
retained fixtures. Do not assume later tensor equality after reassociation.

Only after smoke review, one RGB12M/B8/T16/H15/BF16 ordinary timing pair,
16 warmup+256 measured updates, same checkpoint/config/RNG, original four
world chunks, no profiling/concurrent compilation. All reports/tensors finite;
unchanged heavy-operation inventory and initial state required. Report complete
wall and stages, never extrapolated block savings. No geometry/batch sweep.
CDP follow-up only if RGB saves>=3% complete time; otherwise retain the
capability/evidence without promoting a broad learning-runtime change.

GPU work uses the persistent systemd host-guard queue, Restart=no,
KillMode=control-group, no CPU quota, each learner fixture<=600s and service
<=1800s. Expected RTX5080,>=2GiB sampled Vulkan estimated budget headroom;
record allocation warnings. API/numerical/hard failures stop. No NVML polling,
driver recovery, learning-quality claim, installed runtime change or goal closure.
Artifacts: `runs/connected-learner-20261014.wLj595Bi`.

Coverage review after the successful smoke: actual1920x256 policy heads also
retain the existing one-row contribution path, with hierarchical SumRows.
Before timing, add one numerical-only eight-case control/candidate F32/BF16
alias/no-alias check at that shape, same gates and180s bound. Test-only source
addition; retain the original learner binary/runtime and initial52 replays.
No change to the transformation, timing allocation or numerical thresholds.
The first heavy-inventory audit rejects shifted packing-buffer IDs; require
identical packing recipes and source buffers, not merely equal convolution
shapes, before accepting the reviewed comparison. Preserve the failed audit.

Before the unstarted timing pair, complete forward fusion too: the first
integration retained a separate activation dispatch. The revised candidate
emits the BF16 activation directly while retaining F32 saved values. Keep the
separate normalization output when it is externally visible or has other
readers. Preserve the first60 numerical replays and48 smoke updates. New
outputs live under `full/`: repeat numerical qualification with all seven
shapes in one56-case check (plus two staged pairs;60 replays), then a new
24-update smoke per arm before the single declared timing pair. No partial-
candidate timing, tolerance change or size sweep; production stays unchanged.
