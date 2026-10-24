# Attribute the refreshed dense default regression

The complete upstream-refresh comparison passes execution/state checks but
fails performance: RGB42.912→57.108ms, CDP40.701→65.495ms. All272 reports and
initial/first/final tensors match exactly across arms. These are upstream plus
retained extensions, not stock main. Keep the original failure and timings.

One structural difference is already visible: old qualified dense lowering uses
its64x64 shared/reuse implementation on eligible shapes and checked fallback
elsewhere. New upstream ordinary lowering uses its32x32 typed implementation.
Convolutions, grouped products and all non-dense dispatches stay unchanged.
Blade/Naga also changed, so source inspection alone does not attribute latency.

## One changed mechanism

In a separate copy of the corrected integration, restore the **entire previous
dense policy**, directly from its qualified compiler block. Keep latest
Meganeura/Blade/Naga, graph, precision, split counts, flags, checkpoints and
every other extension. All dense products use the previous policy, not a
favorable subset or a new shape/geometry search. It is already represented by
the retained typed generator; no new shader algorithm or precision is invented.

CPU prove the changed source is only that lowering and its dependency paths.
Check every actual compiled dense contract against the old control; preserve
non-dense dispatches, buffers, constants, parameter paths, groups and bindings.
The newer typed generator's independent storage checks must still pass.

Run two24-update restored-policy smokes, RGB/CDP, against retained controls.
Require exact reports, targets and initial/first/final states, finite execution
and exact checkpoint restoration. Then one four-process full comparison on the
same refreshed stack: new ordinary dense policy versus restored policy, RGB
and CDP;16 warmups plus256 measured updates each. New-policy controls must
calibrate within5% of the completed refresh and reproduce it exactly. Each
restored arm must reproduce its own smoke prefix. Retain all samples.

This isolates dense policy from the backend/compiler refresh; it is not another
attempt to make the rejected new-default candidate pass, or permission to tune
a favorable subset. A restored result within5% of the old qualified whole update
supports recovering that performance on current dependencies, not closing the
JAX gap. Report restored RGB against caller-inclusive JAX27.222569ms. No automatic
production promotion or learning claim.

## Bounds

Same reviewed historical-fault guard, device/driver/boot, >=2GiB sampled budget
headroom and warning recording as the refresh. GPU processes360s/12GiB/no swap,
serialized, no CPU quota; controller<=2400s. CPU preparation/build/review uses
1CPU/2GiB/no swap/720s. Distinct source tree and Cargo target cache. No builds
during timing, no kernel search, no driver recovery or Atari actions. Review
each failure before follow-up; keep the complete earlier record.
