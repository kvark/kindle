# Wider local image-motion path: implementation and native qualification

The [partial-state readouts](../results/2026-11-09-breakout-world-state-readouts.md)
do not provide a precise forecast repair. They omit the two-image memory that
the local decoder reads directly. The [posterior-history recipe](../results/2026-11-09-breakout-world-posterior-natural100.md)
improves current reconstruction but worsens long ball forecasts, including on
fitting data. Select a distinct local image-path capacity hypothesis before
another learning allocation. Root `/x/Temp/kindle-world-wide-motion-20261109`.

## Intended change and limits

Current local-action motion0/1/2 and fusion output widths are16, independent of
RSSM model preset. Increasing the global model does not widen this local path.
Add default-off `rgb_wide_motion`: widen those four outputs16→32 and the final
RGB convolution input16→32. Keep their5x5 kernels, layer count, normalization,
nonlinearity, incoming-action placement, global decoded path andtwo-image history.
No object head, pretrained weight, target change, motion rule or extra resize.

At1M/RGB64, expect80 logical parameters/240 saved arrays,13 changed shapes and
71,728 added scalars (603,915→675,643); verify these exact derivations before
native declaration. RGB, encoder/RSSM/latent/event geometry andparameter names
remain. New initialization/compute/gradient paths are part of the architecture;
more width is not an established bottleneck or a predicted learning gain.
The earlier denoiser-width32 failure remains retained: it uses a different
U-Net,2,048-update/224-arrival fitting task andcannot qualify this local decoder.

Require categorical local-action fusion, positive MotionEventRollout suffix and
all-weight standalone training. Allow the qualified physical action-feature
table; reject late action, transport, residual, feedback, Gaussian/mean states,
previous-frame/latent-only encoders, auxiliary/teacher/phased objectives and
whole-agent construction/restore before GPU. Keep legacy/default-off paths.

Persist explicit v27 metadata wrapping localv18 or action-featurev25 as applicable.
Bind the flag andshapes to the format; reject disguised legacy headers,missing
flags/wrong bases andunsupported scope. Do not reinterpret an existing checkpoint
or warm start changed shapes. Existing installed packages/scripts stay immutable.

## Checks and bounded qualification

CPU checks: defaults/serialization, header/constructor/scope rejection before
GPU, exact changed shapes/counts, preserved encoder/RSSM/global path andcausal
history/action/state ancestry. Check history/action derivative connectivity and
lane isolation on a small CPU reference. Reuse qualified convolution/RMS/SiLU/
upsample primitives; no backend port, new operation or broad qualification matrix.
Run affected CPU tests andstrict core/binding Clippy; package a separate matching
native/runner/test executable/source before declaration. Release compiler cache
after packaging andterminal qualification, retaining executable/wheel/evidence.

One host-only guarded native qualification capped1,200s, healthy AMD/Auto, no
emulator. Check new small-fixture values/every parameter gradient against F64
andfixed first-lane invariance across other-lane command changes, zero optimizer
steps in that fixture. With the qualified physical table, two equivalent-command
one-update worlds must have exact initial/trained tensors,13 reports andsaved
forecasts. Required parameter families must change. One trained restore must
reproduce its frozen forecasts, table, format andage. Default-off restore must
match cached controls andsource tensors without querying an old runtime or
repeating a completed assessment. Keep own actual-initial exports and all failures.

Pin exact source/native/runner/config/data andclosed receipts before execution.
Use the existing guard; stop on real faults/nonfinite values, no reset, retry,
extension, sweep or automatic successor. Declare terminal CPU review against
closed inputs, excluding own outputs/open logs. Preserve original failed
receipts if a separately declared CPU repair is required. No learning benefit
or full-world claim follows from successful implementation qualification.

No main learner is allocated here. Only after reviewing native qualification
may a separate fresh fixed16,384-update comparison use the retained1M physical-
feature recipe/data/plan, unchanged dense andbroader physics/event gates and
measured cost. Full action-conditioned physics, natural event calibration,
brick memory,late/above-top coverage,untouched families andfresh confirmation
remain required. Actor/critic,CDP andfive-game work stay parked. The complete
world-model goal remains active andunproven.
