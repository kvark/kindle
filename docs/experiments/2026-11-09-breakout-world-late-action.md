# Breakout: condition decoded motion on commands after image feature extraction

The [edge-rebound audit](../results/2026-11-09-breakout-world-edge-rebound.md)
finds equal real NOOP/LEFT paths but command-dependent model failures. The
[real-history probe](../results/2026-11-09-breakout-world-oracle-history.md)
restores later predictions while leaving some collision-adjacent errors. Earlier
[state/history readouts](../results/2026-11-08-breakout-world-state-history.md)
and [one-time refresh](../results/2026-11-09-breakout-world-refresh.md) already
show image-history sensitivity in other local models; do not repeat them.

Select a different architecture hypothesis for implementation and bounded native
qualification. Root `/x/Temp/kindle-world-late-action-20261109`. No learner or
collector is selected yet. This is not an automatic continuation of a failed run.

## Intended change

Current decoder motion0 consumes six channels of two previous RGB images plus
18 broadcast command features. Its three ConvNorm layers therefore transform
image detail jointly with the current command. Move the same18 planes to the
input of the existing fusion ConvNorm, after the three image-only motion layers
have extracted their16 channels. Keep global decoded features in that fusion.
Do not add a fourth layer, object head, label, crop or extra image resize.

Both affected convolutions have5x5 kernels and16 output channels: remove7,200
weights from motion0 and add7,200 to fusion. Require the same80 logical weights,
603,915 scalars and240 saved arrays at1M/RGB64; parameter shapes differ for those
two weights. Keep latent widths, actor/critic structure, all RGB targets and
RSSM/event action inputs unchanged. This is a fresh architecture, not a valid
reinterpretation or warm restore of an old checkpoint. Initialization/gradient
paths change; constant parameter count does not establish an isolated cause or
a speed improvement.

Hypothesis: extracting local motion before direct command conditioning may
reduce command-sensitive amplification of errors in generated image history.
The current evidence does not establish normalization as the cause, nor that
late conditioning will improve prediction. Retain the original failures either
way. Do not hard-code paddle geometry, ball dynamics or conditional action aliases.

## Scope and persistence

Use default-off serialized `rgb_late_action` for late action conditioning.
Require local-action categorical RGB, fusion, positive motion-event suffix and
ordinary full-world training. Reject transport, residual, feedback, Gaussian,
auxiliary/teacher/phased objectives and whole-agent construction before GPU.
Keep optional physical command features compatible with the selected path.
Observed decoding, imagined training and inference must call the same decoder
implementation, using the same incoming action and reset/history semantics.

Use an explicit new standalone checkpoint format v26. Bind the late flag and
architecture to the format; a header cannot disguise the new shapes as legacy.
Wrap only qualified local-action v18 or action-feature v25 metadata as applicable;
retain their original validation and preserve the feature table. Default-off
legacy construction and restore must remain compatible. No existing runtime or
declaration-pinned script is edited or replaced.

## Meaningful checks and bounded qualification

CPU checks must verify defaults, constructor/checkpoint rejection, equal total
parameter count with the intended two shape changes, and graph ancestry: direct
commands enter fusion while motion0–2 depend only on causal image history.
Check command-plane layout and lane isolation with independent values; check
new history/action derivatives against a small CPU reference. Reuse established
convolution/RMS/SiLU/upsample primitives rather than repeat backend qualification.
Run applicable CPU tests and strict Clippy, package a separate native/runner pair,
then pin exact bytes before a native declaration.

One host-guarded native qualification, cap1,200s, no emulator. Verify new path
values/derivatives on small F64 fixtures, causal action/lane behavior, finite
learning reports and updates at1M/RGB64/B8/T32/H16/Auto. With the retained physical
command table, equivalent raw commands must give exact common-initial updates,
reports and saved forecasts. One trained restore must reproduce its own frozen
forecasts and preserve v26/table/optimizer age. Default-off restores of the
retained feature checkpoint must match cached controls; do not query an old
runtime or rerun a completed assessment. Keep own actual-initial exports.

Require no source mutation during a declaration, unchanged frozen tensors,
successful host guard, exact source/native/runner identity and retained failures.
Stop on numerical or real device faults; no reset, retry, sweep or successor.
Release disposable compiler intermediates after packaging. Known allocation
warnings alone are not an approval gate.

Only after reviewing this qualification may a separately declared fresh learner
use the original fixed16,384-update data/plan and all original dense and broader
physics/event gates. No acceptance, visibility, tolerance, budget or goal scope
is relaxed by this protocol. Full game coverage, changed-brick memory, natural
event calibration, untouched families and fresh confirmation remain required.
