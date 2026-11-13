# Bounded Breakout check with terminal-only continuation weighting

Select one bounded native change after terminal-biased replay improves ending
predictions but lowers gameplay and increases false ending indications at other
life losses. The earlier whole-continuation scale10 also regresses. Test whether
emphasizing true endings only in the posterior continuation objective is useful
without changing every objective's replay distribution. No gameplay benefit or
calibration improvement is assumed.

Add optional `terminal_continuation_weight`, default absent/off. Use weight10 on
true terminal rows (target0) and weight1 on continuing rows (target gamma), retaining
the ordinary row-count denominator and continuation loss scale1. Intermediate life
losses remain continuing. Set neither terminal replay mixture nor open-loop
supervision. Every other objective uses the existing replay samples and weights.
The altered learning can subsequently change states, actions and experience.

The enabled loss uses stable logits: mean(w*(softplus(z)−target*z)), with
w=1+(gamma−target)*(terminal_weight−1)/gamma. Omitted weight and explicit1 retain
the old sigmoid/BCE graph. Require positive finite F32 weight, horizon>1 and finite
effective scales; initially support CDP without open-loop supervision. New
checkpoints require the matching native capability; omitted defaults preserve
legacy serialization. The enabled numerical loss path differs from old probability
BCE, so do not claim an isolated coefficient effect or calibrated probabilities.

Before gameplay, require CPU config/serialization/CLI/capability checks and graph
autodiff/compilation, strict Clippy and packaged adapter tests. In a separately
guarded GPU fixture, compare loss and every logit derivative with F64 over terminal
and continuing targets, several row counts including more than256, weights1/10,
and moderate/extreme logits. Verify continuing-row derivatives are unchanged by
the terminal weight. Keep the original failure if any; no blind retry or tolerance
relaxation. Package native, wheel, test binary and exact source hashes.

Then require one default-off1024-action frozen compatibility capture against the
retained control, matching semantic events and357 invariant tensors. Only after
that passes, train a fresh8192-action candidate using CDP12M64/Auto/N8/B16/T16/
H15/R16, replay-value1/expert100, base/encoder4e-5, dynamics4e-4, behavior8e-5,
continuation1 and terminal weight10. Keep full18/sticky.25/repeat4/noops0,
20% episode guidance/cap1920, all-state labels and one GPU resize. Use seed1009.
No prior weights, replay or experience transfer. Match357 initial tensors and
47 pre-update semantic events to the retained recipe; require first update376,
489 final updates, residual credit.5, finite correctly weighted losses, changed
CNN/reward/continuation/actor/value and full recorded pixel/action/label replay.

Freeze for1024 actions with zero updates and357 invariant tensors. This short
capture is an execution check, not a gameplay assessment. Report actions, emulator
frames, construction/training/frozen times and ending exposure from existing
continuation-target means. Conservatively project200k cost as
training_seconds*200000/8192*1.10; require at most30minutes before separately
selecting a full learner. No full learner, sweep, extension or automatic successor
is selected here.

Use guards and serial stages on an idle host. Cap the numerical fixture180s,
fresh training180s, each frozen capture60s and the bounded workflow600s, with
CPU replay/audits included. Preserve original failures and completed stages;
never restart solely because an observer expires. Independent closure verifies
artifacts and the original process without repeating training or captures.

Root `/x/Temp/kindle-terminal-weight-20261104`. Retain70.750 sampled/143.667 greedy
unless a later full comparison supports changing the recipe. Reused seed/cohort,
native/runner changes and changed experience limit causal and reliability claims.
The original864/90%-two-wall and five-game goal remain unfinished.
