# Breakout: one-time image-history correction helps briefly, then drifts

The frozen [one-time refresh diagnostic](../experiments/2026-11-09-breakout-world-refresh.md)
completes on the new local H0/RGB1 checkpoint. Correcting its two past images
once restores all supported ball positions at the next arrival and improves
several subsequent predictions. Correcting latent h/z alone does not give that
repair. Development forecasts still fail by sixteen steps. This strengthens the
case for investigating robustness to generated image history; it is not a trained
repair or proof that latent dynamics are generally correct.

Root `/mnt/data/kindle-breakout-world-refresh-20261109`. No learner, collection,
actor/critic update or old-model query. The full world-model goal stays unproven.
The [closed H0/RGB1 learner](2026-11-09-breakout-world-local-h0.md) and all its
original failed gates remain unchanged. No successor learner is selected here.

## What differs from the earlier diagnostic

The earlier [state/history substitutions](2026-11-08-breakout-world-state-history.md)
changed only a readout and never fed the hybrid forward. This test changes one
branch's state/history immediately before action4 and then continues that branch
causally to32 with no further correction. Only the newly trained H0/RGB1 model
is queried, not either old H16/H32 model.

Use first16 dense fitting and first16 dense development origins, original B8
order, and seeds83001/83003/83009/83023. These are reused related examples, not
untouched families or independent confirmation. Both groups' first-draw controls
match their own retained dense forecasts exactly. Sixteen query bundles contain
four modes each: prior, image history once, latent h/z once, both once.

Every branch observes the same16-image prefix. Before the fourth future action,
a history refresh inserts the true images at future arrivals2 and3; a latent
refresh inserts the image-informed posterior through future arrival3. Action4's
arrival and every later image are unavailable to a branch. All branches use the
same per-step random numbers and incoming commands. After the pulse, each branch
feeds its own decoded images into history; refreshed latent states feed their
later recurrent transitions. All corrections are privileged future information
relative to the original forecast, and mixed states can be off-manifold.

Checkpoint: final16,384, v20, size1M/RGB64/B8,603,915 weights/80 logical/240 saved
tensors, Auto. Production inference and learning code are unchanged. Test source
520b2fab; executable SHA256
`e80a2c9844514241f78cbf338942a290a68323aecf3209b7d3d8259afd775e94`.
The only Rust changes are a child-module registration under cfg(test) and its
new orchestrator, using the existing production graphs/sampling/history helpers.
No wheel, new primitive, objective or architecture change.

## Immediate recovery is real within the diagnostic

Counts locate visible truth balls within one RGB64 pixel. Missing predictions
fail. Four draws per origin are pooled;48 means12 visible origins times4, not48
independent episodes. Visibility changes by horizon and uses the existing
restricted pixel detector. The complete every-step ball/paddle/brick metrics,
all paired rescues/harms and per-trial support/correctness traces are retained.

Fitting:

| Future action | Prior | History once | Latent once | Both once |
|---|---:|---:|---:|---:|
|1|47/48|47/48|47/48|47/48|
|3, before refresh|36/52|36/52|36/52|36/52|
|4, first after refresh|22/52|52/52|22/52|52/52|
|5|16/52|44/52|16/52|44/52|
|6|9/48|35/48|9/48|35/48|
|8|4/44|14/44|4/44|15/44|
|16|0/48|7/48|0/48|6/48|
|32|1/48|1/48|0/48|0/48|

Development:

| Future action | Prior | History once | Latent once | Both once |
|---|---:|---:|---:|---:|
|1|56/56|56/56|56/56|56/56|
|3, before refresh|32/56|32/56|32/56|32/56|
|4, first after refresh|28/56|56/56|28/56|56/56|
|5|24/52|48/52|24/52|48/52|
|6|23/56|40/56|22/56|40/56|
|8|8/48|24/48|8/48|24/48|
|16|0/60|0/60|0/60|0/60|
|32|0/56|0/56|0/56|0/56|

At the pulse, history-only rescues30 fitting and28 development prior misses,
with zero harms. At8 it rescues10 fitting and16 development misses, with zero
harms. Effects are not uniformly positive at every time: at development6 it
rescues21 and harms4. At fitting32 the latent/both modes each harm one prior
success. No best horizon or mode replaces the original forecast score.

The median consecutive correct run beginning at action4 rises from0 to3 steps
on fitting and from0.5 to4 on development with a history pulse. These runs end on
missing support as well as error and are not physical lifetime estimates. The
original28 development samples rescued at the pulse include24 supported at5,
all24 correct; at8,12 of28 supported samples remain correct; at16,0 of24.
These outcome-selected cohorts are descriptive, not independent causal fractions.

History-only latent features and all reward/positive/continuation heads remain
bitwise identical to the original prior throughout32 steps. That is a structural
control for this model's separate image-history route, not proof of event
competence. Latent/both modes have their own full event and cumulative reward
traces, scored at natural support only, with probabilities averaged over draws
before event scoring. No event-calibration or original retention gate is replaced.

Two fixed first-origin/lane0/draw0 clips retain64 losslessly checked frames, with
posters at4 and8. They are illustrations, not selected best rescues. In the first
fitting case, moving red detail spreads below the bricks even after refresh;
the scalar recovery does not establish an entirely coherent physical image.

## Execution and retained failures

Two new CPU checks pass; the native check is ignored until declared. Initial
strict Clippy finds a fixture Vec allocation; original source/log remain, and
its separate array repair passes final CPU and strict core Clippy. A broad
formatter touched five unrelated files; all five were restored byte-exact to
the base before packaging. Production Rust remains unchanged against d75c70b1.

Initial input preparation writes12 input/target/reference files and completes32
source joins, then stops on a nonexistent flat reconstruction_scale key. The
original source/log/outputs remain. A separate metadata-only finish checks
loss_scales.reconstruction and writes settings/receipt without repeating joins.
No model or emulator is constructed by either preparation path.

The first guard attempt stops before spawning a native child: its kernel-log
baseline read exceeds3s. It remains failed, with25 sealed pins, zero model
queries and no GPU allocation. An independent unchanged read-only host preflight
then completes in.682s with no kernel fault/recovery. A pinned amendment declares
only the still-unstarted diagnostic under a new unchanged guard/output root.
The successful second service does not relabel the original failed attempt.

All134 execution pins pass. The native test completes16 query bundles/four
modes, four cached eight-array reference controls, exact production RGB/features/
heads at every step, common pre-pulse branches, and history-only state/head
invariance. Both before/after snapshots match all240 original tensors and
metadata:480 tensor comparisons, optimizer/model age16,384, zero updates.
All64 full trajectories and16 invariant bundles are finite; CPU analysis7.018s.
The timed post-restore native section is31.018s; the full native test is79.55s,
including initialization/preparation outside that section. Neither timer is an
isolated GPU-kernel benchmark. CPU analysis pins and closure details remain in
`analysis-declaration.json` and `closure-review.json`.

Original failed service `kindle-world-refresh-20261109`, invocation
`9a8b6a633af64e66b88fb85feb2c1583`; successful service
`kindle-world-refresh-after-host-review-20261109`, invocation
`16963b24b91b4636b94b91d1e61b9149`. Successful nine-file guard and original
25-file failed guard are separately audited. No driver reset/reload/recovery or
NVML polling. Source archive/executable/inputs/outputs remain; disposable build
cache is released after packaging and reviewed execution.

## Architectural implication

For this checkpoint and these sequences, a one-time image-history correction is
more useful for the immediate visual rollout than correcting h/z alone. Adding
the latent correction gives little further improvement and does not sustain
recovery. This is stronger evidence about propagated history sensitivity than
readout-only correction, while remaining privileged and potentially off-manifold.
It does not prove a causal percentage, eliminate latent problems elsewhere, or
qualify a learning method.

A subsequent training proposal should address robustness to generated histories
and outcome-conditioned reconstruction rather than assume a sharper posterior
image or independently sampled future pixel loss will solve sustained dynamics.
That proposal needs its own bounded fitting check and original frozen gates;
no unchanged learner or diagnostic is repeated here.

Precise full-game physics/events, natural-frequency calibration, persistent
removed-brick memory, late/tunnel/second-wall/above-top coverage, untouched
families and fresh learner confirmation remain required. No above-top-brick
endpoint is added by this diagnostic. The full world-model and five-game goals
remain unfinished; actor/critic and CDP comparisons stay parked.
