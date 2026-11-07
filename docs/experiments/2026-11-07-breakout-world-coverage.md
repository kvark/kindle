# Behavior coverage and terminal-window sampling for the isolated world

The [closed fixed-plan challenge](../results/2026-11-07-breakout-world-branches.md)
finds useful short paddle response, accumulating ball error, wrong serving and
severe last-life ending underprediction. The fitting corpus has no16-action
ball-absent non-FIRE run, while challenge origins wait31 NOOPs. Select one bounded
**data/sampling experiment**, retaining the native motion-H0 architecture/losses.
This does not isolate data volume, new behaviors and sampling proportions from
each other. It tests whether useful coverage improves the same trainable model
before changing its dynamics objective or capacity. Actor/critic stays parked.

Root `/mnt/data/kindle-breakout-world-coverage-20261107`.

## New fitting supplement

Collect eight continuous native Atari streams,2,048 actions per stream:
16,384 aggregate actions, with actual emulator frames counted separately. Seeds
451001+1009*lane, disjoint from original fitting151001/development191001 and
the closed challenge251001-based families. All overlapping windows stay fitting;
do not use challenge branches or development outcomes as fitting examples.

Keep sticky.25/repeat4/full18/native max pooling, one GPU resize to64, Auto and
the existing native/source3fa37b6d (SHA prefixeafe8359) matching motion runtime.
Restore its actual-initial checkpoint only as the qualified resize utility;
require age0 and all204 parameters/moments invariant before/after collection.
No inference/controller policy or world gradient executes during preparation.
The pixel controller is an external data-generation aid, never a loss target.

Initial non-FIRE waits by lane:32/48/64/96/16/32/48/64. Natural resets and life
changes start further waits from16/32/48/64, fixed by lane/episode/lives. Lives
only trigger collector phases; no model input, target or life-loss penalty is
added. Wait controls are NOOP, LEFT, RIGHT, or eight-step right/left alternation,
fixed in source. No FIRE command is allowed during a waiting phase.

First four lanes otherwise use the fixed pixel controller, with.10 probability
per unblocked decision of beginning a uniformly selected full18 command held for
4/8/16/32 decisions. Last four deliberately use sustained LEFT/RIGHT (direction
depends on lane/lives), firing when the pixel controller reports FIRE. This
generates adverse controls, life loss and game-over without an actor learner.
Every actual command receives feedback after a real controller proposal.

Store incoming actions, reward, terminal, natural reset(-1), native/RGB64 frames,
emulator frames, episode family, native image hashes, analysis-only RAM positions/
lives, and collector phase tags. Reset only after natural game-over; keep terminal
arrivals. No downscale/upscale or retrospective future-action selection.

After collection, audit exact files, action/frame/reset alignment, long waiting
coverage, all18 command counts, distinct seed families, score/brick/late-game
coverage and endings. Before any learning, require at least8 natural endings,
64 complete32-arrival waiting windows spanning at least4 new lanes, and observed
RAM-ball-absent non-FIRE runs of at least32 decisions. Preserve any shortfall;
no automatic retry or budget extension. This supplement does not promise later
walls or unseen brick configurations, and those gaps must remain explicit.

## Matched-init learning after data review

After that audit passes, separately declare one fresh8,192-update learner and
final restore. Keep8,852,675 parameters,RGB64/12M/B8/T32/full BPTT/Auto,seed41001,
4e-4 rates,reconstruction100,KL1/.1,free-nat1,mixing.01,motion cross-entropy,H0,
reward/continuation1. No native rebuild, new loss, trainable parameters or warm
extension. Require all204 initial parameter/moment tensors to equal the retained
motion learner's actual initial checkpoint before the first update.

Each update contains exactly:

1. Four complete windows uniformly sampled from retained fitting streams.
2. Two uniformly sampled from the new fitting supplement.
3. One new complete waiting window: all phase tags waiting, no FIRE/reset after
   the first arrival, no termination. Tags select windows but never enter training.
4. One terminal-focused fitting window from either corpus, with natural termination
   at an offset16..31 and no intervening reset before it. Later genuine resets
   retain ordinary masking. A lost life is not a terminal target.

Pools/proportions/seeds are fixed before learning. Store the complete source plan,
pool identities, exact exposures and finite nine-loss reports. No fitting lane
or branch family enters development. This changes the sampling distribution;
low loss or event frequency under it is not natural-frequency calibration.

Retain0/2048/4096/8192 checkpoints, existing four-draw natural/contact/ending curves,
final256 fitting/128 original-development exports,1/4/8/16/32-step forecasts and
observed reconstruction. Original development locations stay fixed. Fitting
locations/distribution change and must not be presented as matched fitting pairs.
Check all204 tensors through frozen exports and18 arrays through final restore.
Stop at budget, fault or numerical failure; no automatic successor/extension.

Then separately declare the **new checkpoint's** frozen comparison on the retained
prospective oracle, without recollecting it or rerunning a completed checkpoint.
Use history16 primary/32 secondary and the same8 draws/plans. This is reused
development evidence. Compare all groups, action contrasts, proper observable
energy, event Brier, natural false alarms, and supported pixel errors.

Intermediate diagnostic gates: observed development reconstruction remains at
least90%; serve-NOOP8 detection at most10% with FIRE1 draw-match at least90% on
supported oracle targets; last-life32 ending Brier at most.10 across all four
plans and non-ending-group mean32 ending probability at most5%. Report forecasts
even when those gates fail; do not hide regressions or choose a favorable draw.
These gates do **not** complete the full world goal. Long forecasts, calibration,
removed bricks, late-game/second-wall coverage, untouched tests and fresh learner
confirmation still require evidence. No actor/critic integration is selected.

## Preparation and execution limits

Eight CPU tests pass (five new sampler/controller tests plus three affected corpus
tests). An independent CPU receipt verifies all8,192 original uniform source plans
and exposure counts exactly after extracting the shared accounting helper. No
old collector/learner/evaluator repeats. New Python sources, helpers, runtime,
checkpoints, executable, package versions, commands and budgets are pinned before
GPU work. Use the existing AMD host guard/current boot/module identity; no NVML,
reset/reload/reboot or backend qualification project. Collection timeout900s;
learning/restore budgets are separately declared after data review.
