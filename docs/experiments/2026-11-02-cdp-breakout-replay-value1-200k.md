# Fresh Breakout replay-value comparison at the fixed 200k budget

Select one fresh 200k learner after the [bounded check](../results/2026-11-02-cdp-breakout-replay-value1.md)
passes 8,192 actions / 489 updates in 47.690s, with a conservative 200k projection
of 21m20.729s. Its original main, collector and terminal reader are closed. The
full command and observer qualification below passes; check the host is idle
and commit sources before declaring and launching. No check experience is reused.

Only replay-value loss scale 0.3→1 changes from the retained behavior8e-5
control. Keep centered CDP/12M64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
base and encoder4e-5, behavior8e-5, dynamics4e-4, sampled collection/posteriors,
actor-unimix .01, action-effects1, replay100k, expert100/unit action weights/world
gradients/all-state corrective labels, 20% guided episodes/cap1920. Preserve
full18/sticky.25/repeat4/noops0, native pooled RGB and one GPU resize. Learner1009
and evaluation4000001009 are reused development seeds. Native447fe50/SHA35ad0db
and teacher are unchanged. Runnercf4ea238 is byte-identical to the qualified
bounded runner; its only difference from control6c1b69fb is ten added lines
exposing the existing fresh-only replay-value option, with unchanged defaults.

The coefficient weights both critic fitting and representation gradients through
frozen value logits. Real rewards/boundaries and imagined-return bootstraps
already supply targets. Qbert's development gain did not pass fresh confirmation,
and the Breakout critic diagnostics concern neighboring behavior4e-5 controls.
This is a separate empirical test of the recipe, not an established missing-loss
defect, isolated critic cause or guaranteed transfer from Qbert.

Train exactly 200,000 aggregate actions / 12,477 updates within 1,800s. Retain
the 100,000-action / 6,227-update checkpoint, metadata and source hashes using
the original observer. Save actual initial and final weights. Checkpoints omit
replay, belief, games and RNG; they do not support an uninterrupted-restart claim.
No selected intermediate checkpoint, warm-start extension, cap increase or sweep.

Run final/actual-initial × sampled/greedy frozen assessments, with guidance and
labels off. Select the first three natural episodes per eight lanes: 24 per arm,
600k-action/300s caps, zero updates and truncations. Sampled remains primary.
Compare retained sampled59.000/greedy117.291667 in21m44.517s, own initials
1.416667/2.208333 and allzero two-wall successes. A useful development gain
requires higher sampled return and higher initial-subtracted sampled return per
training minute; report greedy tradeoffs. Keep the original90% two-wall gate
(864 points per successful episode). A development gain does not finish the goal.

Require357 initial tensors to match both control and bounded initialization,
47 pre-update semantic events and both actual-initial frozen trajectories.
Verify the actual config differs only in `loss_scales.replay_value`. Require
finite supervised reports, aligned world/behavior imitation, changed CNN/reward/
actor/critic weights, full pixel/action/teacher-label training replay, allfour
frozen replays/videos,357 invariant tensors per arm, host guards, recovery,
curves, exposure and paired comparison. Ordinary frozen logs lack live pixel
hashes; replay checks recorded outcomes/frames/resets without claiming unavailable
live-image equality. Guided scores and replay CE are not frozen competence.
Retain CPU co-host costs without subtracting them from training time.

## Full CPU qualification

Qualification passes03:10:57.580 UTC November2 in1.917s (2.020s wrapper): five
actual commands,32 config/14 header/four observer/ten recovery-body/four metadata
cases, six retention/two collector fixtures,44 paths/18 scripts, seven global
screens and matching observer unit lists. The actual prelaunch function passes
the valid command and rejects wrong/missing replay-value and behavior settings.
All320 bounded/282 control pins stay unchanged. No native or emulator is
constructed. PreflightSHA256
`a01c2beb238ff67c0033e3763a22bdd036a8d8fbf040f2c5d74c72bd6c2eada6`.

The first source generator failed an in-memory compilation with an indentation
error before creating the full directory. Preserve `failed-full-generator.py`
and `full-generator-failure.json`; the corrected generator compiles all18 scripts
before writing and passes its separate receipt. No GPU stage starts or repeats.
The corrected qualification and both original-failure artifacts are pinned.

Root `/x/Temp/kindle-breakout-replay-value1-20261102/full`; trial
`breakout-12m-64-b16-r16-replayvalue1-expert100-200k-seed1009`; main
`kindle-breakout-replay-value1-200k-20261102`, with original progress, recovery,
postprocessing and terminal-collection observers. Main cap4500s, audit1200s,
post/progress5100s, recovery3000s, collection5400s; declared internal deadlines
remain unchanged. Pin sources, schemas, protocols, queue, runtime and reference
evidence before launch. No overlap, retry, extension, cap increase, confirmation
or automatic successor. Record the first GPU launch and startup below.

## CPU declaration repair before launch

The first declaration attempt fails03:12:28.944 UTC after writing the queue and
replay manifest, before writing a declaration or starting any service. Its input
list carries the predecessor's `controller-cpu-review.json` name; this bounded
package instead uses `preflight.json`. The earlier CPU qualification checked
script paths and actual commands but did not resolve this dynamically built
evidence filename. Preserve the original failure and passing qualification.

The separate `complete_full_selection.py` changes only that input-list name.
It verifies all original qualified scripts, the existing queue, replay manifest,
bounded completion and pins; resolves every declaration input; records a repair
receipt and pins the failure, repair and updated protocol. It completes the
declaration without recreating the queue or modifying any qualified source.
Settings, native, runner, teacher and caps stay fixed. No successful GPU or
analysis stage is repeated. Launch remains conditional on this repair passing.
