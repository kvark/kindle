# Kindle: CDP learning roadmap

Updated October26,2026. [Handover](handover.md) contains the reproducible runtime
and machine-transfer instructions. [PR31](https://github.com/kvark/kindle/pull/31)
is the status dashboard; individual result reports preserve the evidence.

**Current AMD development results:** all completed rows pass numerical,
checkpoint, accounting, frozen-tensor and full-replay audits. The learning gate
is separate. Each row describes one learner or warm-start lineage, not broad reliability.

| Game and recipe | Actions / updates | Training wall | Frozen final / starting policy | Task outcome |
|---|---:|---:|---:|---|
| [Boxing1M/RGB32/R16](results/2026-10-25-cdp-cheap-boxing.md) |200k /24,970|6m03s|−3.875 /0.00|10/24 wins; fail|
| Boxing1M/RGB32/R32, seed2017 |500k /124,939|28m22s|76.833 /2.875|24/24 wins; pass|
| Same recipe, confirmation4013 |500k /124,939|28m13s|79.167 /−0.625|24/24 wins; pass|
| [Freeway1M/RGB64/R16, no aid](results/2026-10-25-cdp-cheap-freeway.md) |200k /24,970|8m07s|0.00 /0.00|0/24 milestones|
| Same,5% persistent action aid in training |200k /24,970|8m11s|0.25 /0.00|0/24 milestones|
| [Freeway1M/RGB32/R32, same aid](results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget) |500k /124,939|28m17s|31.042 /0.00|24/24 milestones; pass|
| [Same recipe, confirmation4013](results/2026-10-25-cdp-cheap-freeway.md#fixed-recipe-seed4013-confirmation) |500k /124,939|28m50s|32.000 /0.00|24/24 milestones; pass|
| [Pong4M/RGB48/R16](results/2026-10-25-cdp-cheap-pong.md) |200k /24,970|10m22s|−17.125 /−20.375|0/24 wins|
| [Pong4M/RGB48/R32](results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions) |500k /124,939|48m53s|11.000 /−20.375|23/24 wins; mean15 gate fails|
| [Same recipe, confirmation4013](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation) |500k /124,939|50m14s|7.333 /−20.333|22/24 wins; mean15 gate fails|
| [Same, coefficient.01, seed1009](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result) |500k /124,939|49m55s|14.958 /−20.375|24/24 wins; mean15 narrowly fails|
| [Same, confirmation4013](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-seed4013-confirmation) |500k /124,939|49m58s|14.000 /−20.333|24/24 wins; mean15 fails|
| [Qbert4M/RGB48/R32](results/2026-10-25-cdp-cheap-qbert.md) |200k /49,939|19m45s|809.375 /180.208|0/24 first pyramids|
| [Same, intrinsic coefficient100](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result) |200k /49,939|20m12s|809.375 /180.208|0/24 pyramids; lower coverage|
| [Same coefficient1, six actions](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage) |200k /49,939|18m21s|605.208 /148.958|0/24 pyramids; lower coverage|
| [Full18, more experience at R16](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result) |500k /62,470|26m53s|719.792 /180.208|0/24 pyramids;14.125/21 coverage|
| [Breakout4M/RGB64/R32](results/2026-10-26-cdp-cheap-breakout.md) |200k /49,939|22m30s|6.208 /1.375|0/24 two-wall successes|
| [Same, native actor mixture.01](results/2026-10-26-cdp-breakout-unimix.md) |200k /49,939|22m29s|7.083 /1.375|0/24 two-wall successes|
| [Same, four actions](results/2026-10-26-cdp-breakout-minimal-actions.md) |200k /49,939|20m59s|9.542 /1.167|0/24 two-wall successes|
| [Four actions, coefficient1, actor/critic gradients](results/2026-10-26-cdp-breakout-minimal-actions.md#actorcritic-gradients-give-a-modest-unconfirmed-gain) |200k /49,939|20m41s|11.208 /1.167|0/24; modest unconfirmed gain|
| [Same, coefficient .01](results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score) |200k /49,939|21m07s|5.042 /1.167|0/24; weaker frozen score|
| [Same, H5/500k target](results/2026-10-26-cdp-breakout-compact-h5-budget.md) |487,736 /121,873|45-minute cutoff|8.458 /1.375|500k incomplete;0/24 two-wall successes|
| [Retained Breakout12M/RGB64/H5 warm start](results/2026-10-26-cdp-breakout-short-imagination.md) |50k new /12,439 new|8m02s|27.208 /67.583 source|0/24 two-wall successes; regression|

Boxing's cheap AMD recipe now passes on two fresh learners, each within30 minutes;
this is useful confirmation, not broad seed reliability. The retained
same-seed1M/RGB64/500k/R32 control takes65m38s and scores71.625. Geometry, runtime
and GPU differ, so the2.31× wall-cost comparison does not isolate resolution.
Qbert's mean first-pyramid coverage improves5.83→16.63 of21 cubes, with a maximum
of20, but its task gate fails. Neither these results nor short-screen failures
establish a universal capacity or resolution rule.
Qbert's [completed coefficient100 test](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result)
keeps the200k budget and native model fixed. Frozen score stays809.375 with0/24
pyramids, while mean coverage falls16.625→13.708 of21 cubes. The measured late
bonus/extrinsic ratio rises0.146%→15.84%; that does not establish useful
exploration. All audits pass and service exits. No extension or broad sweep.

Breakout's smaller model gains modestly, with lower score per action than the
earlier T64 run but3.07× lower wall cost at the shared196,608-action point. One
selected frozen episode waits12061 actions after losing a life before choosing
FIRE. Training with native actor mixture.01 scores7.0833 and reduces the longest
selected no-FIRE gap12115→447. Adding the mixture only to the baseline frozen
weights scores6.0833, gap724, with zero updates. All task gates still fail;
shorter waits before FIRE are useful but not Breakout success. No unchanged extension.

Compact aided Freeway now passes in28m17s:31.0417 versus actual initial0.00,
24/24 versus0/24 crossing successes. Frozen evaluation disables the aid. The
same-protocol CPU always-UP reference scores21.6667,0/24. Resolution, ratio,
budget and native runtime differ from the short screen; no isolated attribution.
Focused seed4013 confirmation now passes at32.000 and24/24 successes in28m50s,
with the recipe and budget fixed. Both runs pass all audits and services exit.
Two useful learners do not establish broad seed reliability.
Pong's4M/RGB48/R32 test now completes500k actions and124939 updates in48m53s,
scoring11.0 with23/24 frozen wins versus initial−20.375 and0/24. The mean15 gate
fails despite useful learning. All audits pass. The larger update budget does
not isolate a replay-ratio effect or establish a hardware speedup; no extension.

The [retained Breakout2M policy on AMD Auto](results/2026-10-26-cdp-retained-breakout-auto.md)
now scores67.5833,0/24 two-wall successes, with zero updates. Its21,976-action
frozen check takes14.127s; all346 source/copied/frozen tensors match and all
audits pass. Only precision metadata changes. Historical98.2083 used another
GPU/precision/runtime, so the difference does not isolate precision. This is a
transferred-policy control for any separately declared warm start, not new
learning or a matched-budget comparison with the fresh compact200k policies.

The [completed H5 warm start](results/2026-10-26-cdp-breakout-short-imagination.md)
uses50k new actions/12,439 updates in8m02s but regresses67.5833→27.2083, both0/24
two-wall successes. The frozen source trace matches H15 before learning; all
audits pass and service exits. Replay/games/belief/RNG restart, so new experience
and replay restart confound isolated horizon attribution. Retain the source;
no extension. Counter-aware curves pass70 CPU checks without changing training.

The [Pong4013 confirmation](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation)
completes500k actions/124939 updates in50m14s: frozen7.333 and22/24 wins versus
own initial−20.333 and0/24. All audits and curves pass; service exits successfully.
Both compact learners win most matches but miss the unchanged mean15 gate.
Selected seeds are not held out; no unchanged extension or broad campaign.
Their [completed-report analysis](results/2026-10-25-cdp-cheap-pong.md#exploration-bonus-scale-in-the-completed-learners)
finds a late exploration-bonus/net-predicted-reward ratio of3.76/3.46. Signed
reward cancellation limits that comparison; it does not diagnose the score gap.
This motivates the selected smaller-bonus test below, without establishing that
the bonus caused either failed score gate.

The [six-action Qbert trial](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage)
completes200k/49939 updates in18m21s, but frozen605.208 and10.167/21 mean coverage
are below the full18 control's809.375 and16.625/21. Own initial148.958;0/24 pyramids.
All audits/curves pass and service exits. No extension. A separate
[CPU Breakout references](results/2026-10-26-cdp-breakout-pixel-reference.md)
score11.333 for position following and20.292 for fixed two-frame interception,
both0/24 two-wall successes. Full replays pass; neither supplies training guidance.

The [compact Breakout H5 result](results/2026-10-26-cdp-breakout-compact-h5-budget.md)
retains the45-minute cutoff at487,736 actions/121,873 updates. Frozen8.458 versus
own initial1.375,0/24 two-wall successes; the500k target is incomplete. Its shared
196,608-action/49,091-update prefix takes18m06s versus H15's22m06s,18.06% less wall
time on the same native/GPU/model. This cost saving and modest score gain over
the differently budgeted7.083 control do not establish task competence. All four
recovery checkpoints, numerical/accounting and frozen replay checks pass; the
original training guard/controller retain their time-budget failure. Both
services exit and the comparison is reviewed. No extension.

The [Pong coefficient.01 primary](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result)
completes500k/124939 updates in49m55s: frozen14.9583 and24/24 wins versus own
initial−20.375 and0/24. The coefficient1 control scores11.0/23 wins at the same
budget. All audits/curves pass, initial tensors/semantic evaluation traces match,
and service exits. The unchanged mean15 gate still fails; do not round it into
a pass. Training cost is similar. This result motivated the confirmation below.

The [smaller-bonus confirmation](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-seed4013-confirmation)
completes500k/124939 updates in49m58s:14.0 and24/24 wins versus initial−20.333,
improving the same-seed coefficient1 control's7.333/22 wins. All audits and both
comparisons pass; service exits08:27:53 UTC. Both coefficient.01 learners still
miss mean15. These are reused development seeds, not broad reliability evidence.

The [four-action Breakout test](results/2026-10-26-cdp-breakout-minimal-actions.md)
finishes200k/49939 updates in20m59s,6.64% less training time than the full18
control. Frozen9.542 versus own initial1.167 improves on7.083/1.375, but still
0/24 two-wall successes. Changed dimensions/initialization require own controls;
no matched-weight or seed-reliability claim. All audits/comparison pass and
service exits09:01:55 UTC. Retain the modest gain; no extension selected.

The [Qbert500k/R16 result](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result)
completes in26m53s: frozen719.792 versus initial180.208,0/24 pyramids and14.125/21
coverage, below the200k/R32 control809.375/16.625. Experience increases2.5× and
updates1.251× in1.36× training time, but this joint change does not improve the
recipe. All audits/comparison pass; service exits09:33:03 UTC. No extension.

The [bounded PongB16 check](results/2026-10-25-cdp-cheap-pong.md#b16-bounded-check-passes-with-lower-early-wall-cost)
passes16,384 actions/2,002 updates in72.224s versus B8's94.054s,23.21% less early
wall time. Full config, matched initial tensors, weight changes, ordinary restore
and346-tensor invariance pass; service exits09:38:03 UTC. This is not a full-budget
throughput measurement or natural-match competence result.

The [full-budget PongB16 result](results/2026-10-25-cdp-cheap-pong.md#b16-at-the-fixed500k-budget)
completes500k/62,454 updates in38m45s,22.37% less training time than B8. Frozen
11.875 versus initial−20.375 retains24/24 wins but is below B8's14.958 and misses
mean15. Full audits/curves/comparison pass; service exits10:28:51 UTC. The online
curve develops earlier in wall time, but fixed frozen quality is lower. No
extension or seed confirmation is selected for this recipe.

The [bounded Qbert12M/RGB48/B16 check](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-rgb48-bounded-check)
passes16,384 actions/2,002 updates in149.387s, versus93.813s for4M/B8. All checks
and ordinary restore/346-tensor invariance pass; service exits10:35:06 UTC. This
measures execution/early cost, not competence or a full-budget speed ratio.

The [completed Qbert capacity/batch screen](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-the-fixed200k-budget)
scores959.375 versus own initial165.625, with0/24 pyramids. At200k actions,
12M/B16 takes32m19s versus4M/B8's19m45s; coverage16.292/21 is below16.625/21.
All audits and completed comparison pass; service exits11:15:13 UTC. The score
increase does not justify63.70% more wall time for this compute-constrained
screen. No extension or confirmation selected.

The [completed Pong doubled-rate test](results/2026-10-25-cdp-cheap-pong.md#doubling-b16-learning-rates-lowers-the-frozen-score)
scores9.125 versus the B16 control's11.875, both24/24 wins and below mean15.
Training38m32s is essentially unchanged from38m45s. All audits, initial-tensor/
semantic-trace equality and completed comparison pass; service exits11:59:15 UTC.
Retain the original rates; no extension or seed confirmation selected.

The [completed Qbert training-aid test](results/2026-10-25-cdp-cheap-qbert.md#persistent-training-aid-does-not-improve-frozen-coverage)
scores678.125 versus809.375 without aid; mean cube coverage15.125 versus16.625,
both0/24 pyramids. Training20m17s versus19m45s, with10,039 verified overrides.
All audits and completed comparison pass; service exits12:24:16 UTC. Retain the
unaided recipe; no extension or confirmation selected.

The [completed PongH5 test](results/2026-10-25-cdp-cheap-pong.md#shorter-imagination-saves-time-but-lowers-the-frozen-score)
finishes500k/124,939 updates in38m07s,23.65% less training time than H15. Frozen
8.667 is below14.958, both24/24 wins and below mean15. All audits, matched initial
tensors/traces and comparison pass; service exits13:18:17 UTC. Retain H15; no
extension or conditional4013 confirmation is selected.

The [completed smaller-bonus Breakout test](results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score)
finishes 200k actions/49,939 updates in 21m07s, but lowers frozen score
9.542→5.042, both 0/24 two-wall successes. All audits and comparison pass after
copying the omitted, verified replay manifest and rerunning only the unchanged
CPU audit. The original service failure at 13:43:51 UTC and incomplete watcher
receipt are retained; no GPU work repeats. Retain coefficient 1; no extension.

The [Pong B16/R40 run is interrupted](results/2026-10-25-cdp-cheap-pong.md#host-restart-interrupts-b16r40)
after a host restart:265,440 complete logged actions/41,416 finite reports,
saved200k/31,192-update checkpoint. Its separate frozen assessment scores−9.708
versus initial−20.375, both0/24 wins. All frozen checks pass after a CPU-only
checker-import correction; original failures/unsealed tails remain. The500k
quality question is unresolved. No resumed training or extension.

The [completed Qbert H30 test](results/2026-10-25-cdp-cheap-qbert.md#longer-imagination-costs-more-and-lowers-frozen-coverage)
finishes200k actions/799,441 frames/49,939 updates in26m43s,35.34% more training
time than H15. Frozen582.292 versus809.375 and coverage9.583 versus16.625/21
are worse; both0/24 first pyramids. Retain H15. Final training/frozen/replay/
guard/comparison checks pass, including346 matched initial tensors and identical
semantic initial traces. The declared100k recovery was never written because the
pinned queue saved every200k. Original watcher failures remain; separate CPU
completion records `protocol_complete=false`. Service exits16:26:25 UTC. No GPU
repeat, extension or confirmation for H30.

The [completed six-action Pong test](results/2026-10-25-cdp-cheap-pong.md#six-actions-save-time-but-lower-frozen-quality)
finishes500k/124,939 updates in44m21s,11.17% less training time than full18.
Frozen10.375 versus own initial−20.042 is below full18's14.958; both24/24 wins
and below mean15. All numerical/config/counter/tensor/replay/guard checks,
four recoveries and curves/comparison pass; service exits17:28:27 UTC. Dimensions
and initial weights differ, and the control used the older kernel. Retain
full18; no extension or conditional4013 confirmation is selected.

The [bounded Breakout actor/critic check](experiments/2026-10-26-cdp-breakout-actor-critic-gradients.md)
passes4096 actions/963 updates in22.474s and1024 frozen actions in0.489s. All357
tensors preserve frozen invariance; all963 initial policy/value loss reports
are nonzero, CNN/reward weights change and346 shared initial tensors match the
control. Both guards pass; service exits17:34:23 UTC. This verifies execution,
not useful task learning or gradient correctness. Native unchanged; the exact
313-test CLI patch is applied as runner`fbdaee3`.

The [completed Breakout actor/critic test](results/2026-10-26-cdp-breakout-minimal-actions.md#actorcritic-gradients-give-a-modest-unconfirmed-gain)
finishes200k/799,187 frames/49,939 updates in20m41s, with frozen11.208 versus
9.542 and own initial1.167; both0/24 two-wall successes. All357-tensor/frozen,
100k recovery, numerical/replay/guard checks and matched346 shared initial
weights/semantic traces pass. Main service exits18:05:14 UTC. A separate CPU
comparison import failure is retained; corrected comparison passes without
repeating GPU work. This is a modest unconfirmed gain, with an older-kernel
control; no extension, seed confirmation or global default change is selected.

Select the [prepared QbertRGB64 test](results/2026-10-25-cdp-cheap-qbert.md#rgb64-compact-option-cpu-preparation-only)
after reviewing that completed result. Fresh1009/eval4000001009,4M/RGB64/Auto/
N8/B8/T16/H15/R32, full18/coefficient1/actor0, original rates/no aid, ac_gradsfalse.
Only native geometry changes versus Qbert48; parameters and initialization also
change. Keep200k/49,939 updates, first248,100k recovery, zero debt,30-minute
training cap and fixed final/own initial24 natural episodes with full346-tensor/
replay/coverage/guard checks and unchanged15k/90%-pyramid gate. Started18:16:53
UTC/source`093e5c6`; startup passes4,131,729 parameters, all346 initial tensors
matching the completed4M64 native shape and248:1/256:2/256:3 updates. Recovery,
fixed evaluations, CPU replay/coverage/comparison monitoring are attached.
No cap increase, extension or automatic next allocation.

A [Pong greedy-action assessment](results/2026-10-25-cdp-cheap-pong.md#entropy-review-and-unselected-greedy-action-assessment)
is CPU-prepared but unselected. It would evaluate retained final/own initial
weights with zero learning, preserving the original sampled failure and training
cost. Entropy-log analysis gives no causal diagnosis. Review Qbert before selection.

New boot`a8648671-8577-4f42-9b46-a4345d3ac297`, kernel7.0.0-38, AMD module
`9BDBC866F55BF296F1B7FA2`; same RX7900XT/Mesa26.0.8/native binary. Host scan,
512-action restore/346-tensor invariance and natural frozen evaluation pass.
No host recovery performed. Qbert's refreshed queue uses this identity; its H15
control used7.0.0-34, so record the kernel change as a comparison confound.
Check live service state before GPU work; selected learning owner
`kindle-cheap-qbert-rgb64-200k-20261026`.

Earlier context comparisons are complete: [AMD B2/T64 F16](results/2026-10-25-cdp-amd-takeover.md)
finishes2M in11h40m, frozen500k/final/initial14.71/10.04/1.42. The
[Auto500k comparison](results/2026-10-25-cdp-amd-precision.md) takes2h53m and scores
15.67 versus initial1.42. All0/24 two-wall successes; all audits pass. Do not
extend either life. BF16 is unavailable on RX7900XT/RADV, and the early paired
precision update difference does not establish the cause of the historical gap.

**Compute priority, updated from the user's October25 direction:** finding useful
game-specific configurations matters more than maintaining one universal set of
hyperparameters. Measure frozen learning against GPU time and experience. A
working development recipe is useful evidence; reliable success across seeds is
still a separate requirement. The completed comparison retained its fixed budget.

## Goal and architecture

Learn to play games through experience, initially with sparse explicit rewards
and human guidance, eventually transferring skills between games. First make one
actor learn reliably. Video priors, native games, transfer and swarms follow.

The main path is Dreamer with **centered CDP**: a jointly learned small RGB CNN,
categorical recurrent state-space model, latent prediction instead of a pixel
reconstruction decoder, and imagined actor/critic learning. The action-effects
exploration ensemble currently has coefficient1. Native Rust/Meganeura/Blade
perform learning and inference; Python adapts games and analyzes results.
Frozen Large/Tiny JEPA and DINO are not the active frontend. Learned RGB with
reconstruction remains the control. CDP's five-game advantage over matched RGB
has not been established; the earlier Seaquest comparison is narrower evidence.

Use eight environments with batched inference and one shared learner, independent
stream histories, native max-pooled frames and one GPU resize (64 for the
reference;32/48 for declared game-specific tests). The current
Atari reference is12M/B8/T16/H15/BF16/replay-ratio32, full18-action space,
sticky actions.25 and action repeat4.500k actions are roughly2M emulator frames;
budgets count aggregate actions across streams, not per-stream actions.

## Verified position

| Game | Latest useful frozen result | What remains |
| --- | --- | --- |
| Boxing | AMD1M/RGB32/R32/500k seeds2017/4013 score76.83/79.17;24/24 wins each in28m22s/28m13s | Reference99.61; broad reliability remains open |
| Pong | AMD4M/RGB48/R32/500k: coefficient1 seeds1009/4013 score11.0/7.33; coefficient.01 seeds1009/4013 improve to14.958/14.0 and24/24 wins each in49m55s/49m58s | Reference20.45; compact mean15 gate and broad reliability remain open |
| Freeway | AMD1M/RGB32/R32/aided500k seeds1009/4013:31.04/32.00,24/24 successes each in28m17s/28m50s; historical aided cohort32.10,72/72 | Reference33.40; broad seed reliability remains open |
| Breakout | Retained B8/T16:2M98.21, same weights AMD Auto67.58; AMD B2/T64:2M10.04 | Reference381.81;zero two-wall successes; reliability open |
| Qbert | Historical2M mean7,367;62/72 first pyramids | Reference193,220.77; intermediate15k/90% also unmet |

These references are published long-run Dreamer scores, not equal-budget
controls. Five-game supremacy is **not achieved**. Report matched-frame and
wall-clock learning alongside any comparison to these endpoints.

Freeway's aid is5% persistent-uniform16-action blocks, disabled in evaluation.
Keep its cohort separate from unassisted31.83/30.58/0 (mean20.81,48/72 successes).
Do not hide seed3019's failed unassisted run or the intervention's selection on it.

Breakout warm-starting from2M to4M-total and8M-total yields80.17 and85.17,
below the98.21 source control. No unchanged extension follows. Greedy action
selection has inconsistent effects, so sampled evaluation remains standard.
Reward forecasts improve; trained state contains ball information, but longer
warm-start training does not improve15-step ball-x prediction. This supports a
temporal-learning experiment, not an immediate encoder rewrite.

Evidence and videos:

- [Pong](results/2026-10-21-cdp-pong-budget.md)
- [Freeway replication](results/2026-10-23-cdp-freeway-replication.md)
- [Breakout budget](results/2026-10-22-cdp-breakout-budget.md),
  [warm start](results/2026-10-23-cdp-breakout-warmstart.md),
  [forecasts](results/2026-10-22-cdp-breakout-forecasts.md),
  [state diagnosis](results/2026-10-23-cdp-breakout-state.md)
- [Qbert](results/2026-10-11-cdp-qbert-budget.md)
- [Boxing](results/2026-10-09-cdp-boxing-capacity.md)

## Current execution order — strategy reset

1. **Transfer the actual runtime and unfinished checkpoint.** Follow the handover;
   do not rebuild the older PR source and assume it matches recent training.
   Backend optimization and upstream porting are parked. Native/JAX parity is
   not a prerequisite for learning.
2. **Qualify initialization and frozen restore on the target machine.** Original
   host PMU faults block GPU work there. A new machine needs its own bounded
   check, not the old host's boot/device pins. Preserve the867-update smoke
   checkpoint; compare all346 tensors before/after frozen evaluation.
3. **Completed: the declared Breakout context experiment.** B2/T64 versus retained
   B8/T16 keeps128 sampled positions/update and ratio32; it changes temporal
   gradient length and independent-sequence count together. One fresh seed1009,
   2M actions, fixed500k/final/actual-initial evaluations of24 natural rounds.
   No checkpoint selection or automatic extension. See the
   [exact declaration](experiments/2026-10-23-cdp-breakout-context64.md).
4. **Completed: matched Auto learning comparison.** Frozen15.67 versus14.71
   under F16 is a small development-seed difference with zero task successes;
   both remain weak and take similar wall time. Early update differences do
   not establish the cause of the T64 regression. Use protected Auto precision
   for subsequent cheap trials; do not extend this context experiment.
5. **Find cheaper recipes for the remaining gaps.** Use T16 as the development
   starting point, eight independent lanes and batched GPU inference. Try lower
   replay ratios and smaller capacity in bounded development runs; changing
   several settings together is allowed when seeking a working cheap recipe.
   Use the game-specific capacity evidence below instead of shrinking every
   game blindly. Confirm promising configurations across seeds before claiming
   reliability. Keep useful gains separate from published endpoints.

The original B2/T64 smoke completed4096 actions/867 updates in45.245s with finite
346 tensors and exact initial tensors. Normal restores faulted before run_start;
one traced restore passed. AMD restore now passes with all346 tensors unchanged
under the explicitly adapted precision. The fresh2M life and fixed evaluations
are complete. Hardware, precision and execution-path differences remain
comparison limitations.

[Incident and diagnostic boundaries](results/2026-10-23-cdp-context64-gpu-fault.md).

## Compute-constrained development

The historical500k-action capacity cohort supports specializing by game:

| Game | Size1M frozen mean | Size12M frozen mean | Cost-conscious direction |
| --- | ---: | ---: | --- |
| Boxing |77.69 |85.26 | Prefer1M initially; both sizes won72/72 selected matches |
| Freeway |30.26 |27.54 | Prefer1M initially; preserve separate reporting for action aids |
| Pong |4.54 |12.40 | Test intermediate capacity rather than assuming1M is sufficient |
| Breakout |5.14 |38.21 | Small capacity was weak; test an intermediate preset before a long run |
| Qbert |993.75 |2657.99 | Small capacity was weak; prioritize learning per GPU-hour |

[Archived capacity data](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/results/2026-10-09-cdp-five-game-capacity.json),
blob`b600def5ce8bb55894eacf64e8a0b1c043251ad3`. These are historical development
cohorts, with some runtime differences; the roughly3.56x aggregate wall-cost
ratio is not an AMD speed prediction. They are not the later qualified-backend
results in the main status table. An intermediate native4M preset is now
implemented: deterministic1024, hidden128,32 stochastic variables with8 classes,
8 blocks, units128 and vision depth8. It is a Kindle development preset, not an
upstream benchmark size. Actual parameter count depends on image geometry;
RGB48 centered CDP/action-effects1 has3,578,225 trainable parameters.

The first cheap development configuration should use existing controls: centered
CDP, Auto precision, N8/B8/T16/H15, microbatch8, native frames/one GPU64 resize,
and replay ratio16 as a candidate instead of32. Choose capacity by the game;
1M is a justified starting point for Boxing/Freeway. Halving the ratio halves
scheduled replay-position work per action, not necessarily end-to-end time or
the experience needed to learn. Target100–200k actions and a30-minute training
cap for the first learning screen, with fixed frozen/actual-initial evaluation;
record a wall-cap cutoff explicitly. A short weak result does not prove that
the configuration cannot learn. No broad queue or automatic extension follows.
The firstR16 screens complete quickly but fail their gates. Fresh Boxing500k/R32
then passes within28m22s, with five times as many updates; the experience budget,
ratio and seed change together. Preserve enough learning work to test a recipe:
lower replay ratios are candidates, not a requirement for cheap training.

Native preprocessing, replay shapes and CNN geometry now support32/48/64, with
`--image-size` selecting fresh learned RGB/CDP geometry and restore retaining
checkpoint geometry. The default remains64. Do not simulate lower resolution
with CPU downscale/upscale.48x48 has56.25%
of the pixels and32x32 has25%, but RSSM and imagined actor/critic work do not
shrink proportionally. Check small-object visibility and actual learning before
using lower resolution, especially in ball games. Initial CPU visibility frames
support testing Boxing at32 and Pong at48; Breakout's tiny ball loses contrast
at32, so retain64 initially. Bounded native training/frozen smokes pass for
Boxing1M/RGB32 and Pong4M/RGB48, at578.11 and349.37 aggregate actions/s. They
verify execution and exact frozen tensors, not learning. The existing64x64
checkpoint restores unchanged; see the cheap-runtime result report.

## Runtime and performance

Retained NVIDIA Freeway B8 runs take about38 minutes per500k actions; Breakout2M
takes about2.5 hours. The AMD B2/T64 F16 run takes11h40m at47.59 actions/s, with
97.7% of wall time in learning; sampled device-wide busy is typically93–94%.
The failed B32 Freeway control ran faster (28m14s) but learned
worse. Batch size changes optimizer cadence at fixed replay ratio; synthetic
throughput alone is not the objective. AMD busy samples do not establish kernel
efficiency; the historical learning runs lack corresponding utilization data.
Do not recycle old65% utilization or0.51× realtime as current measurements.

The qualified c8aeba28 stack includes reduced-precision kernels, resident targets,
GPU sampling and convolution reuse. Its last retained RGB synthetic comparison
was1.525× JAX cost, with scope in the performance reports. No further kernel
sweeps or upstream porting are scheduled. Preserve the source; rebuild and check
on the new machine. Identical source does not guarantee identical binary bytes,
GPU behavior or speed across devices and drivers.

Long-term pipeline: GPU game → GPU capture → preprocessing → model → action,
with only actions read by CPU. Atari emulation currently remains on CPU. Playing
and updating can run faster than real time; no real-time throttle is required.
“Online learning” describes the coupled acting/training workflow sufficiently.

## After Atari

- **Video priors:** test whether video pretraining improves dynamics and learning
  at equal total experience/compute. Record data provenance and same-game exposure;
  video-pretraining support is not established by the Atari results.
- **Persistent/native games:** use `/x/Code/mind-games`, beginning with vkQuake2
  or TMNF, then suitable GOG/Wine games. Add sparse reward adapters and report
  end-to-end latency, reliable capture and real gameplay.
- **Transfer and retention:** train across related games, then measure adaptation
  to an unseen game, forgetting and dependence on per-game reward engineering.
- **Swarms:** only after strong single-actor native-game results and transfer.
  Experience sharing/concurrent learners are not prerequisites for this phase.

## Evidence and process

Develop cheaply, confirm promising changes across seeds. Keep actual-initial
controls, frozen evaluations with zero updates, full action accounting and
rollout videos. Disclose aids, truncations, selected seeds, warm starts and new
hardware. Checkpoint restore does not preserve replay/RNG/games/live belief.
Keep compact results in Git, large artifacts on shared storage, and only useful
checkpoint milestones. Do not build another status ledger into AGENTS.md.

[Strategy rationale](strategy_reset_plan.md), [historical reports](experiments/README.md)
and [performance history](backend_training_performance_plan.md) remain available.
Historical “next” paragraphs describe past states and do not authorize queues.
The previous full roadmap is preserved in Git at8d464e0.
