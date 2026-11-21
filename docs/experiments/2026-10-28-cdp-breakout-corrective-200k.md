# Breakout corrective labels at a fixed cheap budget

Select one fresh **200k-action / 24,970-update** life after the bounded native
check succeeds. Its 8192 actions take 36.919s, projecting 15m01s for 200k; this
is an early estimate, not a measured full-run cost or evidence of competence.
Keep a **25-minute training cap**. The preceding check and every Qbert service
are terminal success, and the host is idle at selection.

Use the checked recipe: 4M/RGB64/Auto/N8/B8/T16/H15/R16, coefficient1,
actor-unimix .01, full18 and original rates; seed1009/evaluation4000001009.
Training uses 20% episode guidance capped at1920 actions, expert scale10,
all18 action weights1, expert-world gradients and corrective labels on every
visited state. The edge-reset teacher receives each submitted action as feedback.
Keep native max-pooled frames with one GPU resize. Native447fe50 and the exact
checked Python adapter remain unchanged. No weights, replay, belief, games, RNG
or experience from the bounded check are reused.

Compare fixed final and actual initial policies using sampled actions, guidance
and labels off. Each arm selects the first3 natural episodes per8 independent
lanes, 24 total, with zero updates or truncations and357 invariant tensors.
Keep the original90%-two-wall gate and report scores separately. Retain excess
episodes and partial tails. Require complete frozen outcome replays/videos and
host guards. Ordinary frozen logs lack live pixel hashes; do not claim direct
equality to unavailable live images. Greedy evaluation is not selected here.

The retained same-seed forced-only cap1920 control scores2.625 versus its own
initial1.375, zero two-wall successes, in23m33s/49,939 updates. Also retain the
unaided full18/actor-unimix .01 control's7.083/zero two-wall successes as historical
context. Native/runtime, teacher edge correction/feedback, all-state labels,
world gradients, expert scale1→10, episode probability5%→20% and replay ratio32→16
change together. This is a practical recipe comparison, not an isolated causal
effect or matched update budget. Seed1009 and the cohort are reused development
data. Historical control artifacts are pinned; original declarations are not
rewritten or claimed unchanged at their current canonical source paths.

Save and audit100k recovery at12,470 updates, then the fixed final at24,970;
no checkpoint selection. Require346 shared initial tensors matching the old
control,357 initial tensors matching the bounded check, paired actual-initial
semantic traces, finite reports, changed CNN/reward/actor/value, equal labeled
row/weight sums and raw world/actor expert-loss alignment. Raw losses exclude
the expert coefficient. Verify every training label, input pixel hash, submitted
action and outcome; retain actual override counts, guided/released/unguided
exposure and learning curves. Online guided returns are not frozen competence.

Caps: training1500s; each frozen arm300s/600k actions; training replay600s with
outer630s; total CPU audit1200s; owning service3360s. Automate recovery, progress,
terminal replay and comparison. Stop on real faults or numerical failures using
the existing AMD host guard; no driver recovery, retry, cap increase, extension,
automatic successor or reuse of completed experience.

Root: `/x/Temp/kindle-breakout-corrective-20261028/full`.
Service: `kindle-breakout-corrective-200k-20261028`.
Runtime: `/x/Temp/kindle-breakout-corrective-20261028/runtime`.
CPU preparation parses3 commands, compiles13 controllers, verifies all41 bounded
inputs and346 shared initial tensors, with no native/environment construction.
Pin exact inputs and recheck live state before launch. The five-game goal remains
unfinished; no task outcome is inferred from this selection.

## Launch and CPU observer repair

Declaration`04c278538a2778f7f120ab2e28cf7183a49720eb20be16c4b6488d35cff9bc93`
pins64 inputs. Main starts21:52:19 UTC/source9e8ead1,PID207148/invocation
`f08ef9cc0f4c41f2b54af5a619dd7e5b`. Startup passes at15,576 actions/1916 finite
reports,all64 pins,346 shared old initial tensors,357 matching bounded initial
tensors and first6 expected updates. No frozen outcome is available yet.

The original CPU recovery observer fails at21:52:19 because the log file exists
before its header is written: JSONDecodeError at line1/column1. The failure is
preserved under`full/recovery-observer-repair/original-journal.txt`, SHA
`a7238c763ce32b2216a72584aab0bf82412ef85f1a08c4acc26d2b8a934ca8af`.
After validating the complete header, launch the exact unchanged watcher in
`kindle-breakout-corrective-recovery-repair-20261028`. The original learner stays
on the same PID/invocation and all64 pins stay unchanged. No GPU stage repeats.
The separate completion collector records the original failed observer and
requires the replacement to succeed; the pinned original collector is untouched.
The100k milestone and terminal audits remain required. No training restart,
cap increase,extension or next allocation follows this observer repair.

A1.131s affined CPU recount of retained Qbert training labels overlaps this life.
It constructs no environment or native agent. All64 inputs and the same learner
PID/invocation remain unchanged. Keep the co-host duration without correcting
wall time. Evidence is in the cheap-Qbert training-phase diagnostic; it selects
no new GPU allocation.

The replacement recovery observer succeeds22:01:11 UTC at100k actions/12,470
updates,357 finite tensors. Its observed complete prefix has102,424 actions/
12,772 finite reports. The original failed observer remains recorded; all64
inputs and the original learner process stay unchanged. Repair completion:
`full/recovery-observer-repair/completion-review.json`.

A further.228s affined CPU scan of retained Breakout reference detections finds
no observed descending velocity outside the existing tracker gate. It constructs
no environment or agent,changes no policy,and preserves all64 inputs and the
same learner process. Retain the co-host duration without time correction.

## Completed comparison

Breakout corrective200k completes200k actions/799,234 frames/24,970 updates in
17m25s,26.02% less training time than the old forced-only cap1920 control.
Frozen4.292 versus own initial1.375 improves the old guided2.625 but stays below
unaided full18's7.083; all0/24 two-wall successes. All64 pins,100k recovery,
346 shared old/357 bounded initial and357 invariant frozen tensors,paired initial
semantic traces,finite reports,full replays,guards,curves,exposure and comparison
pass. Main succeeds22:12:00 UTC,postprocessing22:12:22 on October28. Original
CPU observer startup failure remains; replacement succeeds,no GPU stage repeats.
Actual overrides67.135%; guided mean203.275 is not competence. Multiple settings
and native/runtime change together on reused1009. Retain stronger controls;
no extension,seed confirmation or automatic successor. No active GPU allocation.
See the corrective-200k protocol and completed pixel-reference result.

The complete receipt explicitly retains the original failed observer and the
successful replacement. Its separate CPU collector changes no declared input.
No training/evaluation stage repeats,no cap changes,and no completed life is
extended. [Full result](../results/2026-10-26-cdp-breakout-pixel-reference.md#corrective-labels-save-time-but-remain-below-unaided-breakout).
