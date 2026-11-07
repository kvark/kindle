# Lower-replay Breakout execution and cost check

Prepare and qualify scripts only while the original Qbert H5/R32 learner runs.
**No Breakout trial,queue,declaration,service or GPU allocation is selected.**
Review after Qbert training,assessments and audits close. A later deliberate
selection must be recorded here and requires an idle host;there is no automatic
successor or concurrent GPU work.

The retained200k/R16/encoder4e-5 Breakout learner reaches56.292 sampled and
104.667 greedy in21m28s,allzero of24 two-wall successes. Changing imagination
and increasing replay updates lowered imitation loss without improving play.
The roadmap also calls for testing lower replay ratios. A possible400k/R8
comparison would instead double experience with12,489 versus12,477 updates,
approximately the same scheduled world/behavior/imagined work. It could test
whether experience coverage is more useful than additional fitting work at
this budget. There is no evidence yet that this improves Breakout. The earlier
Qbert400k/R8 comparison lowered its primary score;that negative result remains.

For a possible bounded check,change only train ratio16→8 as a learner setting.
Keep centeredCDP/12M64/Auto/N8/B16/T16/H15,encoder/base/actor4e-5,dynamics4e-4,
sampled collection/posteriors,actor-unimix.01,disagreement1,expert100/unit action
weights/world gradients,all-state corrective labels,20% episode guidance/cap1920,
full18/sticky.25/repeat4/native max-pool and one GPU resize to64. Native447fe50,
exact runner6c1b69fb and teacher stay unchanged. Reuse learner1009/evaluation
4000001009 as development inputs;no weights,replay,beliefs,RNG,games or old
experience enter a fresh check or later learner.

The prospective check is8192 actions/245 updates,first update376,residual.25,
then1024 frozen greedy actions withzero updates. R8 schedules at100k/200k/300k/
400k are3114/6239/9364/12489 updates,each residual.25. A future400k versus200k
comparison would change experience,cadence,replay sampling and warmup action
duration together,not isolate data coverage. No full learner is selected.

CPU preflight validates both actual CLI paths,audit-config rejection cases,
357 checkpoint tensor shapes,all227 control pins,the five schedules and exact
runner equality without native/emulator construction. Queue validation remains
in memory;preflight creates no trial or queue. Preserve original active Qbert
PID792387/invocation460e0220583948f4b088b5dc7c92956a and490 pins before/after.
Retain all CPU co-host time without correcting training wall time.

If selected later,the bounded training/frozen/audit caps would be180/60/180s,
service480s. Require357 matching initial/invariant frozen tensors,matching
pre-update events,finite supervised reports,changed CNN/reward/actor/value,
full guided pixel/action/label replay and both host guards. Conservative400k
projection `check_seconds * 400000 / 8192 * 1.10` must stay≤2100s (35min).
A passing check would establish execution/early cost only. Any full comparison
requires a separate fixed protocol,actual-initial/final sampled/greedy24-episode
assessments,declared recovery points and the unchanged task gate.

Root `/x/Temp/kindle-breakout-r8-20261101/native-check`,prospective trial
`breakout-12m-64-b16-r8-encoder4e5-expert100-check`,prospective service
`kindle-breakout-r8-check-20261101`. No extension,retry,cap increase,ratio sweep,
queued successor or automatic promotion.

[Retained control](../results/2026-11-01-cdp-breakout-encoder4e5.md),
[more-fitting comparison](../results/2026-11-01-cdp-breakout-h5-r24.md),
[roadmap](../kindle_single_life_dreamer_plan.md).

## CPU preparation complete; still unselected

Preflight passes12:24:59 UTC November1 in.676s (3.127s enclosing wrapper):
two actual CLI paths,13 config cases,357 tensor shapes,all227 control pins and
the8192/100k/200k/300k/400k schedules. No native/emulator construction or module
loading occurs. Original Qbert identity and490 pins remain unchanged. No
prospective trial directory,queue,declaration or service exists. Sources and
qualification are retained;do not repeat the passing preflight. Review selection
only after the original Qbert comparison and audits close.

## Selected after Qbert closure

The original Qbert H5/R32 comparison and all five services have closed
successfully. Its terminal collector passes12:59:27 UTC November1 with490
unchanged inputs;completionSHA256
`72d57f10e787f3e67c006c5ba68ece44ed16017151aae0a8461fede020d553ab`.
No Kindle unit remains running at selection review.

Select only the qualified8,192-action/245-update check followed by1,024 frozen
actions,with the caps,settings,cost gate and audits specified above. The original
CPU qualification remains valid and will not be repeated. This is a deliberate
new execution/cost test after closure,not a queued successor. More replay work
failed to improve either game's retained comparison;the possible lower-ratio
Breakout direction instead changes experience coverage at roughly fixed update
count. This motivation is not evidence of a benefit or an isolated causal claim.

No400k learner,extension,ratio retry or automatic promotion is selected. Require
the bounded result and original terminal collector before deciding whether a
separate full comparison is justified. Reuse the exact qualified native and
runner;no rebuild or prior experience reuse.

## Completed bounded check

All declared runtime checks and original collector pass. Training8,192 actions/
32,723frames/245updates takes29.062s;1,024 frozen/.868s haszero updates/no natural
episodes. Conservative400k projection26m01s passes35min. All259 pins,357 tensors,
47 initial semantic events,full guided replay and guards pass. Main13:02:45 and
collector13:03:34 UTC November1/source479f4c1;no failure or repeated stage.
Do not rerun the check or its passed preflight. See the [result](../results/2026-11-01-cdp-breakout-r8.md).
A [separate full protocol](2026-11-01-cdp-breakout-r8-400k.md) selects preparation
of one fresh400k learner;full CPU qualification is required before allocation.
