# Breakout B8/T32 bounded execution and cost check

Initial preparation, before the separate selection recorded below:
prepare a possible bounded check while the Qbert behavior-rate learner runs.
This preparation selects no GPU allocation. Require completed Qbert assessments,
original audits/collection and an idle host before separately selecting a check.
No trial directory,queue or declaration is created by CPU preflight.

The candidate changes batch/world-microbatch16→8 and sequence/world-backprop16→32
on the retained encoder4e-5 Breakout recipe. Keep256 sampled replay positions per
update and R16. Longer temporal gradients might help sustained trajectory
control,but cached recurrent state already carries history and is refreshed
after replay learning. This is not an added-memory claim. Sequence diversity,
sampling,behavior-target rows240→248 and warmup also change;their effects are
not isolated. Recent takeover-state diagnostics do not prove short gradients
cause the control gap.

Negative context evidence remains:older unsupervised B2/T64 Breakout was weak,
and supervised Qbert B8/T32 lowered both means at4.44% more training time.
This is a different game/regime from Qbert and a less extreme batch reduction
on the stronger12M64/encoder4e-5/expert100 Breakout recipe. It needs a short
execution/cost check before any fresh gameplay comparison. No context sweep,
repetition of old lives or automatic full learner is selected.

Keep centeredCDP/12M64/Auto/N8/H15/R16,encoder/base/actor4e-5,dynamics4e-4,
actor-unimix.01,action-effects1,sampled collection and posteriors,full18/sticky.25/
repeat4,native pooled frames with one GPU64 resize. Keep expert100,unit weights,
world imitation gradients,all-state corrective labels,20% episode selection/
cap1920 and unchanged Breakout teacher. Native447fe50/SHA35ad0db and exact runner
6c1b69fb remain;use`/x/Temp/kindle-breakout-corrective-20261028/runtime`.
No native build,production edit,old weights or experience reuse.

If separately selected,fix fresh8192 actions/481 updates,first update504,residual
credit.5,then1024 greedy frozen actions withzero updates. Schedules are100k/6219
and200k/12469. Require357 control-matching initial/invariant frozen tensors,
matching semantic prefix through action376 before either learner updates,exact
config/counters,finite supervised reports,changed CNN/reward/actor/value,full
training pixel/action/label replay and both host guards. Development seeds
1009/4000001009 are reused;this cannot establish independent reliability.

CPU preflight must parse both actual commands,check config rejection cases,
schema and schedules,validate the truncated common-prefix comparison,and retain
the active Qbert's original identity and524 pins. Native/emulator construction
is forbidden. Preserve failures;do not repeat completed GPU work for CPU repairs.
Training/frozen/audit caps180/60/180s;service480s. Early cost gate is
`check_training_seconds * 200000 / 8192 * 1.10 <= 1800` (30min).
Execution is not competence. A full200k comparison requires separate selection,
retained four-file100k recovery and four final/actual-initial24-episode assessments
against sampled56.292/greedy104.667. Original90%-two-wall/864 gate remains.

Root`/x/Temp/kindle-breakout-context32-20261101/native-check`,prospective trial
`breakout-12m-64-b8-t32-r16-encoder4e5-expert100-check`,prospective service
`kindle-breakout-context32-check-20261101`. No selection or GPU launch yet.

## CPU preparation complete, still unselected

Preflight passes15:33:43 UTC November1 in.663s;protected controller completes
15:33:44 in3.164s (3.201s wrapper). Both actual CLI paths,15 audit-config cases,
357 schema tensors,227 retained control inputs,three schedules andfour actual
common-prefix fixtures pass. Zero native/emulator constructions. All524 active
Qbert inputs and originalPID809245/invocation361f29163c1e4b2896fd352f2b905ca8 stay
unchanged. Keep this CPU co-host time without correcting Qbert's training time.
No failure,repeated preflight,trial directory,queue,declaration or GPU job exists.
Do not rerun successful qualification;separate selection remains conditional on
completed Qbert closure and idle state. The exact checkpoint schema,runner and
controller sources are retained under the root above.

PreflightSHA256`765be1794efc5643c32bddbcf44c18c7a3371ffa3d4e293657e37e8324b9aaaa`.

## Selected after Qbert closure

The Qbert behavior8e-5 comparison and all five original services complete
successfully;original collector15:46:07 UTC November1,completionSHA256
`aa66f0dd69f66e9ab1ac334d35394721b10cee91c2f1dcd9fa95d8b1cc3c58c9`.
The host haszero running Kindle services at15:49:35. Its near-unchanged greedy
quality does not select a Qbert rate sweep. Separately select the prepared short
Breakout check above to measure the cost of longer training gradients in the
stronger encoder4e-5 regime. Recent saved-state diagnostics leave the source of
sustained-control error unresolved;they are not evidence that context is causal.
Reuse the completed CPU qualification unchanged. Require idle host,exact source
pins,clean committed selection and original service identity at launch. This
selects only8192/481 plus1024 frozen,no full learner or automatic promotion.

Launch15:51:35 UTC November1/sourcec2f7abf,originalPID812306/invocation
`c51dbe709ad14d8289d6a921769c621c`;261 inputs pinned. Service
`kindle-breakout-context32-check-20261101` is active. No full learner selected.

## Completed bounded result

Main succeeds15:52:52 and original collection15:53:01 UTC November1. Training
8,192/32,730frames/481updates takes53.799s;frozen1,024/.870s haszero updates/no
natural episodes. Conservative200k24m04.805s passes30min. All261 inputs,357
matching/invariant tensors,47 common-prefix events,full replay and guards pass.
No failed or repeated GPU stage. No full learner or automatic promotion selected;
host idle at closure. See the [completed result](../results/2026-11-01-cdp-breakout-context32.md).
