# Fresh Breakout imagined-value weight0.3 comparison

Separately select one fresh200k-action learner after the bounded check passes
execution and a conservative21m26s projection. Test whether reducing imagined
critic weighting improves frozen gameplay against retained70.750 sampled/
143.667 greedy, each0/24 two-wall. No full outcome is available at selection.

Only `loss_scales.value` changes1→.3; replay-value stays1 and
`actor_critic_gradient` staysfalse. This scales imagined-return fitting and its
slow-value regularizer together. Replay lambda targets still use imagined-return
bootstraps. This is neither complete-observed-return learning nor a demonstrated
bug repair. The bounded protocol records the NanoDreamer motivation and limits.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/lambda.95/
discount333/base+encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100,
unit expert action weights/world gradients/all-state corrective labels, sampled
collection/posteriors, unimix.01, action-effects1, replay100k, warmup1000,
20% episode guidance/cap1920, seed1009/evaluation4000001009, full18/sticky.25/
repeat4/noops0/native pooled RGB/one GPU64 resize. Preserve raw rewards and
true boundaries. Native447fe50/SHA35ad0db and teacher remain unchanged;
qualified runner76440d32 adds only the fresh-CDP imagined-value option.
Transfer no weights, replay, beliefs, game/RNG state or bounded experience.

Require200k actions/12,477 updates, first376 and residual.5, within1,800s.
Retain100k/6,227-update recovery and actual initial/final checkpoints. Require
357 matching initial/invariant frozen tensors,47 paired pre-update events,
finite reports, changed CNN/reward/actor/value, full training pixel/action/label
replay and host guards. Do not infer independent derivative correctness.

Assess fixed final and actual-initial checkpoints with sampled and greedy
actions: first3 natural episodes per8 lanes,24 perarm, guidance/labels off,
zero updates/truncations,357 invariant tensors and recorded-action replays/videos.
Preserve the90% two-wall gate. Sampled is primary; report greedy separately,
absolute scores and initial-subtracted return per training minute. Do not choose
checkpoints or evaluation settings after seeing outcomes. Ordinary frozen logs
lack live pixel hashes; their replay verifies actions/rewards/frames/boundaries.

Qualify all5 actual commands, exact configurations, flag handling, startup and
recovery observers, source paths, schema and checkpoint retention on CPU before
declaration/launch. Train1800s, each frozen arm300s, audit1200s, main4500s.
Launch once with original progress/recovery/postprocessing/collection observers.
No overlap, retry, extension, cap increase, coefficient sweep, confirmation or
automatic successor. Preserve failures and completed stages during any CPU repair.

Root `/x/Temp/kindle-breakout-imagined-value03-20261103/full`; runtime remains
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Pin exact qualified sources,
bounded closure, control, commands and host. Record guidance exposure, curves and
CPU co-host timings without correction. Reused seed/cohort and coupled critic
terms limit mechanism and reliability claims. Breakout and the goal are unfinished.

## Full qualification

The CPU harness passes08:17:44.844 UTC November3 in2.407s (2.512s wrapper):
18 scripts,5 commands,38 config checks,10 command cases,19 startup-header cases,
4 observer-parser cases,14 recovery cases,6 retention fixtures,2 collector
retention cases,4 metadata cases,44 literal paths and826 declaration-path entries.
All393 bounded/338 control pins and357 tensor shapes remain unchanged. No native
module,agent or environment is constructed during this qualification.

Preserve five earlier preparation failures: an old date in the copied collection
observer; an over-broad source guard that also matched the valid old control path;
a preflight invocation after generation had failed and produced no script; the
exact command assertion expecting `0.3` where the qualified command spells `.3`;
and a receipt-existence check before its wrapper could create that receipt.
Original sources and receipts are pinned. Corrections change preparation checks
and observer wiring, not the coefficient, learner, teacher or bounded stages.
The first bounded CLI fixture's separate unintended native initialization remains
recorded in the bounded report; no full learner has run before this qualification.

## Original launch and startup

The main launches **08:18:30.071 UTC November3**, source
`44f2cb481e6ae22035fb75f6181dfcefdc0fd2e3`, PID994988/invocation
`a1dbb6587e484fdf90a7b99ae093c03a`, with448 pins. Declaration SHA
`41ab9c3761c35be3dbcd15ddfd0e7bb9181746780ff5d3788370fd4026bf033f`.
Unit `kindle-breakout-imagined-value03-200k-20261103`; original progress,
recovery, postprocessing and collection observers are attached. Do not restart
this learner, repeat completed stages or overlap another GPU allocation.

Startup passes08:18:53.565:357 matching initial tensors against bounded/control
starts,47 matching pre-update events, exact configuration and first-six update
cadence,179 finite reports through3,232 actions. The later8,192-action progress
point takes48.022s with489 updates and residual.5. This is early cost only;
full training cost and frozen gameplay remain unmeasured.

The terminal reader qualifies08:19:30.509 in2.410s with5 score cases and all448
pins. Fixed descriptive coverage qualifies08:19:43.141 in3.546s with7 config
cases, reusing5 retained boundary fixtures. Thresholds216/432/648/864 describe
raw-score exposure, not verified wall transitions or causal coverage. Both retain
the original process identity. Keep CPU co-host timings without correction.
Confirm original100k recovery once after its observer; run terminal and coverage
readers once after original completion. No full outcome is available yet.

## Recovery retained

The original recovery observer succeeds at **08:29:36.892 UTC November 3**.
Independent confirmation completes once at **08:30:32.422**, in 1.281s CPU
(1.301s wrapper), preserving the original main PID/invocation and all 448 pins.
The 100k-action/6,227-update checkpoint contains 357 finite tensors. Its four
retained files total 206,067,161 bytes and match the original hashes. The observer
checks 6,384 finite reports through a complete prefix of 102,520 actions.

This retains weights and metadata; replay, belief, games and RNG are omitted.
The learner continues unchanged, and frozen gameplay remains unmeasured.
Recovery retention SHA
`c576858a78dbaefeef736d8666d0b1267cc5d38190c84f11c6f200bdd3d94d15`;
metadata SHA
`cc59c98edabce321e34729e2f26c6814aecbd797711b57603dde627d3470ad4e`.
Do not repeat the recovery observer or confirmation. Terminal and coverage readers
remain pending until the original full completion. Keep CPU co-host time uncorrected.
