# Three-step world supervision worsens frozen Breakout at the short budget

The fresh H3/scale1 comparison lowers sampled **70.750→60.875 (−13.96%)**
and greedy **143.667→104.625 (−27.18%)**, with **zero of 24 two-wall successes**
in every arm. Training takes **22m06.821s**, **3.02% longer** than the retained
control. Initial-subtracted score per training minute falls **16.76% / 29.72%**.
Retain the control without H3 and constant actor/world imitation100. No
extension, coefficient sweep, confirmation learner or automatic successor is
selected from this result.

| Fixed evaluation | Retained control | H3/scale1 | Two-wall successes |
| --- | ---: | ---: | --- |
| Final sampled, primary | 70.750 | 60.875 | 0/24 each |
| Final greedy | 143.667 | 104.625 | 0/24 each |
| Actual initial sampled | 1.417 | 1.417 | 0/24 each |
| Actual initial greedy | 2.208 | 2.208 | 0/24 each |
| Sampled initial-subtracted score/min | 3.230038 | 2.688757 | — |
| Greedy initial-subtracted score/min | 6.590132 | 4.631370 | — |
| Training time | 21m27.910s | 22m06.821s | — |

Both learners complete 200,000 aggregate actions and 12,477 updates; candidate
raw frames are 799,552 versus 799,595. Each frozen arm selects the first three
natural episodes in each of eight lanes. There are no truncated selected
episodes or learner updates. The candidate's four arms total 83,272 actions,
332,902 raw frames and 67.401s, with all 357 tensors invariant in each arm.
The unchanged gate requires 90% success at clearing both walls (864 points);
it remains unmet. The initial frozen semantic traces match the control at
800 sampled and 795 greedy events, and both initial video hashes match.

The [fixed full protocol](../experiments/2026-11-03-cdp-breakout-open-loop-200k.md)
adds `open_loop_supervision={horizon:3,scale:1}` to the retained centered CDP
12M64/Auto/N8/B16/T16/H15/R16 recipe, with encoder4e-5, behavior8e-5,
dynamics4e-4, policy1 and imagined/replay critic1/1. Actor/world imitation stays
100 throughout. Teacher guidance and corrective labels are unchanged.
There is no imitation decay, crop, centroid loss, added parameter or inference
input. No weights, replay, belief, games or experience transfer from the bounded
check or another life.

From one posterior start per sequence, the auxiliary objective rolls recorded
actions through three prior steps without intervening observations. Detached
centered CNN targets supervise steps2–3; reward and continuation targets
supervise steps1–3, with component weights500/1/1 and global scale1. Reset
crossings and invalid tails are masked. This is a training objective, not an
evaluation-only comparison.

All 12,477 auxiliary reports pass finite-value, weighting and mask checks.
Mean prediction/reward/continuation losses are .155537/.296153/.030939,
giving mean weighted total78.095699. Mean valid fraction is .996540, with
range [.875,1]. These are fitting metrics; forecast accuracy and action ranking
were not measured in this comparison.

Late 50k-action replay imitation CE rises **.560514→.648492**. All 12,477 updates
have supervision, with 2,989,310 repeated labeled replay rows and equal weight
sum; maximum raw world/actor CE difference is 8e-7. Actual guidance overrides
97,749 actions (**48.8745%**) versus 86,117 (**43.0585%**) in the control.
The remaining 102,251 actions receive passive labels. Guided online last50
mean rises **67.78→84.74**, while frozen quality falls. Neither guided returns
nor replay fitting establish unaided competence or an isolated failure mechanism.

Nativec2d627f3/SHA0015dff7 and runner8946f5f3 differ from the retained
447fe50/cf4ea238 control. The current explicit
`replay_value_bootstrap=imagined` is the old implicit default and is the only
configuration normalization besides H3. Initial values match all357 tensors
against both bounded check and control; 47 pre-update events also match.
These scoped checks do not establish broad runtime equivalence or isolate
the causal effect of the loss. Seed1009 and its evaluation cohort are reused
development data, not an independent confirmation. This result does not rule
out other forms of multi-step supervision.

## Full comparison closure

All **644 pins**, first-update/cadence checks, finite reports, changed
CNN/reward/actor/value, full training pixel/action/label replay, frozen
action/outcome/frame/boundary replays and videos, host guards, curves, exposure
and comparison pass. The training replay verifies 275 completed episodes.
Ordinary frozen logs lack live pixel hashes; their replay does not establish
unrecorded live-image equality.

The original100k recovery observer succeeds12:01:22.587 UTC November3.
Independent reading passes once12:01:38.194, verifying100,000 actions/6,227
updates,357 finite tensors and four retained files. These retain weights and
metadata, not replay,RNG,games or live belief. The original learner continues
unchanged after this check.

All five original services succeed. Main/postprocessing/collection finish
**12:16:22.850 /12:16:49.658 /12:16:54.736 UTC November3**, source
`ffc4a26dc8d46d3138cbefb2c8f359b4e68fd746`. MainPID1020432 has invocation
`7c92d99e93bb4780a04e64d685c6d714`; full observer identities remain in the
protocol and private launch records. The predeclared independent terminal
reader passes once **12:17:22.865** in1.382s CPU/1.404s wrapper.

There is no full preparation or run failure, changed cap, repeated GPU stage,
replay, collector or final reader. The earlier bounded collector's parent-PID
assertion failure and separate CPU repair remain preserved below. Startup,
reader preflight and recovery CPU co-host costs remain uncorrected. No new
native build or repeated numerical/legacy capture occurs.

Root `/x/Temp/kindle-breakout-open-loop-20261103/full` and all its readers are
closed; do not repeat them. No Kindle allocation remains active at closure.
Check live state before further GPU work. Breakout and the five-game goal
remain unfinished.

| Full artifact | SHA256 |
| --- | --- |
| Declaration | `0bce56751ed8248c09de1cf643a0d3c34ffe0c09008161b8d1257977e4e0366a` |
| Learning review | `106fe1c879a6223e67234d59016dd615d2d0936326aeba9f50f2ca2cdb7c82e5` |
| Comparison | `bec1d7f7010710e4ea329634d8664c9106adea7c1fa21030c24387caee57c502` |
| Completion | `2f8503990218a6e6604571eb544cdee1fa429b45785f860823ea6ffa2d44474b` |
| Independent terminal summary | `3c7c8d75bb65d3a3a41fc37bf0d39cbd3823be9116eebc5761ec17076647bbf0` |

## Earlier bounded execution and cost check

The fresh H3/scale1 check completes **8,192 actions /32,717 frames /489 updates
in48.603s**. Its conservative200k projection is **21m45.264s**, below the fixed
30-minute gate. The1,024-action frozen restore takes .893s, with4,096 frames,
zero updates and no natural episodes. This establishes execution and early
cost; improved gameplay and forecasts remain unmeasured.

The [bounded protocol](../experiments/2026-11-03-cdp-breakout-open-loop.md)
adds existing three-step world supervision to the retained12M64/Auto/N8/B16/
T16/H15/R16,encoder4e-5,behavior8e-5,dynamics4e-4,policy1,critic1/1 recipe.
Actor/world imitation stay100; no decay, crop or centroid loss. Teacher guidance
and corrective labels remain unchanged. Future detached CNN targets at steps2–3
and rewards/continuation at steps1–3 supervise prior rollouts without intervening
observations. Existing weights500/1/1 and global scale1 apply. The earlier
Qbert comparison regressed; this is not evidence for a general benefit.

All **602 pins**,357 matching initial/invariant frozen tensors,47 matching
pre-update events,first update376,residual credit.5,finite reports,changed
CNN/reward/actor/value,full training pixel/action/label replay and host guards
pass. All489 reports satisfy the declared auxiliary weighting and valid-fraction
bounds. Mean total/prediction/reward/continuation loss is293.475813/.577952/
4.397767/.101957; mean valid fraction.995569 (minimum.9375). These are training
metrics, not prediction accuracy or task competence.

All489 updates have imitation supervision;117,048 repeated labeled replay rows
equal their weight sum. Maximum raw world/actor imitation-loss difference is8e-7.
Actual overrides are3,778/8,192 (**46.1182%**), with4,414 passive labels and4,007
teacher/actual-action disagreements. The full replay verifies21 natural training
episodes. Guided online returns are not frozen competence.

No native build, numerical test or legacy capture repeats. Nativec2d627f3/
SHA0015dff7 and runner8946f5f3 are reused from the actor-only package. CPU review
matches the entire open-loop module and four integration/metric blocks exactly
to qualifiedab80486, verifies current packaged source/runtime hashes, and reuses
the completed357-tensor/128-event disabled-mode compatibility capture. This is
scoped source/evidence agreement, not broad runtime equivalence. Native and
runner differ from the retained447fe50 control; the current explicit
`replay_value_bootstrap=imagined` preserves its implicit default.

CPU preparation passes both actual CLI paths,nine config cases,eight auxiliary
report cases,357-tensor schema and source/runtime checks in2.311s wrapper,
completing **11:39:46.137 UTC November3**, without native or emulator construction.
Original mainPID1019894/invocation`7b942f6a912842eea2e3c49f11e24cc4` starts
11:40:01 and succeeds **11:41:09.152**, source`63c347f9`. Training,frozen restore
and the original audit all succeed; no GPU or replay stage repeats.

The original terminal collector fails its added PID assertion: the wrapper
does not emit its own journal messages, while child logs and systemd start/end
records carry the correct invocation. Its source and1.051s failure remain.
A separate CPU repair uses the existing journal-identity checker, with six
valid/invalid identity fixtures and seven additional pins. It confirms one
original start/end, consistent invocation and successful service state, completing
**11:43:00.131** in1.065s wrapper. All602 original pins remain unchanged. Closure
is `complete_with_cpu_collection_repair`; the failed original collector is not
relabelled successful or restarted. No learning, frozen evaluation or replay
repeats.

Root `/x/Temp/kindle-breakout-open-loop-20261103/native-check` is closed. Keep
the original declaration/review, repair declaration and both execution receipts.
No allocation is active at closure; check live state before new GPU work. No
full learner, extension, coefficient sweep or automatic successor is selected
by this check. A separately declared full comparison is needed to measure
gameplay against70.750/143.667 and the unchanged90% two-wall gate. The five-game
goal remains unfinished.

The passing cost check now supports a separately declared
[fresh200k comparison](../experiments/2026-11-03-cdp-breakout-open-loop-200k.md).
Its protocol selects preparation with fixed recovery and four frozen arms;
the original full comparison subsequently launches and completes as reported
above. No bounded experience transfers.

| Artifact | SHA256 |
| --- | --- |
| Declaration | `28c3fdb863e82d454973d8cbdcad8f8bb24feb2ab1913c6e5cc0f5b15a3bf4be` |
| Review | `839a69779f60eb3153768e3badeff8de8fabda9a30d6e538947b46ebb587a57e` |
| Completion after CPU collection repair | `3ca98f1017a37ba170c6f7b070516eb7860246c5b993f6b5e6fdd5c7a3d729cc` |
