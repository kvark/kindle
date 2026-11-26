# CDP prediction scale1000: bounded check and fresh learning comparison

The fresh200k comparison is complete and negative:77.167 greedy/42.750 sampled
versus retained143.667/70.750, at21m30.371s versus21m27.910s. Primary own-initial
gain/min falls47.111%; both paired gain/min intervals are below zero. Retain
prediction scale500. All full-run audits and original services pass; the two
earlier CPU failures and their separate repairs remain preserved. No successor
allocation is selected; original864/90% two-wall and the five-game goal remain
unfinished. [Full results](#full200k-comparison).

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-cdp1000.md) selects
one fresh8,192-action/489-update check of prediction scale500→1000. Only
`loss_scales.future_prediction` changes. The retained native447fe50, original
convolution/max-pool encoder, teacher, geometry and learning schedule remain
fixed. No native build is needed. After the completed bounded check, a separate
[fresh200k comparison](../experiments/2026-11-06-cdp-breakout-cdp1000-200k.md)
was selected and has now completed.

Earlier gradient attribution found imitation already a major representation
objective, partly aligned with CDP. This does not establish that stronger CDP
will improve control; the check tests an untried coefficient without adding
parameters, rollout steps or updates. The completed stride-two comparison
remains closed and its option stays off.

## CPU validation and preserved import failure

The first canonical-runner check fails before pytest collection in0.095063s:
`ModuleNotFoundError: No module named 'kindle._return_boundaries'`. It combines
the current runner with the older retained Python package. Preserve the exact
failure in `canonical-tests.json`; no GPU/emulator construction is reached.

A separate CPU-only package combines current Python sources with the unchanged
native binary and blocks native construction. Canonical checks pass39 tests
in0.220011s including process overhead. The private retained runner passes38
tests, with one explicit skip for the unsupported newer open-loop option,
in0.239808s. Both report zero native construction attempts. Production code and
tests are unchanged by the package repair; its source and binary receipt is
`canonical-package-repair.json`. The GPU queue uses the original retained
runtime unchanged, not this CPU fixture.

Tests cover omitted/explicit500/1000, centered and uncentered config identity,
invalid F32 weights, encoder/evaluation/restore rejection, checkpoint value
preservation and effective open-loop overflow in the canonical runner.
The exact two-command preflight passes in0.118725s, with zero native/emulator
construction,357 tensor/schema checks and47 initial-trace fixture checks.
It rejects a500-weight config and verifies1000-weight loss arithmetic against
a saved report. Those fixture results are not new learner measurements.

## Allocation and evidence

Root `/x/Temp/kindle-breakout-cdp1000-20261106` retains the private runner,
pre-edit canonical bytes, runner diff, all CPU declarations/results,
`cpu-review.json`, `runtime-reference.json`, fixed protocol and bounded helpers.
Native SHA256:
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`.
The source-copy receipt and preflight pin the exact commands and audit sources.

## Completed bounded check

The queue completes once at source957bed96. Training takes46.680678s for
8,192 aggregate actions/32,728 emulator frames and489 updates, versus the saved
47.689684s control (2.116% less observed time). This is not a paired speed result.
Construction takes2.352502s separately; the padded200k projection is1253.631498s
against the1800s gate. Conservative full-replay headroom is11.221905GiB against
the2GiB gate. Both gates pass.

All489 reports are finite and satisfy the1000-weight world loss and behavior
loss arithmetic. First update376 and residual credit.5 match the declaration.
All357 actual-initial tensors and47 pre-update events match retained. Encoder,
reward, continuation, actor and value weights change. Full independent replay
verifies all8,192 actions, pixels, frames, rewards, boundaries and expert labels,
including3,804 overrides (46.436% of actions;20% is the episode-selection rate).

Frozen restore completes1,024 actions/4,096 frames in0.922426s, with zero updates,
no guidance, preserved1000-weight config and all357 tensors unchanged. Its128
ticks per lane complete no natural episodes; this is not competence evidence.

| Training stage | Seconds |
|---|---:|
| Learning |40.404433|
| Acting |2.466933|
| Environment |2.506276|
| Observe |0.500698|
| Reset |0.094614|
| Checkpoint |0.262006|

Reported learner timing totals40.348870s, including posterior6.156417,
world training13.647480, imagination16.570146 and behavior training2.508965s.
These are wrapper/session timings, not isolated kernel costs.

The controller completes in65.909064s at18:14:27.545 UTC November6. Both host
guards pass, both GPU children exit0, and all49 declared pins plus four experiment
source files remain exact. Independent0.337578s CPU publication closure at
18:15:32.218 UTC verifies53 completion artifacts and terminal controller/child
identities1531541/1531549/1531722. No active Kindle service or render holder
remains. The completed GPU stages, audits and replays are not repeated.

Evidence is in `bounded/check/completion.json`, `review.json`, guidance/frozen
replays and guards, with `bounded/publication-closure.json` sealing source,
artifact and terminal checks. The original CPU import failure remains preserved.
No full learner was selected by that bounded declaration. Retain143.667
greedy/70.750 sampled; original864/90% two-wall and the five-game goal remain
unfinished. No automatic successor follows from the bounded result.

## Fresh200k comparison selected

The separate fixed-budget declaration selects one fresh200k/12,477-update
learner, seed1009, with100k recovery and actual-initial/final sampled/greedy
24-natural-episode assessments. Only prediction loss500→1000 changes. Favor
the candidate only if greedy raw score and own-initial gain/min improve, with
a positive lower paired eight-lane95% gain/min interval. This is a reused
development seed/cohort; even a passing result would need fresh confirmation.

The first full CPU helper preflight fails after2.567353s at the plotting fixture:
its copied path points to `check/train.jsonl` instead of `bounded/check/train.jsonl`.
The original failed invocation, plot receipt and source bytes remain retained.
Earlier helper assertions reached this plot assertion, including exact bounded/
retained expert accounting, header/config rejection,1000-weight loss arithmetic
and rejection of a500-weight total. The comparison entry fixture had already
passed11 negative-header cases.

A separate1.187419s CPU repair fixes that path and the fixture's expected
coefficient, completes only the missing plot/source checks and imports no
native module. It does not rerun the original failed helper invocation or any
completed GPU stage. `helpers-review.json` explicitly records
`passed_with_plot_fixture_repair`, linked to the preserved failure and separate
repair. The original canonical-package import failure also remains preserved.

The exact five-command preflight passes in0.343291s with137 declared pins,
zero native/emulator construction attempts and zero GPU descriptors. It checks
the fresh config and four restore paths, rejects three changed training
commands and four invalid replay paths, and validates checkpoint cadence/schema.

Root `/x/Temp/kindle-breakout-cdp1000-20261106/full`, execution source
`5b8d21a1af8d5a486aea28dc00815413720fd3b7`. Service
`kindle-breakout-cdp1000-200k-20261106` starts18:25:01.109 UTC November6,
main PID1532223/invocation`d6760fb8ef014cd194dd90a70cb5c0e5`.
Collector PID1532226/invocation`1cb0ccb509dc447ab99ba05768e358b8` and
recovery observer PID1532229/invocation`4915905c186149cb8881badc97bb90a7`
were launched independently. All three original services have now completed
successfully; their identities and terminal outputs are retained.

Startup passes18:25:07.766 UTC: all357 initial tensors match bounded and
retained, all47 pre-update events match, first update376 and exact declared
header/config change. The startup observer takes.335182s, included in training
wall time. Training, complete replay and frozen outcomes have since completed.
No extension, second learner or automatic successor is selected.

## Verified100k recovery during the original run

The saved100k/6,227-update checkpoint passes independent CPU verification at
18:36:01.673 UTC November6. All357 tensors,51,502,659 parameter/optimizer
elements, checkpoint config/counters/checksums and137 declared pins pass.
The original main PID1532223/invocation`d6760fb8ef014cd194dd90a70cb5c0e5`
remains active before and after the check. The recovery directory is0555 and
its four files0444. These files omit replay, live games/beliefs and full RNG;
they do not constitute an uninterrupted-learning resume.

The verifier takes.355836s, with zero model forwards or learner updates.
Its observer exits0 after.414311s; the main checkpoint observer takes.337602s.
Co-host CPU time remains in measured training wall time. The recovery service
is now inactive/success, with the original main and terminal collector still
active at that checkpoint observation. Training passed109,544 actions/6,824
updates at18:36:44 UTC; that observation was progress, not a frozen gameplay
result. No stage was restarted or repeated.

## Full200k comparison

Only prediction scale500→1000 changes. The same retained native/runtime,
teacher, model geometry, initial weights and scheduled updates are used.
The fresh candidate completes200,000 actions/799,557 emulator frames and
12,477 updates in1290.370570s, versus1287.910495s retained:0.191% more time.
Construction takes2.397449s separately. This development point fails retention.

| Frozen measure | Scale1000 | Retained500 |
|---|---:|---:|
| Greedy return |77.167|143.667|
| Sampled return |42.750|70.750|
| Actual-initial greedy |2.208|2.208|
| Actual-initial sampled |1.417|1.417|
| Greedy own-initial gain/min |3.485433|6.590132|
| Sampled own-initial gain/min |1.921929|3.230038|
| Final two-wall successes, each mode |0/24|0/24|
| Training time |21m30.371s|21m27.910s|

Greedy raw difference is−66.500, paired95% interval[−109.208,−30.458]. Primary
gain/min difference is−3.104699[−5.093525,−1.425874], a47.111% reduction.
Sampled raw difference is−28.000[−41.583,−14.167]; gain/min difference is
−1.308110[−1.940473,−.664421], a40.498% reduction. Only one of eight lanes
improves in each mode. These are10,000-resample intervals over the reused
eight-lane cohort, not independent training-seed reliability estimates.

Each arm uses the first three natural episodes per lane:24 episodes, zero
truncations, guidance or learning, and357 invariant tensors. Initial frozen
semantic traces match retained exactly after documented time fields are
ignored:799 sampled events and794 greedy events. All357 actual-initial tensors
and47 pre-update training events also match. No episode is substituted.

| Frozen arm | Aggregate actions | Emulator frames | Seconds |
|---|---:|---:|---:|
| Trained sampled |28,016|112,031|17.850785|
| Initial sampled |5,944|23,719|3.943718|
| Trained greedy |39,856|159,385|25.043617|
| Initial greedy |5,936|23,706|3.910018|

All12,477 finite reports,1000-weight world loss totals, behavior loss totals,
actor sharing, parameter-change checks and checkpoint/config checks pass.
The complete110.373914s CPU training audit/replay verifies every action, pixel,
frame, reward, boundary, corrective label and override. Actual overrides cover
46.1105% of actions;20% denotes episode selection. The265 completed training
episodes reach at most426 points including tails. There are190,478 actions
below216 pre-action points and9,522 at216–431, with none at432 or above.
Guided returns and these coverage counts are not frozen competence.

Training stage times are learning1146.552453s, acting62.822535s, environment
62.311666s, observe12.404550s, reset1.132791s and checkpoint.564717s. Reported
learner timing totals1133.173740s, including posterior181.635957, world training
357.372711, imagination452.467629 and behavior training66.504749s. These are
wrapper/session measurements, not isolated GPU-kernel costs.

Training completes18:46:36.222 UTC, main18:50:32.434 and terminal collector
18:51:18.640 UTC November6. All five GPU children exit0 with passing host
guards; main, recovery and collector are terminal/success. Independent
1.744101s CPU publication closure at18:52:01.151 verifies137 active/338 retained
pins,56 completion artifacts,28 checkpoint files,192 candidate/control selected
episodes and both paired gain/min calculations. No native module is imported
by that closure, and no service or render holder remains active.

`full/comparison.json`, `report-result.json`, `terminal-summary.json` and
`publication-closure.json` retain exact results and identities. Checkpoints,
logs, replay receipts, videos, curves, declarations and the two original CPU
failures remain outside Git. No full GPU phase failed or repeated; the earlier
CPU invocations remain failed with their repairs recorded separately.

Retain scale500. This result rejects1000 for this recipe/development comparison;
it does not establish a representation, gradient-conflict or forecasting cause,
or reject every use of CDP prediction training. No extension, coefficient sweep,
second learner or automatic successor follows. The original864/90% two-wall
gate and unfinished five-game goal are unchanged.
