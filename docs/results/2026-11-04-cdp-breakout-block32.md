# Intermittent Breakout guidance lowers frozen gameplay at similar cost

The fresh200k block-guidance learner scores **74.875 greedy /41.250 sampled**,
below the retained episode-guidance143.667 /70.750. Training takes21m11.408s,
1.28% less observed time, while greedy own-initial-subtracted return per minute
falls47.96%. All final arms have zero of24 two-wall successes. **Retain20%
whole-episode guidance.** The block option remains off by default; no extension,
second learner, probability/block-length sweep or automatic successor is selected.

The [full fixed protocol](../experiments/2026-11-04-cdp-breakout-block32-200k.md)
was selected separately after the bounded check below. Its source is
`b07ebbcc82c3ba214018f05db7f59ce41fd84844`; full evidence remains under
`/x/Temp/kindle-breakout-block32-20261104/full`.

## Full200k comparison

| Measure | Retained20% episode guidance | Candidate50%32-action blocks |
|---|---:|---:|
| Training actions / updates |200,000 /12,477|200,000 /12,477|
| Emulator frames |799,595|799,610|
| Training time |21m27.910s|21m11.408s|
| Frozen greedy / own initial |143.667 /2.208|74.875 /2.208|
| Frozen sampled / own initial |70.750 /1.417|41.250 /1.417|
| Greedy own-initial-subtracted return/min |6.590|3.429|
| Sampled own-initial-subtracted return/min |3.230|1.880|
| Final two-wall successes, each mode |0/24|0/24|
| Actual override fraction |43.0585%|51.4930%|

Primary greedy return falls47.88%, sampled41.70%. The fixed candidate-selection
rule fails: both absolute greedy return and gain/min are lower. Paired eight-lane
bootstrap intervals use10,000 resamples, seed0:

| Candidate minus retained | Mean |95% interval |
|---|---:|---:|
| Greedy return |−68.792|[−104.542,−38.625]|
| Greedy gain/min |−3.161|[−4.826,−1.759]|
| Sampled return |−29.500|[−38.958,−18.500]|
| Sampled gain/min |−1.350|[−1.792,−.836]|

These are intervals over lanes of one reused development cohort, not training-
seed reliability estimates. Probability and temporal distribution of guidance
change together. The native model, teacher, labels, cap, loss scales and learning
cadence remain fixed; the runner and guidance-module hashes change as declared.
No isolated mechanism is established. The original864/90%-two-wall gate fails.

## Training exposure and cost

The independent block ledger matches all25,000 vector transitions. It records
102,986 teacher overrides and97,014 native actions, with1505 native-to-teacher
and1584 teacher-to-native switches within episodes. All270 started episodes
contain a selected block. There are262 completed episodes and eight partial tails;
their actions and rewards reconcile exactly. All native actions occur before the
1920-action cap; there are zero native-after-cap actions.

The candidate reaches pre-action score216–431 for878 actions:410 teacher and468
native. The retained control has5947 such actions, including1262 native, from the
[closed score-coverage diagnostic](2026-11-02-cdp-breakout-score-coverage.md#training-coverage).
Neither reaches432. Candidate maximum accumulated score is363 including tails;
two episodes reach216. These score bands are exposure measurements, not verified
wall events. This schedule does not expand the measured later-score experience.

Late50k-action replay teacher cross-entropy is.552181 versus retained.560514.
The states differ, so this is not a matched-data fitting comparison. Slightly
lower own-replay loss accompanies worse frozen gameplay. Mixed-control completed
episodes average29.710; the online last50 mean reaches65.1. Neither is an unaided
competence measure.

Construction takes2.429s, reported separately. Wrapper timers attribute1126.118s
to learning,63.438s to acting and63.017s to environment stepping. They do not
isolate GPU kernels. The.330s startup check,.303s100k save check and.453s independent
recovery verification co-host with training; actual wall time is not corrected.
The small historical cost decrease does not establish a reliable speedup.

## Full-run validation and closure

All357 actual initial arrays match both bounded and retained controls. All47
bounded pre-update semantic events reproduce. The100k/6227 recovery has finite
weights/moments and read-only files; independent verification passes02:01:03 UTC
with all126 pins unchanged and the original learner identity. Recovery still
omits replay, games, live beliefs and full RNG state.

All12,477 reports are finite, cadence/label bounds pass, and CNN/reward/actor/value
parameters change. Complete training pixel/action/label replay and its independent
block ledger pass; the training audit takes111.795s. Four frozen arms total74,072
actions/296,131 frames/48.424s, with zero updates or truncations and all357 arrays
invariant per arm. Each selected cohort contains24 natural episodes. Sampled and
greedy actual-initial runs retain28 and27 completed episodes respectively, with
the first three per lane selected. Their799/794 semantic events reproduce the
retained controls; excess episodes and tails remain in the evidence.

All full recorded-action replays/videos, native/source/ROM checks, host guards,
curves, exposure, coverage and the complete comparison pass. Changed CPU helpers
passed3.764s before launch, including actual bounded/retained exposure fixtures,
the independent ledger and the complete comparison with11 corrupted headers.
All five queued commands passed. The bounded qualification failures below remain
preserved; the full learner and its original services have no failures or repairs.

MainPID1096988/invocation`a6f5a52d4afa44e79327ef980528934a` launches01:50:02 UTC
November4. Training completes02:11:19; main completion is02:15:09 and original
terminal collection02:15:35. Main, recovery observer and collector all succeed.
All126 current/338 retained pins and52 completion artifacts pass. No successful
stage or GPU work repeats; no host recovery occurs. No learner or GPU allocation
is active at closure. Do not restart this learner or its collectors.

| Full-run artifact | SHA256 |
|---|---|
| Declaration |`469d78d168ca7605d5bf44c507133602b99afc20c61413ca8cad25e7e98e951a`|
| Training log |`2e1cc57f2d617c99187003dbd7c0423a27a98a62ab9648c0c0a37fb723f4fcf8`|
| Recovery verification |`5e30ab7cddfb40b5a170566749fd09becd1fad057dccbafce0a11a0cc1630266`|
| Comparison |`d6958427743a71d73436a5a5788b3216904ddae426d560c0b76ffd67ad08c390`|
| Completion |`9ba24b079cf37870857f81b517cae4e15eeb12e4ab152a970da324156ed06d22`|
| Terminal summary |`8ff3970844396b2bab3d7fc796e45973746d15e85ea6534d8059521472fd746f`|
| Exposure |`9bb251df09db0e3de4975bf8d34e7e76ce644fe6d8191a3da7f2f500813111ae`|
| Score coverage |`1d1a4c8aaaebacf8035248bec9976cfc8909bec9915c1870fbbe963f2a11027a`|

The tested intermittent schedule does not turn the teacher's local rescue ability
into better learned control. Keep the existing teacher and retained episode
schedule; this result does not rule out every corrective-control approach. The
five-game goal remains unfinished.

## Earlier bounded execution and early-cost check

The optional32-action block schedule completes8192 fresh actions and489 updates
in **46.934 seconds**. Its padded200k-action projection is **21m00.426s**, below
the fixed30-minute gate. A1024-action frozen restore takes.884s with zero updates
and all357 tensors unchanged. This establishes execution and early cost;
gameplay benefit was still unmeasured at this stage. The full comparison above
was selected separately afterward.

[Fixed protocol](../experiments/2026-11-04-cdp-breakout-block32.md), source
`e7c3598bf1b86ff69c5d2bc4ff3d984e76bd4bf0`, root
`/x/Temp/kindle-breakout-block32-20261104`.

### Motivation and implementation

The existing teacher rescues84/120 fixed current-policy failure states in the
[completed takeover comparison](2026-11-04-cdp-breakout-command-takeovers.md).
The command-dependent candidate rescues79 and fails its selection rule. Earlier
[50% whole-episode guidance](2026-11-03-cdp-breakout-guidance50.md) lowers frozen
scores. This check therefore changes when the existing teacher acts: independently
select each32-action block with probability.5, while retaining the1920-action
episode-age cap. Previously the retained learner selected whole episodes with
probability.2. Both probability and temporal distribution change; the comparison
cannot isolate either effect.

Every state still receives a corrective label. Teacher history and feedback from
the action actually executed continue through native blocks. Each lane has its
own RNG; only completed lanes reset. A block draw occurs when its first action
is requested, avoiding unused next-block draws at terminal boundaries. Episode
`guided` now means at least one selected block, not exclusively teacher control.
Started/guided blocks and actual overridden actions are reported separately.

The option is off by default. Native447fe50, all three teacher classes, response
coefficients and the original shared episode scheduler remain unchanged. The
separate runtime packages the new Breakout schedule and audit recognition with
a minimal block-option patch to the old qualified runner. Canonical minimal4
support remains available; this check uses full18. No native rebuild occurs.

The retained recipe stays centered CDP12M64/Auto/N8/B16/T16/H15/R16/context1,
encoder/base4e-5,dynamics4e-4,behavior8e-5,replay-value1,disagreement1,expert100,
unit action weights/world imitation gradients, sampled collection/posteriors,
actor-unimix.01 and replay100k. Native pooled frames receive one GPU64 resize.
Training uses development seed1009 with fresh experience; frozen seed4000001009.

### Bounded results

| Measurement | Result |
|---|---:|
| Training actions / emulator frames |8192 /32,736|
| Updates / first update action / residual credit |489 /376 /.5|
| Training / construction time |46.933691s /2.343108s|
| Retained bounded training time |47.689684s|
| Padded200k projection |1260.426s|
| Frozen actions / frames / updates |1024 /4096 /0|
| Frozen time / natural completed episodes |.883960s /0|
| Actual overrides / native actions with labels |4050 /4142|
| Actual override fraction |49.4385%|
| Started / guided blocks |273 /133|
| Native-to-teacher / teacher-to-native transitions within episodes |61 /65|

The independent RNG/reset/cap reconstruction agrees with all1024 vector
transitions and logged counters. All29 started episodes contain a selected
block;21 complete during training. Those aided online trajectories do not measure
unaided competence. Selection probability, selected-block fraction and action
override fraction are different quantities.

All489 reports are finite, with117,152 labeled replay rows and matching total
weight. Teacher recommendations include2570 NOOP,1111 FIRE,2008 RIGHT and2503 LEFT.
There are3615 teacher/executed-action disagreements, all on native-controlled
actions. Maximum reported world/actor expert-loss difference is8e-7. These are
execution diagnostics, not evidence of a useful final policy.

All357 actual initial arrays match the retained current-critic control. The first
32 vector transitions/256 aggregate actions exactly reproduce the previous.5
episode-guidance bounded run before block resampling. Later pre-update experience
is allowed to differ. CNN, reward, actor and value parameters change during
learning. All357 final arrays remain unchanged through frozen evaluation.

Full training pixel/action/label replay, checkpoint schemas, accounting, source
identities and both host guards pass. No host recovery or repeated GPU stage
occurs. Wrapper timing attributes40.534s to learning,2.533s to acting and2.562s to
environment stepping. These timings do not isolate GPU kernels. The small
historical bounded-time difference is not a reliable speedup claim or a guarantee
of full-run cost.

### CPU validation and preserved failures

The initial broad canonical run records714 passes andtwo failures in the optional
replay-bootstrap test: the local native metadata lacks the newer
`replay_value_bootstrap` field. Both reproduce against the unchanged HEAD runner;
the original result and baseline output remain saved. This check uses the
retained native configuration and does not enable that option.

The first packaged test attempt records21 passes/32 failures: the canonical
Breakout constructor passed `expected_action_count` to an older shared base that
does not accept it; one additional CLI fixture expects newer error wording. The
full18 constructor now uses the existing default argument path; explicit4-action
mapping retains the newer parameter. The original packaged output and receipt
remain. No GPU work had started.

After the correction,94 canonical relevant tests pass in.85s, including minimal
and full action spaces, actual runner/replay feedback, resets and malformed
guidance. Packaged41 schedule/audit/CLI tests pass in.11s;12 legacy CLI cases are
deselected because the packaged runner has its original wording. The separate
actual-command preflight covers training/restore arguments, wrong probability/
cap/block length, checkpoint contracts, replay argv and terminal collector.
It passes in1.753s with zero real agent/emulator construction and no GPU descriptor.
Legacy and new episode modes match1170 actions across two teacher modes and three
probabilities, including partial resets, cap release and executed-action feedback.
No unchanged native Rust tests or build are repeated.

### Completion and evidence

Main service`kindle-breakout-block32-check-20261104` launches01:40:38 UTC November4,
PID1096509/invocation`f7460ff6c8894809ba30655fa12c5aef`. It completes01:41:42
after63.827s; original collection completes01:41:50. Both services are terminal
success. All61 current and338 retained pins pass; the one changed historical
replayer is resolved from its exact saved source. No learner or GPU allocation
remains active at closure.

| Artifact | SHA256 |
|---|---|
| Native |`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`|
| Packaged runner |`acee3889ce90b199437e97b3f19f1c8f5bdc547bae631b5430418a93f251b721`|
| Breakout guidance |`498f8f7b913fbc30fdf97faa7b74c5e4d340bcb486ad01c6ecb3853099dc3a6e`|
| Shared episode schedule |`12f671f5043d6736a1c96e78e9510d663e8059d10f6f80518df73ab6e0029191`|
| Declaration |`bd31fadf03bff9ce1ae66ab73f3ff30003e58cbedce66765560007e4ccd8c5f2`|
| Review |`5bcf3e89fd1f895cc1fca6daf33c2a4c2695b305793bcf665e1c94b33d323ffb`|
| Full training replay |`a8df075cb9a51c3ae5ed781cee6e98aafc6731fa18bded1f046b190bad919e1f`|
| Completion |`1ba0f86dacfe66c2623ecfc4959f3e01a35649a0db7ed9020f4e38316e4bc293`|
| Terminal review |`7b029e8abf90ca1f4e134ea06dd5cd3794c8e49c3f560fde7343e4ac4e48a92b`|

The bounded check required separate full-run selection, fresh experience,100k
recovery and four24-natural-episode final/actual-initial arms against retained
70.750 sampled/143.667 greedy. It did not itself select a full learner, sweep,
extension or automatic successor. The separately selected comparison is reported
above; the original864/90%-two-wall and five-game goal remain unfinished.
