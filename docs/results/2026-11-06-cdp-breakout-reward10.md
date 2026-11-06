# Stronger reward supervision lowers matched-seed Breakout learning efficiency

The full200k comparison scores52.167 greedy/42.625 sampled versus matched
75.250/44.708 at reward scale1. Primary own-initial gain per training minute
falls31.086%, despite.802% less training time. All0/24 two-wall. Keep scale1;
[full comparison and limitations](#full200k-comparison) below. The original
five-game goal remains unfinished.

## Bounded qualification

The [bounded protocol](../experiments/2026-11-06-cdp-breakout-reward10.md) changes
only the existing posterior reward-prediction loss scale from1 to10. One fresh
seed1009 run completes **8,192 actions /32,730 emulator frames /489 updates in
46.723357s**. Its padded200k projection is1,254.777659s (20m54.778s), below the
fixed30-minute screen. Historical bounded control takes47.689684s; this short
comparison does not establish a speedup or useful gameplay learning.

The1,024-action greedy restore takes.896658s with zero updates and357 invariant
tensors. All357 initial tensors and47 pre-update semantic events match the
retained1009 control. Encoder, reward, continuation, actor and critic parameters
change. All489 weighted world and unchanged behavior totals pass; an independent
saved-log recount finds maximum world-total absolute error.00013553 within the
declared relative tolerance. Full CPU replay reconstructs every training action,
reward, frame, boundary and corrective pixel label;3,788 overrides are recorded.

Both canonical and private runners pass34 targeted CPU cases with zero
native/emulator construction. Actual queued commands and audit rejection fixtures
pass in1.476s. Native447fe50/SHA35ad0db and the original teacher remain unchanged;
the private runner adds only the fresh-training reward override. Both GPU guards,
62 allocation pins and338 retained pins pass. No GPU stage fails or repeats.
Execution source080c8b94, root `/mnt/data/kindle-breakout-reward10-20261106`.

The main completes21:31:10.844 UTC November6; collector21:31:15.605 and separate
CPU closure21:31:44.531. Both original services terminate successfully. Closure
recounts losses, initial/frozen tensor bytes, pre-update events and cost arithmetic.

Initial preparation for the subsequent full comparison rejects one closure-log
hash: the closure script hashed its open stdout before printing its final result.
The original failed helper execution and source remain preserved. A separate
resolution verifies that only this log differs and its complete JSON equals the
terminal summary without the file map; every other sealed file is unchanged.
The corrected full-helper check passes, including11 corrupted-header cases,
comparison direction checks, loss rejection fixtures, retained checkpoint schema
and an isolated full plotting fixture. No native/GPU stage is repeated or
original receipt rewritten.

This qualifies cost and execution, not natural-episode competence. A separate
[full200k protocol](../experiments/2026-11-06-cdp-breakout-reward10-200k.md)
uses development22003 with its matching retained final/actual-initial controls
on cohort4000022003. No experience from the bounded check is transferred. The
original864/90%-two-wall and full five-game goals remain unfinished.

## Full200k comparison

The [fixed full comparison](../experiments/2026-11-06-cdp-breakout-reward10-200k.md)
is complete and negative. **Retain reward loss scale1.** Only posterior reward
prediction scale1→10 changes against the retained22003 learner. Both frozen
point scores fall; the small reduction in measured training time does not offset
the loss in primary learning per minute.

| Measure | Retained22003 | Reward scale10 |
|---|---:|---:|
| Training wall time |21m30.822s|21m20.465s|
| Greedy return |75.250|52.166667|
| Sampled return |44.708333|42.625|
| Actual-initial greedy/sample |2.291667/1.875|2.291667/1.875|
| Greedy own-initial gain/min |3.391251|2.337042|
| Sampled own-initial gain/min |1.990980|1.909463|
| Two-wall success, either final mode |0/24|0/24|

Primary greedy gain/min falls **31.086%**, difference−1.054208 with paired95%
interval[−1.936771,−.046998]; only2/8 lanes improve. Raw greedy difference is
−23.083333,[−42.001042,−1.458333]. Sampled gain/min falls4.094%, difference
−.081517,[−.298401,.147666], with3/8 improved lanes; raw sampled difference is
−2.083333,[−6.708333,2.791667]. Training takes.802% less time. Intervals use
10,000 resamples over eight paired reused lanes, not independent training seeds.
This one coefficient/seed does not isolate a representation or optimizer cause
or reject all reward-supervision recipes. Retained1009's94.625/70.500 on this
cohort is additional context, not the primary comparator.

The candidate completes **200,000 aggregate actions /799,583 emulator frames /
12,477 updates in1,280.464522s**. All357 starting tensors and47 pre-update events
match retained22003. Frozen initial/sample and initial/greedy traces match all
845/704 semantic events respectively; only episode wall times differ. Each
frozen arm selects three natural episodes in each of eight lanes,24 total, with
no guidance, labels, updates or truncation and357 invariant checkpoint tensors.
All action/outcome/frame/boundary replays and videos pass.

The full training replay takes109.254s and reconstructs257 completed episodes,
every action, reward, frame and corrective pixel label. Teacher overrides cover
75,923 actions (37.9615%); repeated imitation exposure totals2,989,685 rows.
Coverage reaches maximum cumulative score405, with827 native actions in216–431
and no state reaching432. These guided training statistics do not establish
frozen competence or a coverage mechanism. Learning accounts for1,136.163s and
Atari stepping62.480s of training wall time.

Execution source3be21d90 launches21:37:11.656 UTC November6, original PID1552095 /
invocationf6402af01ae946019805b26479f206de. The100k/6,227 recovery passes independent
CPU verification21:48:13.435 in1.569s:357 finite tensors/moments,263 unchanged
pins,0555 directory/0444 files and unchanged original learner identity. The
verifier performs zero model forwards or learner updates.

The main completes22:02:30.394 UTC and collector22:02:45.029. The separate CPU
closure finishes **22:03:43.920** in4.652s. All263 allocation/338 original retained
pins,55 completion artifacts and28 candidate checkpoint files pass. Independent
recount verifies192 candidate/control evaluation episodes, both raw and gain/min
paired intervals, all12,477 learner reports and loss arithmetic. Maximum total
errors are.00010061 world/.000027055 behavior within their declared tolerances.
All five guarded GPU children and three original services terminate successfully;
no render holder remains. No learner, GPU stage or collector repeats.

Root `/mnt/data/kindle-breakout-reward10-20261106/full` is closed. Keep the retained
recipe and its measured seed/cohort variation; no extension, coefficient sweep,
second learner or automatic successor is selected. Original864/90% two-wall and
the full five-game goal remain unfinished.

## Preserved full-preparation failures

Full-helper checks pass in4.343s. The first actual-command preflight fails before
native/emulator construction because its forbidden-API test double lacks the
`act` signature inspected by the runner. The separate repaired fixture adds only
that signature, still forbids execution and is AST-identical when this method is
removed. All five commands and four wrong-command checks pass in1.110s with zero
native/emulator construction and no GPU descriptor. Original source/failure/logs
remain pinned. The separate launch qualification verifies repair provenance,
then calls the unchanged declared launcher; no training code, queue or cap changes.

A read-only live-publication reader initially used system Python, which lacks
NumPy; it stopped on import before verification or writes. The failure receipt
is preserved. The corrected reader uses the declared Python environment and
verifies the running identity and pins; no learner or collector is affected.
