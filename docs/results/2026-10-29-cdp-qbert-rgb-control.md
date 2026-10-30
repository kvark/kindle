# Compatible Qbert RGB control favors the retained CDP recipe

The fixed comparison favors this CDP recipe on the reused development cohort.
CDP scores 9,654.167 with greedy actions in 26m07s of training; RGB reconstruction
scores 6,090.625 in 28m41s. CDP's greedy score is 58.51% higher with 8.93% less
training time. Both miss the original mean-15,000 Qbert gate. This is one
comparison with strong expert supervision, not five-game superiority or
independent learner confirmation.

The [protocol](../experiments/2026-10-28-cdp-qbert-rgb-control.md) fixes the RGB
budget, cost prerequisite, four evaluations and comparative decision before RGB
outcomes. The completed CDP learner is reused; no new CDP training occurs.

| Fixed policy | CDP mean / first pyramids | RGB mean / first pyramids |
| --- | ---: | ---: |
| Final greedy, primary | 9,654.167 / 23 of 24 | 6,090.625 / 24 of 24 |
| Own initial greedy | 263.542 / 0 of 24 | 263.542 / 0 of 24 |
| Final sampled, secondary | 7,569.792 / 24 of 24 | 4,335.417 / 23 of 24 |
| Own initial sampled | 182.292 / 0 of 24 | 182.292 / 0 of 24 |

Each arm selects the first three natural episodes in each of eight independent
lanes. Guidance and labels are off, with zero updates or truncations. Excess
episodes and partial tails remain recorded. Twenty-four episodes are not
twenty-four learners. RGB has one more greedy first-pyramid success, despite its
lower mean score; the unchanged task gate requires both mean 15,000 and 90%
first-pyramid success.

## Predeclared comparative decision

For each lane, subtract its own initial greedy mean from its final greedy mean,
then divide by that learner's training minutes. Pair the eight CDP-minus-RGB
lane values. The fixed 20,000 paired lane-block resamples, analysis seed
20261028, give a central 95% percentile interval of **[110.311, 206.365] score
points per training minute**, with mean difference **156.359**. All three
episodes stay together within each resampled lane.

CDP averages 359.514 points of improvement per training minute; RGB averages
203.155. All eight observed paired lane differences are positive. The lower
interval endpoint exceeds zero, CDP's absolute final greedy mean is higher,
and every required audit passes. The locked rule therefore classifies this
comparison as favorable to this CDP recipe. The interval describes this
evaluation cohort, not variation across independent learner seeds.

## Matched settings and measured cost

Both use 4M48/Auto/N8/B8/T16/H15/R16, full18/sticky .25/repeat4, seed4013,
evaluation seed4000004013, 200k aggregate actions and 24,970 updates. Keep base/
actor rate4e-5, encoder6e-6, dynamics4e-4, warmup1000, action-effects coefficient1,
actor-unimix0, replay100k, native max-pooled frames and one GPU resize. Training
uses 20% episode continuation guidance, cap1920, expert scale100, all18 action
weights1, world gradients and corrective labels at every visited state.

Only `cdp_centered: true→false`, reconstruction loss `0→1` and future-prediction
loss `500→0` change. The copied experimental RGB runner preserves the split
rates and exploration bonus that stock RGB CLI would change. Native code,
repository CLI and defaults are unchanged. Decoder versus future-predictor
heads produce different parameter counts: RGB3,432,500 versus CDP3,578,225.
All342 shared initial tensors match. Own RGB initial399 tensors also match the
bounded RGB initialization. Complete own-initial semantic traces match CDP:
1,140 sampled events and 1,354 greedy events, excluding timing and header identity.

| Cost component | Retained CDP | New RGB |
| --- | ---: | ---: |
| Aggregate training actions | 200,000 | 200,000 |
| Executed emulator frames | 799,703 | 799,695 |
| Learner updates | 24,970 | 24,970 |
| Training seconds | 1,567.221 | 1,720.979 |
| Fresh agent construction seconds | 2.346 | 2.097 |

RGB's four frozen assessments total 71,656 actions / 286,478 frames / 33.300s,
plus 6.744s of agent construction. The main service takes 40m07.319s including
training, initialization, evaluation, guards and CPU audits. GPU stages finish
00:46:27 UTC; CPU audits complete with main-service success at 00:57:02, and
terminal CPU postprocessing completes 00:57:28 on October29. These costs are
reported separately from training. The mandatory fresh RGB cost check adds
68.085s training and .502s frozen evaluation; its whole service takes1m42.696s.
Its preset conservative full projection, 30m28s, passes the 35-minute cap.

This comparison matches experience budget and settings, not parameter count,
wall time, total development spend or identical on-policy experience. CDP's
outcomes were already known and seed4013/cohort reused before selecting RGB.
Guidance exposure differs with the visited trajectories: RGB overrides90,197
actions (45.0985%), versus CDP41.383%. No check weights, replay, belief, games or
RNG are reused by the fresh full RGB learner.

## Execution and audits

Service `kindle-qbert-rgb-control-200k-20261029` starts00:16:54 UTC on October29,
source58dad39, PID219461, invocation2b57733b633540c88847636af3f62c86. All195 pins
remain unchanged. The100k recovery verifies12,470 updates and399 finite tensors.
Final audits verify exact configuration/counters, all24,970 finite reports,
positive reconstruction and zero future-prediction losses, changed CNN/reward/
decoder/actor/value parameters, and399 invariant tensors in every frozen arm.
Full frozen outcome replays/videos, training label/input-pixel/action replay,
guards, curves, coverage, exposure and the locked comparison pass. Main,
postprocessing, progress and recovery services all exit successfully.

All24,970 reports contain supervised rows;2,993,226 labeled replay rows equal
their weight sums. Maximum raw world/actor expert-loss difference is8e-7.
Labels comprise151,736 NOOP and48,264 hops, including109,803 passive states and
23,984 actual/teacher disagreements. Forty-eight of215 episode starts are
selected for overrides. Guided completed episodes average15,097.826 with46/46
first pyramids; unguided online episodes average1,980.435 with65/161. These are
training observations, not frozen competence. There are9,737 native actions
after selected episodes reach the guidance cap.

No full-run preparation failure, GPU failure, repeat, cap increase or runtime
edit occurs. Earlier bounded-preparation shape/import corrections remain in the
protocol and evidence package; neither invoked an environment or GPU stage.
Ordinary frozen logs have no live pixel hashes: their replay verifies recorded
actions, rewards, frames and boundaries, not direct equality with live images.

Native447fe50/SHA25635ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2
runs on RX7900XT/RADV NAVI31, boot a8648671-8577-4f42-9b46-a4345d3ac297,
kernel7.0.0-38-generic, AMD module9BDBC866F55BF296F1B7FA2. Use the matching
`/x/Temp/kindle-breakout-corrective-20261028/runtime` and pinned experimental runner.

Evidence root: `/x/Temp/kindle-qbert-rgb-control-20261028/full/`
`qbert-rgb4m-48-r16-expert100-200k-seed4013`. Logs, weights, videos, curve and CPU
controllers remain outside Git. The historical CDP138 inputs resolve through
the pinned archive map; original declarations remain untouched.

| Evidence | SHA256 |
| --- | --- |
| Declaration | `342419d193e3268f129384c12eb37f49abee137633ee2c27cd459eed75cbca24` |
| Learning review | `01fd5fc39e8a8a808678c5a320b95735413ea1795e311ce173d40c8f399cf729` |
| Comparison | `208ba0986c9a5d8cbd3d840d68b2420e56424110192846e96949e17ececbe21d` |
| Training replay | `1cc313ae85ed7dffef876ba37773a700ee84cd3234ec9a42609b1fb70ca39580` |
| Completion | `eeed814d96facd579919016a92535823c85df78185d2977e60b4114939f6fded` |

Retain the favorable bounded comparison and the stronger CDP policy. No
extension, confirmation, additional learner or automatic successor is selected.
Qbert still misses15k and Breakout remains weak; the five-game goal is unfinished.
The device is idle at completion. Check live state before future GPU work.
