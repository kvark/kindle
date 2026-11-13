# Fresh200k longer-sequence Breakout with the retained critic

Separately select one full comparison after the completed
[bounded B8/T32 check](../results/2026-11-03-cdp-breakout-value1-context32.md).
It completes8192 actions/481 updates in51.909s, with a conservative200k projection
23m14.028s below30min. All357 initial/invariant tensors,47 shared pre-update
events, full training replay and host guards pass. Original CPU preflight and
manifest-path failures remain preserved; repaired closure passes without a
repeated GPU stage. Do not rerun that check, its replay or either collector.

The older B8/T32 result improved greedy104.667→130.417 but lowered the then-
primary sampled result. This comparison tests its combination with the current
behavior8e-5/replay-value1 recipe, whose retained sampled/greedy scores are
70.750/143.666667. It does not assume that the gains combine.

Use exactly the bounded12M64/Auto/N8/B8/T32/fullBPTT/micro8/H15/R16/context1
configuration, encoder/base4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/
replay-value1, disagreement1, expert100/unit weights/world gradients and all-state
corrective labels. Keep sampled collection/posteriors, actor-unimix.01,
replay100k,20% episode guidance/cap1920, the unchanged full18 pixel teacher,
sticky.25/repeat4/noops0/native pooledRGB and one GPU64 resize. Native447fe50/
SHA35ad0db and runnercf4ea238 stay byte-identical. Use the qualified runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime`; no build or production edit.

Start fresh with learner1009/evaluation4000001009 and the retained eight lane
offsets. Transfer no weights, replay, games, beliefs or RNG state. Train200k
aggregate actions /12,469 updates within30min, first update504 and residual
credit.5. Save the actual initial, read-only100k/6,219 recovery with moments and
final checkpoint. Recovery omits replay,games,beliefs and complete RNG state;
it is not exact continuation. Require357 initial tensors equal to both the
bounded candidate and retained control, the full bounded pre-update trace and
the shared control prefix through action376.

Assess final and actual initial with sampled and greedy actions, without guidance
or labels. Each arm selects the first three natural episodes per eight lanes,
24 total, withzero updates/truncations and357 invariant tensors. Preserve
600k-action/300-second bounds and100k-frame episode limits; do not substitute
timeouts for natural episodes. Require both actual-initial semantic traces to
match the retained initial controls. Keep excess episodes and partial tails.

Require full training pixel/action/label replay, full frozen recorded-action
replays/videos, finite supervised reports, changed CNN/reward/actor/value,
correct cadence and target-row accounting, exact source/native/ROM/host guards,
curves, guidance exposure and score-band coverage. Training replay cap900s,
frozen replay/video cap500s each, main7200s and terminal observer7500s. The
read-only recovery observer never starts GPU work. Audit all actual replay input
paths before launch, including the manifest placed in the candidate directory.

Primary evaluation is **greedy**, fixed before outcomes; sampled is secondary.
Compare retained training1287.910495s, greedy143.666667/initial2.208333 and
sampled70.750/initial1.416667. Report absolute return, actual training time,
separate construction and own-initial-subtracted return per training minute.
Use10,000 paired lane-bootstrap resamples,seed0,with95% intervals. Favor the
candidate only if greedy return and gain/min both rise and the greedy gain/min
interval excludes zero. Check the original mean864/90%-two-wall gate directly.
No checkpoint selection, mode switch, threshold relaxation or automatic extension.

Batch/microbatch16→8 and length/backprop16→32 preserve256 sampled positions per
update but change temporal gradients, diversity, sampling, warmup and target rows
240→248 together. Cached recurrent state already carries history. Reused
seed/cohort and historical control timing limit this to development evidence;
no isolated memory effect or training-seed reliability claim. Record CPU co-host
time without correcting training duration. A promising result would need
separately selected fresh-seed confirmation before a reliability claim.

Qualify the changed audit/accounting/comparison helpers, all five actual commands,
checkpoint schemas, recovery schedule, replay argv/inputs and terminal identity
handling on CPU before launch. Resolve native metadata before replacing package
and native agent exports in command fixtures. Preserve the original bounded
failures and exact historical replay source; never rewrite prior declarations.

Root `/x/Temp/kindle-breakout-value1-context32-20261103/full`; service
`kindle-breakout-value1-context32-200k-20261103`. Check live state before launch.
No overlapping allocation, successful-stage repeat, cap increase, context sweep,
second learner or automatic successor. Stop on actual faults/nonfinite reports.
The original five-game goal remains unchanged and unfinished.

## CPU helper qualification

The changed helper qualification passes21:27:54.468 UTC November3 in3.251s.
Actual expert/exposure readers reproduce both saved bounded and retained logs;
eight expert-rejection cases,four selection cases,seven header-rejection cases,
a corrupted-loss case,357 checkpoint shapes and63 bounded pre-update events
pass. The terminal identity handling matches the already successful collector
when systemd clears its invocation field. All67 bounded/7 repair/338 retained
pins verify,with one exact historical source resolution. No native module,
model forward,emulator orGPU work is used. Five actual queued commands andtheir
replay-input paths still require separate qualification before launch.
