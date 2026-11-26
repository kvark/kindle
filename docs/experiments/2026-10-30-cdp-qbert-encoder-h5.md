# Qbert H5/R32 with the moderate encoder rate

Select one bounded fresh 12M48/Auto/N8/B16/T16/H5/R32/expert100 check with
encoder learning rate 2e-5. The completed H15/R16 rate comparison improves
greedy 11,984.375→13,785.417 and sampled 6,759.375→9,987.500 in 30m16s.
Separately, H5/R32 at the original 6e-6 rate reaches 14,050/13,331.250 in
40m24s on seed4013. These gains do not establish that combining them helps.
The original 15,000 mean / 90% first-pyramid gate remains unchanged.

Relative to the retained H5/R32 control, change only encoder 6e-6→2e-5 as a
learner setting. Keep centered CDP, 12M48, B16/T16, world microbatch16,
backprop16, coefficient1, actor-unimix0, full18 actions, replay100k,
base/actor4e-5, dynamics4e-4 and warmup1000. Preserve 20% episode continuation
guidance/cap1920, expert100, all18 action weights1, world imitation gradients
and all-state corrective labels. Use native max-pooled frames and one GPU
resize to48. Seed4013/evaluation4000004013 and the cohort are reused development
inputs; no weights, replay, belief, games, RNG or prior check experience are reused.

Native447fe50, Qbert teacher and the qualified optional-rate runner remain
unchanged. The runner also contains Breakout corrective-label plumbing, disabled
here; retain the exact diff against the H5 control. No native build or production
code change is required. Matching initial tensors and traces do not prove
complete cross-version learning parity. Relative to the new cheaper H15/R16
point, H5/R32 changes horizon, sampling, cadence, warmup duration and total
work together; do not attribute a future difference to horizon alone.

Run 8,192 aggregate actions / 978 updates, then 1,024 greedy frozen actions
with zero updates. First update376, then every8 actions, residual credit0.
Require exact config/source/counters, 357 finite initial tensors matching the
H5 control, matched pre-update semantic events, 357 invariant frozen tensors,
changed CNN/reward/actor/value, finite aligned world/actor expert losses,
measured guidance exposure, full training pixel/action/corrective-label replay
and both host guards. The existing 12M48 schema is unchanged. CPU preflight
checks both actual commands, config rejection cases, collector accounting and
cost gates, H5 control and completed moderate-rate evidence without native or
environment construction.

Caps: training180s, frozen60s, CPU audit180s, service480s. Require conservative
200k projection `training_seconds*200000/8192*1.10 <=2700` (45 minutes) before
considering a separately selected full learner. A short pass establishes
execution and early cost, not natural-episode competence. No full learner is
selected by this protocol. No retry, cap increase, rate sweep, extension,
automatic promotion or check-experience reuse. Stop on real faults/nonfinite
reports; no driver recovery. Preserve CPU failures and repair them without
repeating completed GPU stages.

Root `/x/Temp/kindle-qbert-encoder-h5-20261030/native-check`, trial
`qbert-12m-48-b16-h5-r32-encoder2e5-expert100-check`, service
`kindle-qbert-encoder-h5-check-20261030`. The prior moderate-rate learner and
all observers are terminal success; recheck live state before launch.

## CPU preparation

Preflight passes in .855s: two actual commands, 14 config cases, six collector
accounting/cost cases, all229 H5 control and235 completed moderate-rate pins,
357 schema tensors and the 978/12,454/24,954 update schedule. No native module,
agent or environment is constructed. Runtime, runner, teacher and schema are
copied unchanged from the qualified rate experiment. No preparation failure or
GPU stage has occurred at this point.

## Completed bounded check

The check completes8192 actions/32,751 frames/978 updates in95.153s,followed
by1024 frozen actions/4096 frames in.798s with zero updates and no natural
episodes. All288 pins,357 initial/frozen tensors,47 pre-update events,finite
reports,parameter changes,full replay and guards pass. Conservative200k
projection2555.387s (42m35s) passes45min. Main succeeds02:32:47 UTC October30,
source8b9936a;terminal collection02:33:13. No failure or GPU repetition occurs.
See the [result](../results/2026-10-30-cdp-qbert-encoder-h5.md). No full learner
is selected by this bounded protocol; gameplay benefit remains unmeasured.
