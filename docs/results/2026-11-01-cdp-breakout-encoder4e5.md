# Doubling the encoder rate more than doubles compact Breakout scores

Retain encoder4e-5 for the compact12M64/B16/T16/H15/R16/expert100 development
recipe. Fresh200k training completes in21m28s,with frozen sampled56.292 and
greedy104.667 versus the retained2e-5 control's25.292/42.625 in22m27s.
Both modes improve substantially per training minute. All remain0/24 two-wall
successes; the five-game goal is unfinished.

| Encoder rate | Training wall | Frozen sampled / own initial | Frozen greedy / own initial | Two-wall successes |
|---|---:|---:|---:|---:|
| Retained2e-5 control |22m27s|25.292 /1.417|42.625 /2.208|0/24 in every arm|
| Fresh4e-5 candidate |21m28s|56.292 /1.417|104.667 /2.208|0/24 in every arm|

Sampled return rises122.57% and greedy145.55%,for4.43% less measured training
time. Own-initial-subtracted return per training minute rises1.063→2.557 sampled
and1.800→4.774 greedy. Sampled remains the declared primary assessment; greedy
also improves. Each mode selects the first3 natural episodes per8 lanes=24.
Excess natural episodes and partial tails stay retained rather than changing
the selection. The four frozen arms total83,600 actions/334,244 frames/72.329s,
with zero updates and truncations. Their actual-initial traces reproduce800
sampled/795 greedy semantic events exactly.

## What changed and what this establishes

Encoder2e-5→4e-5 is the sole learner-setting change. Keep centered CDP/12M64/
Auto/N8/B16/T16/H15/R16/expert100,base/actor4e-5,dynamics4e-4,warmup1000,
actor-unimix.01,action-effects coefficient1,full18 actions and native64 resize.
Native447fe50,the exact optional-rate runner and pixel teacher are unchanged.
Guidance remains20% episode selection/cap1920,all-state corrective labels,
world imitation gradients and all action weights1. No previous weights,replay,
games,belief,RNG or bounded-check experience enter the fresh learner.

The earlier4M/B8/expert10 rate4e-5 failure remains negative evidence. This is a
distinct12M/B16/expert100 regime on reusedseed1009/evaluation4000001009,not
independent confirmation or a universal rate rule. The result supports encoder
optimization as a productive direction at this budget. It does not identify the
remaining architectural cause,measure this candidate's position readouts,resolve
the earlier optimizer numerical-protocol failure or establish five-game reliability.
No extension,rate sweep,seed confirmation or automatic successor is selected.

Final50k-action mask-corrected replay imitation CE falls.591395→.558097. These
are changing replay distributions,not held-out accuracy or the cause of the
gameplay gain. Actual99,343 overrides are49.6715% of actions;100,657 labels are
passive and52,712 executed/teacher actions disagree. Online last50 rises73.32→
103.80,but aided training returns remain separate from frozen competence.

Training completes200,000 actions/799,581 emulator frames/12,477 updates in
1287.701s. Whole main-service time is25m29.319s,including construction,evaluation
and CPU audits. CPU observers,.849s startup and.542s recovery confirmation
co-host with learning; retain these caveats without wall-time correction. The
time difference is not an isolated backend-speed measurement.

## Completed validation and evidence

All227 pins,100k/6227 recovery,357 matching control/check initial tensors,
47 pre-update transitions,357 invariant tensors per frozen arm,12,477 finite
reports,changed CNN/reward/actor/value,paired initial semantic traces,full
training input-pixel/action/label replay,frozen outcome replays/videos,guards,
curves,exposure,comparison and terminal collection pass. All2,989,423 labeled
replay rows equal the weight sum; maximum raw world/actor expert-loss difference
is8e-7. Ordinary frozen logs lack live pixel hashes; replay does not compare
unavailable live images. Video and plot hashes match their receipts,and the
comparison plot is visually inspected.

Main succeeds03:32:40 UTC November1/source2f9af49,postprocessing03:32:44 and
collection03:32:47. All five original services are terminal success;no GPU stage
repeats or host recovery occur. The earlier CPU header-rate assertion failure
is preserved and was corrected before the full declaration. Independent terminal
confirmation verifies227 pins,nine completion artifacts,four video hashes and
all five service states. No learner or GPU allocation is active at closure.

Root`/x/Temp/kindle-breakout-encoder4e5-20261101/full`,trial
`breakout-12m-64-b16-r16-encoder4e5-expert100-200k-seed1009`.

- Learning review`c6b171161766d9f70b3f9f6be4c4716b7dc8d02c993cd1a96541f1932b42b14e`.
- Comparison`5bdc537bc71cf4cfd8321ac5c9f25c81c26b56ae6281b0c1de532e031d2ede57`.
- Completion`57e5b798c505c1d139e385ae89b29bede091480abc67ff1203c6baff15719f35`.

The final/initial checkpoints,recovery metadata,replays,videos,curves,comparison
PNG/SVG and`final-summary.json` remain with the trial.
[Fixed full protocol](../experiments/2026-11-01-cdp-breakout-encoder4e5-200k.md).

## Earlier bounded execution and cost checks

The fresh12M64/Auto/N8/B16/T16/H15/R16/expert100 check completes8192 actions/
32,721 emulator frames/489 updates in46.899s. Frozen1024 actions/4096 frames
take.899s with zero updates and no natural completed episodes. Conservative200k
projection20m59s passes the30-minute gate. This establishes execution and early
cost,not a gameplay gain or measured full-run time.

Only encoder2e-5→4e-5 changes from the retained compact control. Native447fe50,
the exact retained runner,pixel teacher and all other learner settings stay fixed.
The earlier4M/B8/expert10 rate4e-5 comparison remains negative evidence; this
distinct12M/B16/expert100 test does not establish a universal rate or causal
architecture diagnosis. Seed1009/evaluation cohort are reused development inputs.
No previous experience enters the check.

All194 declared inputs,357 control-matching initial and invariant frozen tensors,
47 pre-update events,exact config/counters,finite reports,changed CNN/reward/
actor/value,full training input-pixel/action/label replay and both host guards pass.
First update376/residual credit.5 match. All489 reports are supervised;
117,051 labeled replay rows equal the weight sum,with maximum raw world/actor
expert-loss difference8e-7. Actual3792 overrides are46.2890625%;4400 passive
labels and4065 action disagreements are exposure,not unaided competence.

Service`kindle-breakout-encoder4e5-check-20261101` starts03:03:57 UTC November1,
sourcee11932e,PID732514/invocation`9d574be690dc445290d0c14c0101cfd0`,and succeeds
03:05:01 after64.243s. Terminal collection succeeds03:05:09. No failure,GPU
repeat or host recovery occurs. CPU preflight passes two actual CLI paths,
13 audit-config cases,the357-tensor schema and173 control pins in.425s with
zero native/environment construction. No GPU allocation is active at closure.

Root`/x/Temp/kindle-breakout-encoder4e5-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder4e5-expert100-check`.

- Declaration`5837c966a748c36bbb7cba3b5bc0e5968c454b91378cf4266256dfcfb36d37a4`.
- Review`e9bf3082ad7400111d9c69bd28e35478207bc8d567640e4dc9708af33d41d2c2`.
- Completion`7a410ce63e46a48bfddb4595efd8bac2b031bdaf6b3aa4b010b7040a7022d803`.

The [bounded protocol](../experiments/2026-11-01-cdp-breakout-encoder4e5.md)
selects no automatic promotion. After its completed checks, a
[separate fresh200k comparison](../experiments/2026-11-01-cdp-breakout-encoder4e5-200k.md)
tests frozen gameplay against the retained25.292/42.625 control and completes
above. The five-game goal remains unfinished.
