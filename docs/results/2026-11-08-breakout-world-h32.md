# Breakout explicit32-action world supervision: fails retention

The bounded learner and final frozen assessment are complete. This recipe
fails its fixed retention rule. Keep H0 as the broad inspectable reference.
Actor/critic and the five-game campaign remain parked;the full world-model goal
is active and unproven. No extension,collector,sweep or successor is selected.

[Protocol](../experiments/2026-11-08-breakout-world-h32.md).
Root `/mnt/data/kindle-breakout-world-h32-20261108`.

## What changed

Same qualified v18 local-action architecture and native bytes:RGB64/12M/B8,
categorical RSSM,two-image history,direct action planes,motion width16,
80 logical parameters/8,893,203 scalars/240 checkpoint tensors. Native/source
`7f797a2babebec97edf4a801e4c92e590283ad2e`,SHA256
`38680c739929c82c8e1e5a4f15bfa1277ba32170e9c7d14034c2d9c3f75ba323`;
runner `96d3a8090c74636806966e2be54ce97f5d908579`. No native build or repeated
primitive qualification. Learning/inference remain native on Meganeura+Blade;
Auto precision on AMD RX7900XT/RADV. Other host GPU work was present;no exclusive
performance comparison is claimed.

Fresh4,096 updates from zero,T48/full BPTT48/H32 keeps the same16 observed
frames. World/dynamics rates8e-4;encoder4e-4. Keep mixing.01,RGB100,KL1/.1/free-nat1,
all event scales1. Compared with H16,fewer optimizer steps,higher two rates,
longer context/credit/suffix,terminal placement and RNG/work all change together.
This is a practical coupled recipe,not an isolated horizon or rate experiment.

Raw suffix slots1,048,576 match old8,192/H16;observed slots drop2,097,152→1,572,864.
New deterministic4/2/1/1 T48 plan uses the same16 fitting/eight development streams.
Terminal arrivals span offsets16..47;all48 waiting slots remain waiting/noFIRE/
noending/noreset. There are965,072 valid suffix targets,5,460 terminal and16,857
positive reward targets. Observed replay includes6,130 terminal/26,043 positive
rows,38,712 context resets/6,086 source resets,and32,822 of32,850 fitting arrivals.
Valid exposures differ from H16;do not claim matched samples,work or compute.
No new environment actions or emulator frames. RAM/phase/lives are selection and
analysis labels only,never model inputs or losses.

## Frozen development forecasts

Same384 dense actual-initial/final origins:256 fitting/128 development,fixed
first draw. Cells below count visible balls within one resized RGB64 pixel;
support changes by endpoint. These reused seeds/families are development.

|Model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Retained H0|108/112|94/109|33/109|14/107|3/114|0/95|
|Local action H16|98/112|101/109|63/109|43/107|29/114|11/95|
|Local action H32|95/112|92/109|38/109|25/107|15/114|8/95|

Fitting remains inaccurate too:observed/1/4/8/16/32 counts are134/163,
137/177,75/180,48/176,17/171,16/179,versus local-action H16's142/163,
150/177,105/180,79/176,43/171,30/179. This is not a failure confined to new
trajectories. It does not isolate insufficient optimization,representation or
recurrent/decoder error.

All other draws,paddle/reward/continuation/brick results and actual-initial
arrays remain in `learning-review.json` and retained exports.
Pixel/image loss alone is not a physical or event-accuracy gate. At512 updates,
all fitting/development natural/contact forecast-ball accuracies were zero;
observed reconstruction was already partially fitting. That progress observation
did not select an earlier checkpoint or alter the fixed budget.

## Final fixed-action challenge

All84 groups,histories16 primary/32 secondary,eight model/oracle draws and
1/4/8/16/32 endpoints use the same cached oracle. No completed control is queried
again. Primary category-balanced survival-weighted16/32 ball energy is
0.329402470601,versus H0 .169502219090 and local-action H16
.240565683475;required <=.127126664317. Relative reduction versus H0 is
-94.335% (94.335% worse). Ending16/32 Brier is
0.000890787467/0.398760383680;
RIGHT16 mean false-ending probability 0.033947321688
(oracle zero). There are152/304 individual
nonending cases above5%,maximum 0.527980231507. Gates
use each nonending category mean;individual alarms remain retained too.

All retained oracle outcomes are alive through16 actions. Lower16-action Brier
therefore measures suppression of false endings here;it does not establish
recognition of genuine terminal events or natural-frequency calibration. The
32-action ending Brier worsens from H16 .208852 to .398760. Both lower false
alarms and higher32-action ending error must remain visible. See the
[oracle support diagnostic](2026-11-08-breakout-world-survival-decomposition.md).
Freeflight8 ball detection is64/64 NOOP,63/64 LEFT,64/64 RIGHT and47/64 REVERSE;
the last fails the90% per-command requirement.

|Original intermediate requirement|Result|
|---|---|
|observed ball|Fail|
|one step ball|Pass|
|eight step ball|Fail|
|serve noop|Pass|
|ending brier|Fail|
|ending16 brier|Pass|
|ending16 right|Pass|
|nonending false alarms|Fail|
|mean ball energy reduction|Fail|
|freeflight each plan presence at8|Fail|

Freeflight8 presence for every command,serve-NOOP false detection,all matched
point pairs,both histories and every action/horizon aggregate remain in
`branch-comparison.json`. Point/detection metrics omit survival weighting;
related pairs are not independent episodes or calibrated distribution samples.
`supplementary-metrics.json` retains every regression versus six cached controls
and all individual alarms. Against H0,errors worsen in108/210 ball-energy,
184/210 paddle-energy,201/210 reward-Brier,127/210 ending-Brier and18/209
brick-Brier cells. These related cells are not independent trials.
Oracle-assisted rescoring cannot replace the ordinary gates.

## Execution,failure preservation and scope

First CPU adapter check:two failures/28 passes because exposure accounting still
sliced32 arrivals. Separate config-length repair passes30/30,including actual
stand-in training and legacy sampler/CLI checks. Original source/log retained.
Original native qualification then fails its combined metadata/getter assertion
before any update or prediction export. Saved metadata and all240 initial
weights are exact;the direct getter serializes F32 floats more briefly. Separate
repair restores that saved zero-update initial and normalizes getter config to
F32. The repair's first CPU declaration incorrectly expected nine sealed pins
for a failed guard (actual27);a mistakenly launched controller had no queue.
Both failures remain;neither spawned a native worker. A later CPU-only source
parse caught an accidental text-replacement syntax error in the unexecuted
audit script;its source/error remain,with a separate repair before any audit.
No failed stage is relabeled.

Corrected distinct repair2 shape check passes1261 pins,all240
initial tensors/18 cached v18 arrays,18 own initial and18 trained restore arrays,
13 finite components,B*32 validity/total,required families/77 changed parameters,
240 frozen tensors,in94.066321s. Reused native derivative/value/
causal receipts are unchanged;no primitive test repeats.

The learner first report exactly matches the thirteen-component shape check.
It completes all4,096 finite thirteen-component updates,all80 weights
change,full declared plan/counts match,initial/2048/final metadata match exact
rates/T48/H32,and final weights stay finite/frozen. Dense actual-initial controls
and18 final restore arrays pass. Native learning 1515.732651s;
measured training loop 1691.448028s. The loop includes periodic evaluations and dense-final exports. Initial
construction/dense-initial exports,guard worker/poll spans,preparation and failed
stages are additional costs. Final frozen
assessment native-worker function wall 220.130800s;
CPU branch comparison 96.506064s. Native learning costs9.087% more and the
measured loop7.869% more than the earlier local-action H16 run;the new recipe
has no retained quality gain for that measured cost. These shared-host timings
are not an isolated horizon effect or controlled speed benchmark. Wrapper timings do not
isolate kernels or establish exclusive-GPU speed.

Closure verifies3453 pins,four successful guards/36 sealed pins,
original failed guard27 sealed pins,and allfive service identities terminal.
Weights,logs,arrays,curves,figures,runtime/source and original failures remain.
Three fixed first-draw freeflight/contact/life-loss causal clips retain every
supported decision and native oracle truth;all lossless decoded frames match.
They are display aids,not best draws or independent confirmation.
No new build cache exists. The failed original qualification remains failed.

Full physical trajectories,event timing/natural-frequency calibration,persistent
removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh
confirmation remain unproven. Existing oracle has no above-top-brick endpoint.
Intermediate gates—even if passed—cannot establish a perfect world model or
finish the five-game task.
