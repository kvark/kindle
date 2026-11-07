# Weaker representation regularization lowers frozen Breakout scores

The fresh200k comparison finishes in **21m23.007s**, close to the retained
control's21m27.910s. Reducing representation KL scale0.1→0.01 lowers primary
**greedy score143.667→93.208** and sampled70.750→35.792. All final arms remain
zero of24 two-wall successes. **Retain representation scale0.1** and the existing
recipe. No extension,coefficient sweep,second learner or automatic successor
is selected. Original864/90%-two-wall and the five-game goal remain unfinished.

## Full200k comparison

The [fixed full protocol](../experiments/2026-11-04-cdp-breakout-representation001-200k.md)
changes only the existing posterior KL weight. Dynamics KL1,both free-nat floors1,
native447fe50/SHA35ad0db,the corrective runtime and all other retained settings
stay fixed. This adds no parameters or computation. Keep centered
CDP12M64/Auto/N8/B16/T16/H15/R16,replay-value1/expert100,base/encoder4e-5,
dynamics4e-4,behavior8e-5,sampled collection/posteriors,actor-unimix.01 and20%
episode guidance/cap1920. Teacher,all-state labels and actual-command feedback
are unchanged. No weights or experience transfer from the bounded check.

| Measure | Retained weight0.1 | Candidate weight0.01 |
|---|---:|---:|
|Training actions /updates|200,000 /12,477|200,000 /12,477|
|Executed emulator frames|799,595|799,591|
|Training time|21m27.910s|21m23.007s|
|Sampled final /actual initial|70.750 /1.417|35.792 /1.417|
|Greedy final /actual initial|143.667 /2.208|93.208 /2.208|
|Sampled /greedy two-wall successes|0/24 /0/24|0/24 /0/24|
|Sampled initial-subtracted return/min|3.230|1.608|
|Greedy initial-subtracted return/min|6.590|4.256|

Greedy return falls35.12% and gain/min falls35.42%. Paired greedy return
difference−50.458 has95% lane-bootstrap interval **[−84.500,−10.750]**;
gain/min difference−2.334505 has interval **[−3.921731,−.481545]**. Sampled
return difference−34.958 has interval[−47.375,−22.500]; gain/min difference
−1.622487 has interval[−2.201062,−1.041611]. The locked preference rule fails.
These fixed10,000-resample/seed0 intervals describe eight reused evaluation
lanes,not training-seed reliability. This is one reused learner seed1009/cohort
with historical timing,not an isolated mechanism or broad reliability claim.

## Fitting and exposure

Raw posterior/prior KL increases while reported imitation loss decreases.
Neither change establishes better visual precision or explains the gameplay
regression. These unweighted report means come from each learner's changing
own replay,not a fixed held-out dataset.

| Training window end | Control raw KL | Candidate raw KL | Control expert loss | Candidate expert loss |
|---|---:|---:|---:|---:|
|50k|6.215493|8.196061|1.210259|1.191953|
|100k|11.217215|14.595532|.659261|.638892|
|150k|11.457282|15.400861|.598579|.578729|
|200k|11.815355|16.080293|.559936|.541684|

Floored representation KL means are6.217546/11.217220/11.457290/11.815389
for control and8.196297/14.595532/15.400861/16.080293 for the candidate.
Expert loss above is world-reported,including zero-mask rows; the separately
normalized final-window label cross entropy is.542237 for the candidate.
All12,477 reports are finite and have the correct weighted world total,
with maximum absolute sum error.000106542. All inspected CNN,reward,
continuation hidden/output,actor and value parameter entries change.

Actual teacher overrides are84,148/200,000 actions (**42.074%**). Native actions
split114,399 before the cap and1,453 after it. Fifty completed episodes with
teacher actions average205.560;200 without teacher actions average17.010.
Guided online scores are not frozen competence. Score216–431 exposure is5,948
actions:4,549 teacher and1,399 native. No training action reaches432; maximum
including tails is406. These score bands are not verified wall events.

## Full execution and closure

All774 current and338 retained pins pass with exact historical replay-source
resolution. All357 initial tensors match bounded and retained controls,with47
matching pre-update semantic events. The100k/6,227 checkpoint and independent
2.047s verification pass,with read-only weights/moments. All four frozen arms
select24 natural episodes,with zero updates/truncations and357 invariant tensors
per arm. Actual-initial sampled/greedy traces match799/794 retained semantic
events. The four arms total73,360 actions in47.355s. Full training pixel/action/
label replay,frozen action/outcome/frame/boundary replays/videos,host guards,
curves,coverage,exposure and comparison pass. Ordinary frozen logs do not record
live pixel hashes; no unrecorded live-pixel equality is claimed.

Full CPU helper qualification initially fails an exact floating-point sum
comparison: sequential and compensated totals differ by at most9.095e-13.
The original source/exit1 remain pinned. A tolerance repair resumes only
unfinished checks; completed fixtures are not repeated. All five queued-command
checks then pass before native construction. The earlier bounded flag-spelling
fixture failure below also remains. No training,evaluation,replay or collector
stage fails or repeats; no cap changes or host recovery occur.

Source`ca33c9fd`,mainPID1152865/invocation
`80605bcd9c744663bf4ee9f4e0cb82f5` starts09:14:08 UTC November4. Main succeeds
**09:39:32.369**,original collector succeeds **09:39:42.775** after3.958s
verification,and separate4.155s saved-result review completes **09:41:35.196**.
All three original services are terminal success;57 completion artifacts,
checkpoint hashes and recorded losses pass the saved-result review.

Construction takes2.405s separately. Learning-wrapper time is1,138.029s,
acting63.230s and environment62.943s; these timers do not isolate GPU kernels.
Training time is.38% lower than historical control without an established speed
difference. Startup/recovery observers and read-only CPU co-host reviews remain
included without wall-time correction. No GPU allocation remains active at
closure; check live state before future work.

Evidence root: `/x/Temp/kindle-breakout-representation001-20261104/full`.
Do not repeat completed stages. No successor is selected.

| Full-run artifact | SHA256 |
|---|---|
|Declaration|`4f0651f1b33d1da1adda15491f1526c0c755fa72d4ac77a985dbd222b48dbe80`|
|Comparison|`b3b9f7d22f3c475b8ae670ea3b1de244067cfb15a20527a514fd8b96eb5259ba`|
|Completion|`e206691d31326c57d7f97dafa8e2ba2df8228ff55b1232372a196b4ad937ef6b`|
|Terminal summary|`1b1254eaa04f542bc288dc4d612697651b5594a044e7f415b6691e3ab1d58472`|
|Saved-result review|`0257260f3569f99214d6c5b4a52935619b6c04c9ffe786b8883ba8fa917262ba`|

## Earlier bounded execution and cost check

Reducing the existing representation KL weight from0.1 to0.01 completes
**8,192 actions /32,719 emulator frames /489 updates in47.300s**. The retained
bounded control took47.690s. The conservative200k projection is **21m10.262s**,
within the30-minute gate. This establishes execution and early cost; it does
not establish better gameplay or a speed difference. No full learner was selected
at the bounded closure; the separately selected comparison above follows.
The original Breakout864/90%-two-wall gate and five-game goal remain unfinished.

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-representation001.md)
changes only `loss_scales.representation`. Dynamics KL1, both free-nat floors1,
native447fe50/SHA35ad0db and the corrective runtime remain unchanged. Keep
CDP12M64/Auto/N8/B16/T16/H15/R16, replay-value1/expert100, base/encoder4e-5,
dynamics4e-4, behavior8e-5, sampled collection/posteriors, actor-unimix.01,
20% guided episodes/cap1920 and all-state corrective labels. All optional
prior-continuation, terminal-weight and multi-step losses stay off.

The hypothesis is that weaker posterior regularization might help representation
and control. Retained full-run raw KL means6.215/11.217/11.457/11.815 over successive
50k windows show an active aggregate penalty, while older two-batch gradient
measurements show opposition to imitation in some groups. These observations
do not identify a causal limitation. The new weight adds no parameters or
computation, and can change learned states, optimization and later experience.

## Execution and early fitting

All706 current and338 retained pins pass, resolving the exact historical replay
helper bytes. All357 initial tensors match the retained learner and all47
pre-update semantic events agree. First update376, final489 and residual credit.5
match the fixed cadence. Every report is finite; independent weighted-world-total
recount has maximum absolute error.000106542. All inspected CNN, reward,
continuation hidden/output, actor and value parameter entries change.

Frozen restore completes1,024 greedy actions /4,096 frames in.872s with zero
updates,357 invariant tensors and no completed natural episodes. Full training
pixel/action/label replay and host guards pass. Actual teacher overrides are
3,906/8,192 actions (47.6807%); guided online scores are not frozen competence.

| Unweighted learner-report mean | Retained weight0.1 | Candidate weight0.01 |
|---|---:|---:|
|All489 raw KL|2.687052|3.941760|
|All489 floored representation KL|2.699389|3.943247|
|All489 expert cross entropy|2.124006|2.122716|
|Last100 raw KL|3.453213|3.998163|
|Last100 expert cross entropy|1.416821|1.415500|

These are each learner's changing replay, not a fixed held-out comparison.
The loss summaries neither establish gameplay benefit nor explain a mechanism.
Construction takes2.466s separately; learning-wrapper time is40.577s of47.300s,
acting2.483s and environment2.531s. These timers do not isolate GPU kernels.

## Qualification, failures and closure

The runner exposes an optional positive finite F32 override for fresh training;
omission preserves defaults and restore rejects overrides. Both canonical and
private qualified runners add exactly ten lines. No native build occurs.
CPU checks cover default/explicit/reduced configurations, invalid values,
restore/evaluation rejection, actual queued commands, loss arithmetic, cadence,
label/guidance bounds, replay argv, checkpoint corruption and terminal collection
before native-agent or emulator construction.

The original canonical test run records34 passes and one failure: a new negative
fixture incorrectly spells greedy evaluation `--evaluate-greedy`. Its exact
source and exit1 receipt remain preserved. Only that fixture changes to the
supported `--evaluate --greedy`; the failed case then passes, and the first
packaged run passes all35 targeted cases. Production source is unchanged by
the repair. Preflight passes in1.539s. The completed GPU and audit stages have
no failures or repeats; the earlier CPU fixture failure is not erased.

Source`ae529140`, mainPID1152316/invocation
`2e74adbf10b54bd9ba6fcc6691f39bd7` starts09:02:55 UTC November4. Main succeeds
**09:04:06.215**, original collector succeeds **09:04:18.505**, and independent
6.424s CPU closure finishes **09:04:41.233**, verifying89 saved artifacts.
A separate.025s saved-report summary completes09:05:13.803. Both services are
terminal success. No GPU allocation remains active; check live state before work.

Evidence root: `/x/Temp/kindle-breakout-representation001-20261104`.
Declaration SHA256`b3e0c169c1ae88e5e2d5edf0cb9f27a697affbf1adda52bf98109d73f60087c6`;
completion`67b7262d6acf9ad29dcf66d96d00201c4930ab607454bf55762b579797b99353`;
terminal summary`5b392230c202ecd3c9c3d69d4aeff8a6c2f360f4c25f8f2f4e3c47e00f07a69d`.
Original failure, repair, historical-source snapshots, checks, checkpoints,
replay, guards and closure receipts remain retained. Do not repeat completed
stages or transfer this bounded experience. The bounded protocol required separate
selection,100k recovery and final/actual-initial sampled/greedy24-episode
assessments against retained143.667/70.750 before a full result. The completed
comparison above fulfills those requirements without repeating bounded stages.
