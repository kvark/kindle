# Qbert H5/R32 with encoder2e-5 does not improve the primary score

The completed fresh comparison trains **200,000 actions / 799,791 frames /
24,954 updates in 40m35s**. Greedy score is **13,889.583**, 1.14% below the
H5/R32/encoder6e-6 control's **14,050**; both clear the first pyramid in23/24
episodes. Sampled score rises3.06%, **13,331.250→13,738.542**, with first-pyramid
coverage23/24→24/24. Both final means still miss the original15,000 gate.
Training takes0.45% longer than the control's40m24s.

Retain encoder6e-6 for the primary H5/R32 development point and encoder2e-5
for the cheaper H15/R16 point. The latter scores13,785.417 greedy in30m16s:
this combined recipe adds only0.76% greedy score for34.04% more training time.
Sampled score improves more,9,987.500→13,738.542, but greedy was the declared
primary assessment. These reused-seed development comparisons do not establish
reliability or an isolated horizon effect. No extension, rate sweep, confirmation
or automatic successor is selected. The five-game goal remains unfinished.

All training and frozen stages completed on October30. The original CPU
postprocessor then failed on an outdated expected update count. A separate
CPU closure on November1 finishes only the missing exposure report, preserving
the failed services and all completed GPU work. Closure status is
`complete_with_cpu_audit_repair`; details and artifact hashes are below.

## Bounded execution and cost check

The fresh 12M48/Auto/N8/B16/T16/H5/R32/expert100 check completes 8,192 aggregate
actions / 32,751 emulator frames / 978 updates in **95.153s**. Its 1,024-action
frozen restore takes **.798s**, with 4,096 frames, zero updates and no natural
completed episodes. The conservative 200k projection is **42m35s**, below the
45-minute gate. This establishes execution and early cost; gameplay benefit
from combining H5/R32 with the moderate encoder rate remains unmeasured.

The [protocol](../experiments/2026-10-30-cdp-qbert-encoder-h5.md) combines two
separately useful Qbert changes. Relative to the retained H5/R32 control, only
encoder6e-6→2e-5 changes as a learner setting. Native447fe50, the Qbert teacher,
the qualified optional-rate runner and the 14,123,473-parameter architecture
remain unchanged. The runner's Breakout corrective-label plumbing is disabled
here; its exact diff against the older H5 control remains retained. Initial
matching does not prove complete cross-version learning parity.

Seed4013/evaluation4000004013 are reused development inputs. No weights,
replay, games, belief, RNG or prior check experience are reused. Relative to
the cheaper H15/R16/encoder2e-5 point, horizon, sampling, update cadence,
warmup duration and total work change together. No isolated horizon effect,
reliable combination benefit or full learner is established by this check.

All **288 declared inputs**, 357 control-matching finite initial tensors,
47 pre-update semantic events, 357 invariant frozen tensors, exact config and
cadence, finite reports, changed CNN/reward/actor/value, full training
pixel/action/corrective-label replay and both host guards pass. First update
is376, then every8 actions, with zero residual credit. All978 reports have
supervision; 234,563 labeled replay rows equal their weight sum. Maximum raw
world/actor expert-loss difference is8e-7. There are6,411 NOOP and1,781 hop
labels,3,702 passive labels and2,588 actual/teacher disagreements. Actual
4,490 overrides are54.810% of actions; guided returns are not frozen competence.

CPU preflight passes in.855s: two actual command paths,14 config cases,six
collector accounting/cost cases,all229 H5 control and235 completed moderate-rate
pins,357 schema tensors and the978/12,454/24,954 schedule. No native module,
agent or environment is constructed during preflight. No new build is needed.

The service starts **02:30:36 UTC October30**, source `8b9936a`, PID378268,
invocation `cac67bba1731491f9bb9b8518034bd1c`. It succeeds **02:32:47** after
2m11.460s; terminal collection succeeds **02:33:13**. Independent confirmation
rechecks all288 pins and five completion artifacts. No failure, GPU repetition
or host recovery occurs. Brief read-only status inspection co-hosts with the
check; no wall-time correction or isolated backend-speed claim follows.

Root `/x/Temp/kindle-qbert-encoder-h5-20261030/native-check`, trial
`qbert-12m-48-b16-h5-r32-encoder2e5-expert100-check`. All work is terminal and
no learner or GPU allocation remains active at closure. A full comparison must
be selected separately with fresh experience and fixed final/actual-initial
evaluations. No retry, cap increase, extension, rate sweep or automatic
promotion follows this bounded declaration. The five-game goal is unfinished.

A [separate fresh200k comparison](../experiments/2026-10-30-cdp-qbert-encoder-h5-200k.md)
was selected after these checks,with a45-minute cap,100k recovery and four
fixed frozen arms against the retained H5/R32 control. No check experience is
reused. Its completed result is recorded above and below.

| Artifact | SHA256 |
|---|---|
| `declaration.json` | `47e794bc8173ac21dc56fe3df4f72e53a8b0cd10eb0e4dfa6118b23b6162ee40` |
| `review.json` | `f56588b6098933d6b649e30666dcc9d6bc5f8e216c5f0a55bf04346e6eb8023d` |
| `guidance-replay.json` | `e38d17390c81502ba422f4ea41ffe3ebb71752ff3445b2584f4029025af5e605` |
| `completion-review.json` | `7b03856705eb8695b44c1e509ba4e2f61e4d9bc0b319e7e7c95d4d86bdcc75df` |

## Fresh learner launched and startup verified

The fresh full comparison starts02:43:36 UTC October30/source254c1bd,main
PID378851/invocation`f89e34b891a84b959ff05cd156fc4697`. At02:44:08 startup
verifies all325 declared pins,357 initial tensors matching control and bounded
check,47 pre-update transitions and287 finite supervised reports through2664
actions. First six updates376/384/392/400/408/416 match the declared cadence.
Progress,100k recovery,postprocessing and terminal-collection observers were
active at startup; all are now terminal. The read-only.803s startup audit
co-hosts with learning;retain the timing caveat without wall-time correction.

Declaration`893fc0ce97c0284f703c59f68bdda362da5ccd86e4b17b02c268c80a750b72d1`;
startup`c7a04ee0d9c7626ecf0b0db178dfdb2c92995f292ebdbf220ce6fa4eb12b3c61`.

## Full frozen comparison

| Assessment | H5/R32 encoder6e-6 | H5/R32 encoder2e-5 | Own actual initial |
|---|---:|---:|---:|
| Greedy mean return | 14,050.000 | 13,889.583 | 80.208 |
| Greedy first pyramids | 23/24 | 23/24 | 0/24 |
| Sampled mean return | 13,331.250 | 13,738.542 | 177.083 |
| Sampled first pyramids | 23/24 | 24/24 | 0/24 |
| Training time | 40m24s | 40m35s | — |

The [full protocol](../experiments/2026-10-30-cdp-qbert-encoder-h5-200k.md)
changes only encoder6e-6→2e-5 as a learner setting against the primary control.
Keep12M48/Auto/N8/B16/T16/H5/R32/expert100,14,123,473 parameters,world imitation
gradients,20% episode guidance with1920-action cap and all-state corrective
labels. Native447fe50 and the Qbert teacher remain fixed; the qualified
optional-rate runner differs from the historical control, as documented above.
There is no centroid or open-loop auxiliary loss. Seed4013 and its evaluation
cohort are reused; no prior learner or bounded-check experience is reused.

The four fixed final/actual-initial sampled/greedy assessments total128,112
actions/512,269 frames in77.029s, with zero updates or truncations. Each selects
the first three natural episodes per eight lanes. Guidance and labels are off.
All357 tensors remain invariant in every arm. Both actual-initial semantic
traces match the control:1139 sampled events and1101 greedy events. Initial
tensors also match both control and bounded check, with47 pre-update training
transitions matching before the first update.

All325 original pins,100k/12,454 recovery,24,954 finite reports,exact config and
cadence,changed CNN/reward/actor/value,full training pixel/action/label replay,
four frozen replays/videos,all five guards,curves,coverage and comparison pass.
The complete training replay verifies137 completed episodes. Ordinary frozen
logs lack live pixel hashes,so their replay cannot establish unavailable live
image equality. The existing comparison plot was visually inspected without
regenerating it.

Actual training overrides are61,264/200,000 (**30.632%**), versus37.515% for
the control. All5,984,010 labeled replay rows equal their weight sum; maximum
raw world/actor expert-loss difference is8e-7. There are149,513 NOOP and50,487
hop labels,138,736 passive labels and18,751 teacher/actual disagreements.
Late50k-action weighted expert cross entropy is.034579 versus.038059 for the
control. This lower training loss and the stronger online curve do not produce
a primary frozen gameplay gain. Changing replay samples and teacher-assisted
returns are not held-out accuracy or unaided competence.

## CPU reporting failure and separate closure

The main service succeeds **03:37:08 UTC October30**,source`254c1bd`,after
53m31.569s including frozen assessments and replay audits. Progress and recovery
services succeed. Postprocessing steps0–2 successfully create the curve,
coverage and comparison; step3 fails at03:37:40 because its copied exposure
helper requires12,477 strict reports instead of the declared24,954. The original
collection observer fails at03:37:42 after seeing that failed service.

The separate repair uses the already accepted H5 exposure helper,byte-identical
except for that12,477→24,954 assertion. Eight budget fixtures check the original
and corrected assertion. A first preparation attempt matched two AST assertions
instead of one and is preserved in`preparation-failure.json`; it performed no
analysis or GPU work. The corrected extraction selects the exact budget check.

The CPU repair completes **01:47:03 UTC November1** in2.020s,including.662s for
the missing exposure analysis. All325 original and43 repair pins remain
unchanged. Existing plots,comparison,coverage,recovery,semantic traces,frozen
replays and video hashes are verified without repeating their stages. There
are zero runtime import attempts and zero GPU,replay or plot repeats. Original
postprocess and collection services remain failed; no original passing
`postprocess-review.json` or`completion-review.json` is created. The separate
repair receipts explicitly record`complete_with_cpu_audit_repair`.

Full root:`/x/Temp/kindle-qbert-encoder-h5-20261030/full`,trial
`qbert-12m-48-b16-h5-r32-encoder2e5-expert100-200k-seed4013`.
Repair root:`/x/Temp/kindle-qbert-encoder-h5-20261030/cpu-closure-20261101`.
All five original services are terminal; no GPU allocation is active at closure.
Do not restart the learner,postprocessor or original collector. Preserve the
startup CPU co-host caveat without correcting the measured training time.

| Artifact | SHA256 |
|---|---|
| Full `learning-review.json` | `a7ec15eeb06ad3ee2cce5c705cdac94c4d55e25323dc1f153ce8041fbf56965f` |
| Full `guidance-replay.json` | `9cfd6886b7a04c97098133f5e4fb5913dc50c9c175fe761a1937c6b071f6ce50` |
| Full `exposure-comparison/comparison.json` | `c1b347947e2caca7d9b5ff4dccedca22de90ded8bfd7038df4bffc54c343faaf` |
| Full `expert-exposure-review.json` | `a3f954c10a4d61d033eab42a7fdd4206248c2df7ac8d84108e95b230d911f169` |
| Repair `repair-declaration.json` | `382451cc9aee278a5266e2c2eabb07397b11871c79a496891216c9204699bd7d` |
| Repair `postprocess-review.json` | `24941b7c8ae022a757134ed8b19880933ff32d100d46e00fa449bfc2f101780f` |
| Repair `completion-review.json` | `bb1dae01070f51dfbef18cb9a5efbd47e22c82d7a9401a365a919a8937bfef88` |
