# Qbert: fresh 240k/R20 comparison

Selected November 2 within the unfinished five-game goal, following the
[completed bounded check](../results/2026-11-02-cdp-qbert-r20-240k.md).
That check passes8,192 actions/611 updates in50.566s, with all940 pins,
357 matched initial/invariant frozen tensors, replay and host guards passing.
Its conservative240k projection is27m09.558s, below the fixed30-minute cap.
The check is closed and will not be repeated or used as starting experience.

## Question and decision

Does20% more experience with almost unchanged learning work improve later Qbert
play? The candidate runs fresh240k/R20, versus retained200k/R24 on the same
seed16001 and original evaluation cohort4000016001. Scheduled updates are18,721
versus18,715. Experience, update cadence and replay sampling change together;
this is a budget comparison, not an isolated replay-ratio or architecture claim.

The retained control scores14,913.542 greedy and14,745.833 sampled, each24/24
first pyramids, in24m45.754s. Its original15k failure remains a failure despite
a passing alternate cohort. This reused seed is development, not independent
confirmation. The stronger14009 learner is contextual evidence, not the paired
control. No cohort or seed substitution is permitted.

Earlier300k/R24 and400k/R8 changes at older recipes regressed; their evidence
remains in the bounded protocol. The current experiment uses the subsequent
RGB64/H5/dynamics1e-4/replay-value1 recipe and a smaller experience increase.
More experience is a hypothesis, not an assumed cure.

The primary endpoint is greedy frozen performance under the original mean15,000
and90% first-pyramid gate. Report sampled results separately, along with cost,
absolute change and actual-initial-subtracted return per training minute against
the control. Report disagreement between task success and efficiency rather than
silently replacing either measure. No full-game competence follows from guided
online returns, imitation loss or execution checks.

## Fixed training and evaluation

- Fresh240,000 aggregate actions,18,721 updates, first update376, residual.625.
  Training cap1,800s; no resumed weights, replay, beliefs, games, RNG or old experience.
- 12M64/Auto/N8/B16/T16/full BPTT/microbatch16/H5/R20;16,334,353 parameters.
  Base/encoder/effective behavior4e-5, dynamics1e-4, replay-value1 with world
  gradients; greedy collection, sampled posterior, actorunimix0, replay100k.
- Expert100, unit action weights and all-state corrective labels;20% continuation
  guidance, cap1920 actions per selected episode. Measure actual override fraction.
- Full18 actions, sticky.25, repeat4, noops0, native pooled RGB and one GPU resize.
- Retain recovery checkpoints at100k/7,784 and200k/15,596 updates. Fixed240k final;
  no intermediate evaluation or checkpoint selection.
- Final and actual-initial sampled/greedy arms: first three natural episodes per
  eight lanes,24 per arm, guidance off, zero updates and zero truncations. Fixed
  evaluation seed4000016001 and lane offsets; each arm cap600k actions/300s.
- Full guided action/pixel/label replay capped1,200s; each frozen replay/video
  capped500s. Overall audit1,800s, main service5,400s, postprocessing6,000s and
  terminal observer6,300s. Observer expiry never authorizes a learner restart.

Native447fe5013b6442c477fc7bfef7eab4122082bc54/SHA35ad0db,
runner e304af89 and teacher fa261c53 remain byte-identical to the control.
Use `/x/Temp/kindle-qbert-color-masks-20261101/runtime`. No build or production
change is selected. Failed stages remain recorded; no GPU retry, cap increase,
extension, nearby-ratio sweep or automatic successor.

## Verification and retained preparation

Require exact headers/cadence,18,721 finite reports,357 initial tensors matching
both control and bounded check,47 matching pre-update transitions, changed CNN/
reward/actor/critic tensors and357 invariant tensors in every frozen arm.
Actual-initial sampled and greedy semantic traces must match the control.
Require both recovery copies, all source/package pins, full replays/videos,
guards, curves, exposure, coverage, comparison and original terminal identities.

CPU qualification passes in9.984s (10.100s wrapper): five actual commands,
38 config checks, six actual preflight cases,20 header cases,12 recovery-body
cases, three two-milestone integration cases and six retention fixtures. All940
bounded and912 control pins remain unchanged. No native agent or emulator is
constructed.

Prelaunch source review catches three stale200k/18,715 exposure assertions and
episode bins ending200k. The initial correction fixture chooses the wrong syntax
node and fails before modifying any source. That failure and original analyzer
remain retained. A separate CPU correction passes five actual assertion/bin
fixtures, covers all240k actions, and retains the partial225k–240k final bin
under nominal250k. Original full qualification is preserved; the corrected
selection receipt pins both it and the reporting correction. No GPU stage has
started during preparation.

Root `/x/Temp/kindle-qbert-r20-240k-20261102/full`;
trial `qbert-12m-64-b16-h5-r20-dynamics1e4-replayvalue1-expert100-240k-seed16001`;
main `kindle-qbert-r20-240k-20261102`, with matching postprocess, progress,
recovery and collection observers. Check live state before launch and never
overlap another GPU allocation.

Launched04:15:04.189 UTC November2/source55ead06, declarationSHA
7a9d5272df6b9a8ec4ce45d5ca35002b214ca6aba8bfcb53a1d133521e155627,991 pins.
MainPID893328/invocation575d8e40b325408e8d4220bfd3b61bb8; postprocess893331,
progress893334,recovery893472 andcollection893475 areactive. Startup passes
04:15:34.903:357 initial tensors match control/check,47 pre-update transitions
match,correct first six updates and294 finite reports through4128 actions.
Startup CPU wrapper3.032s co-host time isretained without timing correction.
Frozen gameplay andterminal completion remain pending. Do not restart oroverlap.

The separate CPU terminal reader qualifies04:21:12 UTC in5.164s (5.220s wrapper).
It reproduces the retained16001 scores/cost/recovery, checks an exact190k–240k
last50k fixture andfour original-journal cases, andverifies991 active pins and
unchanged original learner identity. ScriptSHAef55a64b47d71697d489023e6331e48fe8bff390c88f1cd8c6b2b02fa30638f1;
`/x/Temp/kindle-qbert-r20-240k-20261102/review_full_terminal.py` has separate
`recovery 100000`, `recovery 200000` and`terminal` modes. Invokeeach once after
its original evidence exists. No GPU work or learning occurs inqualification;
co-host cost isretained without correction. Frozen results remain pending.

The original100k recovery passes04:26:11.296 UTC:7,784 updates,357 finite tensors,
7,997 finite reports inits complete log prefix. Four recovery files totaling
206,067,158 bytes are retained at04:26:11.598. ReviewSHA
87f84782b1153a98814f53554ae5c64e452d06d5d6001a6640d62f86fe9b74a0;
retentionSHA3d4204857cbe3362cea82bce6a75b978bc9b7a2a2c1719350bd66e586c0d8d27;
metadataSHAd60dcb8d7c4eb9175109d69fde9105ab6a06bbb5137999538d1aef1c4c98861d.
Separate confirmation passes04:26:32.839 in2.904s (2.925s wrapper), preserving
all991 pins andoriginalPID893328/invocation575d8e40b325408e8d4220bfd3b61bb8.
Its SHA isbeb1784219eda2d039ac28190a01fef43ef67b3b36691b7cf341271108ca1143.
No GPU work,stage repeat or timing correction. Do not rerun100k confirmation;
200k recovery andfixed final frozen results remain pending.
