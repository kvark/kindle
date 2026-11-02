# Qbert replay-value scale1 improves greedy score and passes the short-budget gate

The fresh200k comparison completes **200,000 actions /799,786 frames /18,715
updates in25m36.935s**. Greedy mean **16,525.000** improves14,770.833 by11.88%
and passes the fixed15k/90% gate with24/24 first pyramids. Sampled14,882.292
improves14,254.167 by4.41% but still misses15k; it also clears24/24 first pyramids.
Training takes1.79% longer. Initial-subtracted return/min improves10.03% greedy
and2.63% sampled. Retain scale1 as the stronger primary development option.

This is one reused learner/evaluation seed14009, not independent confirmation.
The five-game goal remains unfinished. No confirmation,extension,coefficient
sweep,cap increase or automatic successor is selected. All original services
are terminal success; no active Kindle allocation remains at closure. Check
live state before further GPU work, and do not repeat completed stages.

## Completed200k comparison

| Measurement | Replay-value0.3 control | Replay-value1 candidate |
|---|---:|---:|
| Greedy frozen mean |14,770.833|**16,525.000**|
| Sampled frozen mean |14,254.167|14,882.292|
| First pyramids,greedy andsampled |24/24 each|24/24 each|
| Training time |25m09.906s|25m36.935s|
| Greedy initial-subtracted return/min |581.079|639.341|
| Sampled initial-subtracted return/min |558.271|572.975|

Own actual-initial means147.917 greedy/205.208 sampled andzero first pyramids
match the control. Every score selects the firstthree natural episodes per
8 lanes,24 total. The four complete frozen arms consume149,704 actions /
598,650 frames /141.271s,zero updates andzero truncations; extra natural episodes
in raw run-end means are not substituted for the fixed cohort.

Only `loss_scales.replay_value` changes0.3→1 under the [fixed full protocol](../experiments/2026-11-02-cdp-qbert-replay-value1-200k.md).
Keep12M64/Auto/N8/B16/T16/H5/R24/dynamics1e-4/base-encoder-behavior4e-5/greedy
collection/expert100,unit action weights,world gradients,all-state corrective
labels,20% continuation-guided episodes/cap1920. Native447fe50/SHA35ad0db and
teacherfa261c53 remain fixed. Runner e304af89 adds only the ten-line fresh CLI
option to control e161bf38. Full and bounded candidate runners are identical;
no weights,replay,beliefs,game state,RNG state or bounded experience are reused.
There are no additional parameters or scheduled updates; actual size16,334,353.

The coefficient changes critic fitting and representation gradients through a
frozen value head, with existing bootstrap targets. The outcome does not isolate
which path helped, establish a missing-loss defect, or prove broad reliability.
Late50k expert CE falls.064864→.059228 on different collected states; this does
not establish the mechanism. Actual guidance is56,526 actions (28.263%) versus
59,478 (29.739%). Online last50 mean15,799.5 versus15,280.5 is aided, not frozen
competence. The comparison plot labels these online curves explicitly.

### Full evidence and closure

- All859 pins,exact commands/configuration,native/teacher identity and18,715
  finite reports pass. CNN,reward,actor andcritic parameters change.
- All357 initial tensors match control andbounded check;47 pre-update events
  match. Sampled/greedy actual-initial semantic traces match across1306/1338
  events. All357 tensors remain invariant in each frozen arm.
- Full training pixel/action/label andfrozen outcome replays/videos,host guards,
  curves,coverage,exposure andcomparison pass. Ordinary frozen logs lack live
  pixel hashes; no direct live-image equality is claimed.
- Recovery100k/9,340 passes andis retained:357 finite tensors,four files /
  206,067,159 bytes. Separate confirmation verifies original metadata,hashes,
  journal identity andall859 pins without repeating the audit orcopy.
- Original mainPID878107/invocation`9f50698bb9894737a8cc8d64c471ade3` launches
  01:21:38.183 UTC November2/source3a6655a. Main/postprocessing/collection succeed
  **01:54:44.234/01:55:18.718/01:55:48.439**. Allfive original services succeed;
  progress hasno Consumed timestamp,with its original journal identity verified.
  No GPU,replay orcollection stage fails orrepeats.
- The separate5.947s terminal reader completes01:56:36.021. It verifies original
  identities,all859 pins,completion hashes,paired initials,comparison andscores.
  The plot is visually reviewed; its curves are guided training returns, with
  frozen means andcosts shown separately.

CPU qualification passesfive exact commands,36 config cases,three actual launch
preflight cases,header/recovery/observer/retention checks and20 script compiles.
The preceding in-memory generator indentation failure andseparate repair remain;
no full-run files orGPU work occurred in that failure. Bounded fixture/anchor
failures below remain preserved. CPU wrapper co-host costs2.621s startup,.029s
reader preparation,.033s source review and2.431s recovery confirmation are
uncorrected. The original control's co-host caveats also remain; timing does not
isolate a native-kernel speed effect.

Full root `/x/Temp/kindle-qbert-replay-value1-20261102/full`, trial
`qbert-12m-64-b16-h5-r24-dynamics1e4-replayvalue1-expert100-200k-seed14009`.
Parent `full-terminal-summary.json` SHA f359e215; declaration8ba2436a;
learning review6f69472a; postprocessfe555604; completion048abefc;
comparisonfa66c107. Retain the original recovery,all checkpoints,videos,replays,
curves andreceipts. Do not rerun learner,observers,collector orfinal reader.

## What the existing replay value term already uses

The pinned native447fe50 source already trains value targets from **observed
rewards and actual episode boundaries**. With B16/T16, each update constructs240
replay targets. Rewards include the existing replay intrinsic bonus. Lambda is.95
and the continuation discount is1−1/333. Each replay posterior state also supplies
its first imagined lambda return as the bootstrap; this experiment usesH5.
This follows the [target construction](https://github.com/kvark/kindle/blob/447fe5013b6442c477fc7bfef7eab4122082bc54/kindle/src/dreamer/agent.rs#L2034)
and [return helper](https://github.com/kvark/kindle/blob/447fe5013b6442c477fc7bfef7eab4122082bc54/kindle/src/dreamer/distributions.rs#L142).

Thus real experience is already part of this loss, while the targets still depend
on imagined outcomes and value predictions. Termination removes the continuation
term; episode boundaries stop the lambda trace. The helper's separately supplied
`value` array is length-checked but unused in the calculation: its `bootstrap`
array determines the mixture and trace initialization. This is source semantics,
not evidence of a defect or of the targets' empirical accuracy.

The0.3→1 coefficient weights replay critic cross-entropy plus slow-value
regularization and the corresponding world-state loss through frozen value
logits. It does not change target construction. Increasing its weight therefore
tests a different question from replacing the bootstrap or using complete
observed-episode returns. This source review selected no alternative target or loss. The completed
gameplay comparison supplies the decision evidence.

The source-only review completes01:29:18.309 UTC November2 in.014s (.033s
wrapper), matching four declared sources to exact447fe50 and retaining the fifth
return-helper source. MainPID878107/invocation remain unchanged. No model,
environment,derivatives or accuracy measurements are involved. Evidence
`/x/Temp/kindle-qbert-replay-value1-20261102/replay-value-target-source-review.json`;
co-host duration is retained without wall-time correction. Separate readout
preparation completes01:24:50.427 in.011s (.029s wrapper), withfive actual config
assertion cases and unchanged original identity. The recovery reader subsequently passes01:35:04.128 in2.412s (2.431s wrapper),
verifying the original audit/copy without repeating either. The final reader completes01:56:36.021 in5.947s (5.968s wrapper), after all
original services finish. No completed stage is repeated.

## Bounded execution and early cost check

The bounded check completes8,192 actions /32,755 frames /733 updates in
**57.202s**. A1,024-action frozen greedy arm takes0.855s with zero updates and
no completed natural episodes. The conservative200k projection is
**25m36.196s**, below the30-minute training cap. Execution and early cost pass;
gameplay benefit was unmeasured at that point. The full comparison was selected
separately after this check closed.

Only the existing replay-value loss scale changes0.3→1.0 against completed
seed14009. The [fixed protocol](../experiments/2026-11-02-cdp-qbert-replay-value1.md)
keeps12M64/Auto/N8/B16/T16/H5/R24/dynamics1e-4/base-encoder-behavior4e-5/greedy
collection/expert100, all-state corrective labels and20% guided episodes/cap1920.
Native447fe50/SHA35ad0db and fast teacherfa261c53 remain fixed. The same ten-line
fresh-only CLI addition is applied to production and the exact private control
runner, e161bf38→e304af89; native defaults remain unchanged when omitted.

The coefficient weights both behavior critic training and representation
training through a frozen critic. Existing bootstrapped targets remain. This is
not Monte Carlo supervision, an isolated critic intervention, or proof that a
missing loss caused Qbert's shortfall. Reusing14009 makes the subsequent comparison development, not independent
confirmation. There are no extra parameters or
scheduled updates; actual parameter count remains16,334,353.

### Bounded evidence

- All815 declared pins stay unchanged, including768 control inputs. Native and
  teacher packages, exact commands and configuration match the declaration.
- All733 learner reports are finite, with the declared cadence, first update376
  and residual credit.75. CNN,reward,actor and critic parameters change.
- All357 initial tensors match the retained control exactly, with47 identical
  pre-update semantic events. All357 tensors remain invariant during evaluation.
- Full recorded action/pixel/label replay and both host guards pass. Training
  has2,663 actual overrides,32.5073% of actions; online returns are aided.
- Both original services succeed. MainPID877568/invocation
  `7245ee2b08064c578610357b76b6fe60` launches01:13:51 UTC November2/source921ad96
  and completes01:15:11.906. CollectorPID877571/invocation
  `34c5f148a2704b88ac27cc00a4ce825b` completes01:15:18.703. No GPU, replay or
  collection stage fails or repeats.
- Separate terminal readout completes01:15:58.242 in2.769s; both original journal
  identities and all815 pins verify. No active Kindle allocation remains.

CPU qualification passes27 focused tests in production and27 in the private
runner, two actual commands, ten config rejection/acceptance cases,357 shapes
and the complete cadence derivation. The first test fixture incorrectly reused
an output path; the failed receipt/source and separate corrected pass remain.
The first source-copy script stops on an anchor containing an optional centroid
flag absent from the older runner. Its ten completed controller copies remain;
a separate script assembles the runner using unique unchanged anchors. Neither
preparation failure constructs an environment or native agent. Original files
and all failures are preserved. No new native build occurs.

### Bounded decision and limits

The check cleared the cost prerequisite for the separately declared fresh200k /
18,715-update comparison against14009's greedy14,770.833 and sampled14,254.167,
both24/24 first pyramids, in25m09.906s. Keep the fixed15k/90% gate,100k/9,340
recovery, and final/actual-initial sampled and greedy24-episode assessments.
No weight, replay, belief, game state, RNG state or experience from this bounded
check may be reused. A single reused learner cannot establish reliability.

The projection comes from early training, not measured full-run cost. Short
frozen execution with no completed episodes says nothing about competence.
Sparse read-only progress inspection is uncorrected host co-work. No extension,
cap increase, coefficient sweep or automatic successor is selected. Do not
restart this bounded run, collector or terminal reader.

Evidence root `/x/Temp/kindle-qbert-replay-value1-20261102/native-check`, trial
`qbert-12m-64-b16-h5-r24-dynamics1e4-replayvalue1-expert100-check-seed14009`.
Package parent retains `bounded-terminal-summary.json` and all CPU receipts.
DeclarationSHA411d2c4c; reviewSHAc225790d; completionSHA91b0870d;
replaySHAd61c9c6a. The five-game goal remains unfinished.
