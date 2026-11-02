# Qbert replay-value scale1 passes execution and early cost checks

After this bounded completion, a [separate fresh200k comparison](../experiments/2026-11-02-cdp-qbert-replay-value1-200k.md)
launches01:21:38 UTC November2 with the fixed30-minute cap and four frozen arms.
Startup passes;100k/9,340 recovery isretained and independently confirmed with
all859 pins and357 finite tensors. No full frozen result yet. No bounded-check
experience is reused; gameplay benefit remains unmeasured.

The bounded check completes8,192 actions /32,755 frames /733 updates in
**57.202s**. A1,024-action frozen greedy arm takes0.855s with zero updates and
no completed natural episodes. The conservative200k projection is
**25m36.196s**, below the30-minute training cap. Execution and early cost pass;
gameplay benefit is unmeasured. No full learner is selected by this completion.

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
missing loss caused Qbert's shortfall. Reusing14009 makes any following comparison
development, not independent confirmation. There are no extra parameters or
scheduled updates; actual parameter count remains16,334,353.

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
observed-episode returns. No alternative target or loss is selected while this
learner runs; the gameplay outcome remains the decision evidence.

The source-only review completes01:29:18.309 UTC November2 in.014s (.033s
wrapper), matching four declared sources to exact447fe50 and retaining the fifth
return-helper source. MainPID878107/invocation remain unchanged. No model,
environment,derivatives or accuracy measurements are involved. Evidence
`/x/Temp/kindle-qbert-replay-value1-20261102/replay-value-target-source-review.json`;
co-host duration is retained without wall-time correction. Separate readout
preparation completes01:24:50.427 in.011s (.029s wrapper), withfive actual config
assertion cases and unchanged original identity. The recovery reader subsequently passes01:35:04.128 in2.412s (2.431s wrapper),
verifying the original audit/copy without repeating either. The final reader
remains prepared and uninvoked.

## Evidence

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

## Decision and limits

The check clears the cost prerequisite for a separately declared fresh200k /
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
