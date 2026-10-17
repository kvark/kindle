# Correct the full-model encoding exactness requirement

The original [full-model smoke](2026-10-16-resident-target-model.md) remains failed:
the first CDP slow-value target comparison rejects5,219 coefficients, maximum
absolute difference5.960464477539063e-8. All four GPU processes and checkpoint
roundtrips completed; the CPU audit stopped at that exactness assertion. Keep
its source, declaration, logs, outputs and failure. No GPU retry is authorized.

The smoke declaration incorrectly required bitwise CPU/GPU slow-value encoding.
The already-qualified component contract allowed1e-5+2e-5 relative for **all**
dense two-hot targets, and already observed small encoding-rounding differences
given identical scalar inputs. Slow targets use that same encoder; they are not
copied arrays. Exactness remains appropriate for posterior/features, replay
masks, control reproduction and an arm's own checkpoint restoration.

Review the entire retained RGB/CDP cohort again with that original component
envelope applied consistently to both ordinary and slow two-hot targets. Record
the number and ULP size of every original exactness failure; do not select only
the first coefficient or discard any array. Keep all other full-model checks,
including independent complete program/pipeline comparisons, original first-
metric gates, finite later states and exact checkpoint restoration.

This explicitly revises the smoke's exactness requirement; it does not make
the original smoke pass, change numerical tolerances, alter the implementation
or establish learning parity. One CPU,2GiB,no swap,720seconds, zero new GPU work.
Any timing after this review needs a separate declaration identifying the
revised contract. Production and game training remain stopped.

The complete CPU audit finishes22:40:53UTC. RGB slow targets are exact; CDP
imagined/replay slow targets differ in5,219/310 coefficients, at most2/1 ULPs.
The earlier provisional RGB label and one-ULP characterization were incorrect.
All first targets/metrics pass the original envelopes, complete programs and
controls match, and both arms restore all292 RGB/346 CDP tensors and normalizer/
counters exactly. Fifteen CPU regression checks pass. Later trajectories differ;
there is no bitwise or learning-parity claim and no GPU repeat.
