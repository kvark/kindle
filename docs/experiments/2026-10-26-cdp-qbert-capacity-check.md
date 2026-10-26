# Bounded Qbert12M/RGB48/B16 check

**Selected after the completed Pong B16 comparison.** Pong uses22.37% less
training time but scores11.875 versus14.958, both24/24 wins and below mean15.
All audits/comparison pass; its service exits10:28:51 UTC. No extension or
confirmation follows. This separate Qbert check tests a different capacity/game;
it is not a full learning allocation or a claim that B16 preserves quality.

The compact Qbert4M/RGB48 tests improve over their actual initial policies but
complete no first pyramids. Increasing experience to500k at R16 scores719.792,
below200k/R32's809.375; changing the bonus or action vocabulary also fails to
improve the result. Historical12M learners completed pyramids, with different
runtime/precision and larger budgets. That supports a capacity candidate, not
a prediction that this short recipe will succeed.

Use the existing12M preset at RGB48, centered CDP, Auto, N8/B16/T16/H15/R32,
full BPTT/microbatch16, replay100k, intrinsic coefficient1 and actor_unimix0.
Fresh1009/eval4000001009, full18/sticky.25/repeat4/noops0,100000-frame episode
limit, no aid, native max-pooled frames and one GPU resize. Relative to the
completed4M/RGB48/R32 Qbert control, model_size4M→12M and batch/microbatch8→16
change together. Capacity, initialization, optimizer cadence, centering and
per-action warmup differ. There is no initial-tensor equality claim across sizes.
No native build, optimizer rescaling or backend change is required.

Declare16,384 training actions/2,002 expected updates, first376, under300s
including construction. Save own actual initial and final; ordinary frozen
restore runs1,024 actions under90s, with zero updates and all346 tensors unchanged.
Verify full configuration/source, exact counters, every numeric report, finite
checkpoints, selected CNN/reward parameter changes and both guards. Audit120s;
service660s. Selected weight changes are not a gradient-correctness proof.
The short restore does not test the natural-episode task gate or select a full
learning budget. Retain any cutoff; no retry, cap increase or automatic promotion.

A CPU checkpoint-header/source derivation predicts14,123,473 trainable parameters
at48, versus16,334,353 at64. The same transformation reconciles all346 shapes of
existing4M/RGB64 and4M/RGB48 checkpoints and their553,504-parameter difference.
It changes no checkpoint and verifies all source hashes remain unchanged. Actual
12M/RGB48 parameter count and execution must still pass the bounded check.
This parameter reduction is not a measured wall-time reduction.

Compare the matching16,384-action prefix of the completed Qbert4M/RGB48/B8/R32
control, separating construction. B8 has4,035 updates/516,480 sampled positions;
B16 expects2,002/512,512. This is an early cost comparison of different recipes,
not a full-budget speed prediction, matched optimization or independent seed.
Use its measured cost and learning evidence before choosing a subsequent budget;
the unchanged Qbert15k/90%-pyramid gate remains for a full frozen learning test.

Artifacts:`/x/Temp/kindle-cheap-20261025/qbert-12m-48-b16-check`;
controller:`run_qbert_12m_48_b16_check.py`; intended service
`kindle-cheap-qbert-12m-48-b16-check-20261026`. Native`bfb8c31`, runner`1ed07af`
and their pinned bytes remain unchanged. Complete queue normalization,
controller compilation and refusal while unselected pass without GPU work.
Prepared declaration SHA256
`2e614b0404ea0f34459857b13b67c4e04d5fa1fa3e91e21fc42937427619f3fc`;
CPU shape review`1a77c213b98848149992e04db6e86abbd8d1dddcd16acccb283aa586f6ee7be3`;
controller`ffec41110f56488832d01b52f5ae373707014d8d06e6fe0ae68eedee97d5b434`.
