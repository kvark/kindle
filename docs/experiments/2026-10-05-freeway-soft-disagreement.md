# Freeway: remove categorical target noise from disagreement

The [completed diagnosis](../results/2026-10-05-freeway-disagreement-diagnosis.md)
finds weak action contrast and little coverage change. Test one factor: train
the four existing predictors on detached posterior categorical probabilities
instead of freshly sampled one-hot latents. Under squared error their expected
parameter gradient is identical; averaging out the target draw removes that
source of variance. Environment uncertainty and representation drift remain.
This does not guarantee useful novelty or upward movement. No new representation,
encoder, privileged labels, shaping, persistent action override or CPU novelty.

The input remains previous sampled RSSM feature plus actual action. Reset masks,
coefficient1, raw disagreement computation, dynamics rate4e-4, optimizer, policy
and return normalizer are unchanged. Bonus computation stays on GPU. Production
matrix precision is unchanged; the preceding F32 readout is only a diagnostic.

## Qualification

CPU graph dependency check: the one-transition ensemble loss depends on the
arrival observation and previous stochastic state, but not the current one-hot
draw. Independently verify that the mean hard-target squared-error gradient is
the soft-target gradient. Retain existing GPU F64 bonus/raw-gradient/detachment,
repeat-versus-novel, transition-alignment and exact zero-scale equivalence tests:
four serialized120-second guarded tests. Full local CPU tests, formatting/Clippy.
One excluded2,048-action/451-update soft-target production smoke (seed9001),
then1,024 frozen actions with exact model/optimizer tensor equality. No matched
speed claim from the smoke; no extra control training is needed.

## Learning declaration

Three fresh candidates, learner seeds1009/2017/3019, each32,768 actual aggregate
actions /8,131 updates. Compare against the retained three hard-target runs and
three extrinsic-only controls at exactly that budget; do not pretend they are
fresh controls or rerun them unchanged. Extra qualification actions are excluded.

Unchanged Freeway/full18/sticky.25/repeat4/no reset no-ops, RGB64 via one GPU
resize, Size1M/N8/B8/T16/H15/R32/microbatch8/replay100000. CDP cosine500,
encoder6e-6/dynamics4e-4/base4e-5, warmup1000, AGC.3, ac_grads=false. No video
pretraining, action aid or shaped external rewards. Each run deadline30 minutes;
serialize under the host guard, require RTX5080 and>=2GiB sampled Vulkan
estimated headroom. Record standalone allocation warnings; review failures
before follow-up. No NVML polling, recovery, silent retries or budget extensions.

Audit all transitions/updates/guards/finite checkpoints and retain episodes/tails.
Measure real first-reward discovery, score-vs-actions/time, intrinsic magnitude,
advantages, entropy, ensemble loss and actual requested action frequencies.
Reuse the CPU trajectory coverage diagnostic where needed. Three learner seeds
are replicates, not streams/episodes. Report differences with seed uncertainty.

If all three candidates remain at zero, skip frozen evaluation and inspect
whether the change improved action contrast/coverage before another mechanism.
If any discovers reward, evaluate all three soft candidates plus the three
retained extrinsic controls: first three natural episodes/stream (24 total),
sampled policies, max200k actions/30min, no learning, whole stream0 videos and
exact saved tensor equality. Reward discovery alone does not complete the goal;
require repeated unassisted crossings and frozen improvement over untrained
controls before a competence claim. Any longer confirmation gets a separate
finite declaration after reviewing this screen.

Runtime pins remain Meganeura592a2f5a/Bladee349cddf. Upstream6288f88 is only
paper/docs/artifact-script changes. Raw artifacts:
`runs/freeway-soft-disagreement-20261005.NeU4KJoy`.
