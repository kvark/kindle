# CDP Freeway: first-reward exploration screen

The user authorizes focusing Phase 3 on unlocking Freeway for CDP. The stopped
five-game CDP/RGB/Tiny matrix remains deferred; its seven completed pairs and
partial eighth are retained. This is a new learning mechanism, not a backend
speed comparison or an unchanged continuation of those runs.

## One mechanism

Four independently initialized, action-conditioned predictors consume detached
RSSM features and the action actually taken. Each predicts the next detached
categorical stochastic state with mean squared error. Each head has two
RMSNorm/SiLU layers of the RSSM preset's MLP width (64 for Size1M), followed by
a linear128-output layer. Their loss is averaged over heads, output coordinates
and batch/time, with reset arrivals masked out. Parameters live in the world
optimizer/checkpoint but no ensemble gradient reaches the RSSM or encoder.
The heads use CDP's dynamics rate4e-4, existing warmup/LaProp/AGC, and the same
replay samples. No ensemble bootstrap mask or extra pixel encoder.

Intrinsic reward is the mean coordinate-wise population standard deviation
of the four current predictions, evaluated on `(state_t, action_t)` and attached
to arrival reward `r_{t+1}`. Numerically use `sqrt(variance + 1e-8) - 1e-4`, floored
at zero. Coefficient **1.0**, no running bonus normalizer or decay in this first
screen. Compute the same current bonus for real replay and imagined transitions.
The world reward head still predicts observed rewards, not cached novelty.
Existing scalar target construction may read back these bonuses; no new latent
feature readback or intrinsic computation enters the acting path.

This is inspired by [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html)
and its [implementation](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/expl.py),
not an exact reproduction: one combined extrinsic/intrinsic actor, four small
heads, no separate exploration policy or reward normalizer. Do not replace it
with raw CDP prediction error or scripted UP.

## Qualification before learning

- Independent scalar/F64 bonus values and masked-loss raw derivatives; detached
  state/action/target gradients. Repeated-state learning must reduce disagreement
  while a held-out state retains more uncertainty.
- Exact transition/action/reset alignment in replay and imagination, including
  chunk starts. Positive imagined advantages with zero extrinsic reward.
- Coefficient zero removes ensemble parameters/computation and matches the
  extrinsic-only control's outputs, updates and RNG usage.
- Finite full updates and checkpoint restore; frozen evaluation changes no saved
  tensors. One matched update timing measures overhead, not a learning claim.
  Excluded production qualification:2,048 actions/451 updates per arm, then
  1,024 frozen actions on the candidate; compare the final1,024-action windows.
- Current Meganeura main592a2f5 / Bladee349cddf, rechecked before implementation;
  bounded, serialized native jobs under the ordinary host guard. Record allocation
  warnings without an approval gate. Stop/review actual failures; no blind retry.

## Learning declaration

Control: fresh extrinsic-only CDP. Candidate: identical CDP plus disagreement.
Freeway/full18 actions/sticky.25/repeat4/no reset no-ops/native-detail input with
one GPU resize to RGB64. N8/Size1M/B8/T16/H15/R32/microbatch8/replay100000.
CDP cosine coefficient500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000,
AGC.3, `ac_grads=false`, external reward coefficient1. No action aid, video
pretraining, new reward shaping or shared histories between streams.

Learner seeds1009/2017/3019, **32,768 actual aggregate actions /8,135 updates per
run**, six runs total. Order: control1009, candidate1009, candidate2017,
control2017, control3019, candidate3019. Each run has a30-minute deadline;
no running job survives a failed guard/audit and no automatic resume/extension.
Audit each run before the next. Preserve every transition, episode and tail.

Measure first positive extrinsic reward, real/replayed positive-event counts,
extrinsic return versus actions/time, intrinsic magnitudes, advantages, entropy,
ensemble/world losses, update cost, total wall time and learner debt. Three
learner seeds, not eight streams, define the replicates. Intrinsic return is
never a game score. Report paired seed differences with bootstrap uncertainty.

If all three candidate seeds discover zero rewards, stop before frozen evaluation
of another all-zero screen and diagnose the mechanism/coverage. If any discovers
reward, evaluate all six frozen policies on the first three natural episodes
per stream (24 total), cap200k actions/30 minutes, sampled policy, no updates,
and retain whole stream-zero videos. Match tensor bytes before/after evaluation.
Reward discovery alone is not reliable learning or competence. A larger budget,
coefficient change or second mechanism needs an evidence-led new declaration,
not a blind extension or sweep. No second game until Freeway's result is reviewed.
