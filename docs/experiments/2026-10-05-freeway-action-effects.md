# Freeway: disagreement about controllable action effects

The [visual-target screen](../results/2026-10-05-freeway-embedding-learning.md)
remains all-zero. Frozen probes find between-state bonus variation18–20 times
the within-state action variation; the policy captures less than1.5% of the
available one-step bonus improvement. Resampling the posterior categorical state
adds roughly2.8–3.0 times the action variation. This is not useful exploration.

A CPU-only, independent F64 transform of the existing frozen heads subtracts
each head's average prediction across all18 actions before measuring ensemble
disagreement. It removes any additive action-independent prediction component.
On256 recorded states/seed, UP beats DOWN in all768 states and the highest-bonus
action is UP in95.7/96.1/99.6%. These labels are analysis only, not inputs or
coefficients. An off-manifold uniform-action reference gives similar diagnostics,
but is rejected: exact finite-action averaging uses only real supported actions.
These are development observations, not new rollout or competence evidence.
Artifacts: `runs/freeway-embedding-diagnosis-20261005.IU4QRyWR`.

## One change

For head k, state s and discrete action a, use
`effect_k(s,a) = prediction_k(s,a) - mean_b prediction_k(s,b)`.
The bonus is the existing coordinate-mean population standard deviation across
heads, with the unchanged smooth square-root floor. Compute all actions in one
GPU batch, then select the actual action. No game-specific label, direction,
progress reward, action override or second encoder. Predictor training remains
the same detached next-CNN-embedding regression. No new parameters, optimizer,
coefficient, return normalization or environment-budget change. Zero coefficient
continues to omit the whole ensemble.

This is an experimental control-sensitive bonus, not a Plan2Explore reproduction
or an information-gain guarantee. It changes reward scale and meaning together.
One learned effect may still be irrelevant to progress; subtracting a background
component cannot prove semantic sufficiency. No production old/new switch or
checkpoint migration is needed; fresh learner seeds only.

## Qualification and budget

CPU tests, formatting and strict Clippy. Four120-second guarded GPU tests:
independent F64 action-centering/state-offset/permutation invariants, repeated
versus novel states after observing all actions, replay/imagination alignment,
and zero-scale equivalence. Three120-second read-only probes on the prior saved
models compare full network values with the independent F64 transform, using
the unchanged3e-6 native-F32 bound and reporting cooperative-math differences.
Production precision stays unchanged. Existing predictor-loss/gradient checks
remain covered by their unchanged implementation and CI.

Then one excluded2,048-action/451-update production smoke, seed9001, plus1,024
frozen actions with exact equality of346 saved tensors. Review memory, cost and
every failed job before training. Action expansion may increase cost; measure it
rather than calling this a speed improvement. Require>=2GiB sampled Vulkan
estimated budget headroom; no peak-physical-free claim.

If qualification passes, three fresh32,768-action/8,131-update seeds1009/2017/3019,
N8 Size1M B8 T16 H15 R32/microbatch8/replay100000. Freeway/full18/sticky.25/repeat4,
one GPU RGB64 resize, no reset no-ops, action aid, pretraining or reward shaping.
CDP cosine500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000, AGC.3,
ac_grads=false, disagreement coefficient1. Reuse the completed visual-target and
extrinsic-only controls; do not rerun unchanged controls.30-minute per-run limit,
auditing each before the next. Retain all episodes/tails, failures and extra
qualification/diagnostic work. Score/action/time curves, first reward, coverage,
policy entropy and whole-update cost determine the next decision.

All-zero: skip frozen competence evaluation, review the new action/coverage
signal before extending any budget. Any positive reward: frozen-evaluate all
three candidates plus three retained extrinsic controls, first3 natural episodes
per stream (24), cap200k actions/30min, sampled policy, zero updates, exact saved
tensor equality and full stream0 videos. Verify repeated unassisted crossings
and improvement over actual untrained controls before declaring Freeway unlocked.

Serialize guarded native jobs in bounded persistent systemd services. Record
standalone allocation warnings; API/numerical/fault/deadline failures stop for
review. No NVML polling, recovery, blind retries or automatic extensions.
Raw new work: `runs/freeway-action-effects-20261005.z9KUOhCS`.

The three-seed screen completes with2/1/1 real rewards; the positive branch
applies. Alongside the declared three candidates and three retained extrinsic
controls, evaluate three actual fresh initial-weight controls with the same
configuration/seeds and held-out environments. Each initialization is limited
to120 seconds, zero game actions and zero updates. All nine frozen evaluations
use the same24-episode cohort/cap and whole stream0 videos. Aggregate service
deadline2 hours; no training extension. Retained extrinsic checkpoints are
evaluated on the current native binary: disclose that change and require exact
source/restored/exported tensors, configuration, protocol and other identities,
plus the already-qualified zero-ensemble path. Do not falsify their original
training binary hash or relax the generic same-runtime pair verifier.
