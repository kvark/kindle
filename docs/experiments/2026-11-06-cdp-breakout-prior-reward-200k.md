# Fixed200k comparison of one-step prior reward supervision

The [bounded check](../results/2026-11-06-cdp-breakout-prior-reward.md) completes
8,192 actions/32,725 frames/489 updates in46.736739s. Its fixed25*1.10 padded
200k projection is1,285.260324s, below30 minutes. Initial/frozen tensors,47
pre-update events, full replay, weighted losses, mask bounds and guards pass.
The independent closure recounts489 reports and verifies783 allocation pins.
Select one fresh200k learner; bounded feasibility does not establish gameplay.

Add only `prior_reward_scale=1`: reward CE on sampled one-step prior states at
every valid replay row, using the shared reward head and aligned real reward
targets. Mask episode starts, retaining the B*T denominator. Ordinary posterior
reward scale remains1. No feature/continuation auxiliary, new target, new
parameter, actor gradient, correction or teacher change. The earlier posterior
reward10 and joint H3 failures remain closed; no causal bottleneck is asserted.

Use qualified nativea89b0b8e/SHA256
`19b6f46c93cdc331bbb1fe7a2775a379aa0335a3681676be503f3acd47358c60` and the exact
bounded runner, separately pinned before launch. The bounded runtime retains
the original Python adapters/teacher byte-for-byte with the new native library;
its separate receipt is retained. Native/runner identity changes against the
control, so initial/frozen checks do not establish broad training equivalence.
The only config changes are the new loss and explicit legacy default
`replay_value_bootstrap=imagined`.

Keep centeredCDP500/12M64/Auto/N8/B16/T16/fullBPTT/micro16/context1/H15/R16,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy1/value1/replay-value1,
reward1/continuation1,dynamicsKL1/representationKL.1/free nats1,disagreement1,
expert100/unit weights/world gradients,actor/latent mixing.01,sampled collection
and posteriors,replay100k,20% episode guidance capped1920. Other optional
interventions remain off. Preserve full18/sticky.25/repeat4/noops0,native pooled
RGB and one GPU64resize.

Train fresh seed22003 for200,000 aggregate actions/12,477 updates, first update376,
residual credit.5, within1,800s. No restored or bounded experience transfers.
Verify357 actual-initial tensors and47 pre-update events against retained22003.
Retain initial,100k/6,227 and200k/12,477 checkpoints. Independently verify100k
weights/optimizer moments on CPU while the original learner continues; restored
weights would not recover replay, live games/beliefs or full RNG state.

Frozen initial/final sampled/greedy arms each use three natural episodes per
eight lanes (24 episodes), seed4000022003, no aids/labels/updates, no selected
truncation, cap600k actions/300s per arm. Require357 invariant tensors and full
action/outcome replays. Reuse the completed primary seed22003 control at
`/x/Temp/kindle-breakout-seed22003-20261106/full/candidate`:greedy75.250,
sample44.708333 in1,290.821634s, actual-initial2.291667/1.875. Its evaluations
stay unchanged. Retained1009's94.625/70.500 on this cohort is context only.

Primary: greedy own-initial return gain per measured training minute. Favor the
candidate only if both raw greedy mean and gain/min exceed matched22003 and the
paired gain/min95% interval is strictly positive. Use10,000 bootstrap resamples
of eight paired lane means, NumPy seed0. Report sampled/raw/initial scores, both
gain/min differences and intervals, time, actions, emulator frames, aids and
failures. Reused-cohort intervals do not measure learner-seed reliability. A
promising result would need a separately declared fresh-seed confirmation; no
sweep, extension, second learner or automatic successor.

The original gate remains greedy mean864 and at least90% two-wall success.
A better short-budget score does not complete it or the full five-game goal.

Root `/mnt/data/kindle-breakout-prior-reward-20261106/full`; service
`kindle-breakout-prior-reward-200k-20261106`. Use CPU-qualified helpers and actual
commands, preserving all earlier qualification failures and separate repairs.
Pin source/native/runner/teacher/protocol/ROM and both retained control packages.
No rebuild or backend optimization. Use existing host guards and2GiB measured
GPU headroom; stop on genuine faults/nonfinite values, with no reset/retry.
Caps: training1,800s, each frozen arm300s, training replay900s, frozen replay500s,
main7,200s, collector7,500s and recovery observer2,000s. Keep full replay/label
audits, weighted world/behavior totals, mask lattice and labeled-row boundary
bounds, changed parameters, curves, exposure/coverage and original service
identity. Sampled row identities are not logged; mask-bound agreement is scoped
evidence. Independently recount frozen outcomes, comparison arithmetic and
terminal artifacts before retention/publication. Do not repeat completed stages.
