# Fixed 200k comparison of stronger reward supervision

The [bounded check](../results/2026-11-06-cdp-breakout-reward10.md) passes finite
updates, tensor/trace/replay checks and the fixed cost screen. Select one fresh
200k-action learner to test the distinct hypothesis that stronger existing
posterior reward supervision improves frozen policy learning per minute. The
bounded run establishes feasibility only. Nearby failed objectives stay closed.

Change only `loss_scales.reward`1→10 against the retained seed22003 learner.
This weights reward-prediction cross entropy and shared-world gradients; it
does not scale observed rewards, intrinsic bonuses or actor/critic objectives,
change targets, balance positive rows or add an imagined reward objective.
No measured causal reward-loss bottleneck is asserted. Clipping and adaptive
optimization prevent interpreting this as tenfold parameter movement.

Use native447fe50/SHA35ad0db and the exact reward-option runner
SHA256d84b13b3dee1525ea2f5dee374741edb081453b3b0667a28a5d3e0274b5fef2c.
Teacher and original runtime stay fixed. Keep centeredCDP500/12M64/Auto/N8/
B16/T16/full BPTT/micro16/context1/H15/R16, base/encoder4e-5, dynamics4e-4,
behavior8e-5, policy1/value1/replay-value1, continuation1, dynamicsKL1/
representationKL.1/free nats1, disagreement1, expert100/unit weights/world
gradients, actor/latent mixing.01, sampled collection/posteriors, replay100k,
20% episode guidance capped1920. All newer optional objectives remain off.
Preserve full18/sticky.25/repeat4/noops0, native max-pooledRGB and one GPU64resize.

Train from fresh seed22003 for200,000 aggregate actions/12,477 updates, first
update376 and residual credit.5, within1,800s. No restored experience or bounded
experience enters this run. Verify357 actual-initial tensors and47 pre-update
events against the retained22003 learner; equality to bounded1009 is not claimed.
Retain initial,100k/6,227 and200k/12,477 checkpoints; verify the100k weights and
optimizer moments independently on CPU while original training continues.
Checkpoints omit replay, live games/beliefs and full RNG state.

Frozen initial/final sampled/greedy arms each use three natural episodes in each
of eight environments (24 episodes), seed4000022003, no guidance/labels/learning,
no selected truncation, cap600k actions/300s per arm. Verify357 invariant tensors,
exact environment/native/runner identities and full action/outcome replays.
The primary baseline is the already completed22003 candidate at
`/x/Temp/kindle-breakout-seed22003-20261106/full/candidate`:
greedy75.250/sample44.708333 in1,290.821634s, with actual-initial2.291667/1.875.
Its saved evaluations are reused unchanged. Retained1009 on the same cohort
(94.625/70.500) is additional context, not a replacement comparator or new test.

Primary: greedy own-initial return gain per measured training minute. Favor the
candidate only if both its raw greedy mean and gain/min exceed matched22003,
and the paired gain/min95% interval is strictly positive. Use10,000 resamples
over eight paired lane means, NumPy seed0. Report sampled/raw/initial scores,
both gain/min differences and intervals, wall time, aggregate actions, emulator
frames, aids and failures. These reused-cohort intervals do not measure learner
seed reliability. Any promising recipe needs a separately declared fresh-seed
confirmation. No sweep, extension, second learner or automatic successor.

The original game gate remains greedy mean864 and at least90% two-wall success.
Neither a smaller fixed-budget gain nor a favorable old143.667 cohort point
completes it. The other four games and broader five-game objective stay intact.

Root `/mnt/data/kindle-breakout-reward10-20261106/full`, service
`kindle-breakout-reward10-200k-20261106`. Helpers reuse qualified audit/recovery/
plotting/comparison code, with exact source receipts and actual-command CPU
preflight. Preserve the original closure-stdout pin rejection and separate
semantic log resolution; all other bounded files are unchanged. Pin the fixed
protocol, helpers, runtime, ROM, source and both historical control packages.
No native build or backend optimization is selected.

Use one learner and existing host guards; stop on real faults/nonfinite results,
with no reset/reload/reboot or retry. Caps: training1,800s, each frozen arm300s,
training replay900s, frozen replay500s, main7,200s, collector7,500s and recovery
observer2,000s. Keep full replay/label audits, weighted world/behavior totals,
changed learning parameters, curves, exposure/coverage and original service
identities. Independently recount frozen outcomes and comparison arithmetic,
verify terminal success and artifact hashes before retention/publication. Do
not repeat any completed stage; retain failures when a separate CPU repair is
needed. The five-game goal remains unfinished.
