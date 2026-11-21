# Frozen residual actor serving ablation

The [completed residual learner](../results/2026-11-06-cdp-residual-actor.md#full-200k-comparison)
scores117.625 greedy/49.333 sampled, below retained143.667/70.750. Its base
actor receives imitation while the correction receives imagined policy/entropy
gradients. The combined policy's regression does not reveal whether its learned
correction helps its own base policy at serving time. Resolve that narrower
question before selecting another learner or actor objective.

Create one separately identified copy of the final200k residual checkpoint.
Set only `behavior.actor.residual.weight` and `.bias` to exact zero. Preserve
the other361 tensors, including every optimizer moment, and all metadata except
the behavior tensor-file hash. The source stays byte-identical. Keep
`actor_residual=true`, model geometry, counters and RNG-seeding rules; this is
a frozen parameter intervention, not a trained base-only control or warm start.
Never train or resume from this diagnostic copy.

Use the exact qualified package7e0bb827/native SHA256
`3003151fd8134e636f156d82221c4455dcdbc2176d7d138e892bd5dee5bb45e9`,
runner and environment of the completed residual comparison. No rebuild,
backend change, model/optimizer update or new initial assessment is selected.
The source initial correction was already zero and its natural-episode traces
match the retained initial controls. Reuse those receipts; do not repeat them.

Run exactly two serial frozen arms, sampled then greedy, each first3 natural
episodes per8 lanes at evaluation seed4000001009. Keep full18 actions,
sticky.25, repeat4, no reset no-ops, native pooled RGB and one GPU64 resize.
Guidance, labels and learning are off. Keep600k aggregate-action,100k-frame-per-
episode and300-second wall caps per arm. No cap increase, episode replacement,
retry or automatic successor. Preserve any failure without relabeling it.

Before GPU work, verify the source's complete/terminal receipts, original input
pins, derived tensor intervention, metadata hashes and both exact commands.
The commands must differ from their completed source arms only in output,
restore and export paths. Existing native/CLI qualification is reused. Check
live host/process identity and use the same host guard; stop on real faults
or nonfinite values, with no automatic reset/reload/reboot.

Require363 finite tensors in the derived checkpoint and363 invariant tensors
after each arm, zero updates/truncations,24 natural episodes, full action/
outcome/frame/boundary replays and videos, and passing guards. The original
source checkpoint and pinned inputs must stay unchanged. Record aggregate
actions, emulator frames, native construction and evaluation wall times.
Frozen logs lack live image hashes; replay does not prove unrecorded image
equality. CPU replay has a500-second cap per arm; the whole controller has an
1,800-second cap with terminal exit/identity recorded by its wrapper.

Greedy return difference, zero-correction minus original combined policy, is
primary. Report eight paired lane means and10,000-resample percentile95%
intervals with seed0; report sampled results secondarily. A positive lower
bound supports a serving-time cost of this correction on this reused cohort.
An interval spanning zero is inconclusive; a negative interval supports its
serving-time benefit. Compare with the retained control descriptively. Do not
turn this into a fresh-training, seed-reliability or causal explanation of the
full learner's regression. Original mean864/90%-two-wall remains the game gate.

Root `/x/Temp/kindle-residual-serving-ablation-20261106`. One controller and its
original wrapper own this allocation; verify their actual PIDs before acting
on observation timeouts. No completed learner, frozen arm, collector or old
diagnostic repeats. The five-game goal remains unfinished.
