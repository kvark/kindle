# Optional expert supervision of world state

The completed continuation Qbert learners improve with expert imitation but
miss mean15000. More guidance with less replay and greedy frozen actions both
lower quality. Breakout's longer guidance also lowers frozen quality. Source
review shows that expert labels currently train the behavior actor from detached
replay features, without direct expert gradients into the CNN or recurrent
state. This motivates one optional native learning change, not a diagnosed
cause of the score gaps or a default architecture replacement.

## Selected implementation

Add `expert_imitation.world_gradient`, default false and omitted when false
in serialized configuration. The world graph evaluates the same weighted,
masked cross entropy through the current frozen actor. The behavior optimizer
continues owning actor parameters; the new path trains only world/encoder
parameters. Both updates use the same pre-update targets and actor parameters.
Keep expert scale and actor-start gating. Emit an optional world expert-loss
metric. Existing configuration/checkpoint behavior remains unchanged when off.

Reuse the existing trusted outgoing-action targets: state t receives the expert
action stored with frame t+1. Last/reset boundaries and untrusted actions stay
masked; the final sampled state has no outgoing target. Correct the world T
average to the behavior T−1 average, including chunked/microbatched training.
Expose `--guidance-world-gradient` only with fresh guided imitation, reject
restore overrides and older native extensions without explicit support.
Inference receives pixels and its own history; the new flag adds no frozen aid.

## Bounded validation before Atari allocation

CPU checks cover legacy configuration round trips, default graph equality,
target rows/masks/normalization, graph connectivity into CNN/posterior with
frozen actor/critic parameters, staged and ordinary graphs, combined optional
actor/critic gradients, CLI configuration and rejection before construction.
Run relevant Python tests, Rust CPU tests and strict Clippy. Preserve failed
development checks externally; a failing fixture is not a GPU result.

Select two guarded native GPU checks only after CPU validation:

- Isolate expert loss in a tiny RGB32/CDP world. Require finite nonzero CNN and
  posterior gradients, an actual CNN update, unchanged frozen actor parameters,
  and zero loss/gradients with all expert targets masked. Cap180 seconds.
- Run the native vector trusted-action integration with world gradients on.
  Require aligned labels, finite positive world/behavior expert losses matching
  within2e−4 from pre-update parameters, and an actor update. Cap240 seconds.

These checks establish their stated gradient/alignment scope, not task success,
general numerical parity or a causal explanation of prior Atari failures. Use
the existing RX7900XT host guard, Auto precision, one GPU job at a time, and
stop on real faults or numerical failures. No reset/reload/reboot or automatic
retry. A failed assertion needs review before any distinct corrected check.

Build/package separately under `/x/Temp/kindle-expert-world-20261027`, retain
the exact source/native/runtime hashes and original79c9a7f runtime. Release
the disposable build cache only after packaging and retaining test executables.
No Atari learner, full training allocation or automatic promotion is selected
by this implementation protocol. A bounded Atari check must follow a separate
review of these completed checks and their cost.

## Native checks pass; bounded Qbert execution selected

Source`0e157df` passes129 Rust CPU tests,466 relevant Python tests both before
and after packaging, workspace/binding strict Clippy and formatting. Initial
compile/fixture failures (missing struct field, unsupported graph debug format,
CLI fixture action meanings/override signature and one absent test path) are
retained externally; no native GPU stage fails or repeats.

Both selected GPU checks pass on the guarded RX7900XT in5.986s total service
wall time,06:25:52–06:25:58 UTC. Isolated expert loss1.099951 has CNN squared
gradient norm.0003636701 and posterior norm.0001376968; a CNN update occurs,
frozen actor weights stay unchanged, and zero targets produce exactly zero
loss/gradients. The vector check has3 labels/weight3/loss.5461493 and passes
world/behavior pre-update agreement within2e−4. These are actual isolated
native derivatives and updates, not an explanation of past Atari failures.
Review SHA256`f0c24b6ec2f5aeb9974c91de159ab4649971f8f0915f4a17aafe607887164b9b`.
Native SHA256`6a9c41c44aef4747cef6504a2525195d8f2f2a62dbece7c6ecc6d1128fff1c1e`;
the original runtime is retained separately.

After terminal review, select one fresh8192-action/1987-update Qbert check:
4M/RGB48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
seed4013/eval4000004013, continuation guidance25%/cap1920/expert1/NOOPweight.05,
with world gradients enabled. The25% episode probability is execution-check
exposure, not a selected full-run recipe. First248/residual0;180s training cap.
Follow with1024 frozen actions/60s, no guidance/updates. It cannot establish
natural-episode competence. CPU audit cap180s; whole service480s.

Require all1987 finite reports, positive weighted actor/world expert losses
when labeled (agreement within.002×(1+actor loss)), zero masked losses, trusted
hop labels, full input/action/outcome guidance replay, host guards and all357
finite checkpoint tensors. The11 additional world tensors are frozen actor
copies and must equal behavior actor weights. All346 shared initial tensors
must match the retained4013 control; all357 final/frozen tensors must match.
Check changed CNN/reward/actor/value parameters without attributing changes
in a joint update exclusively to the new term. Preserve full config, source,
seeds, frame/action/time counts and every failed attempt.

Trial:`/x/Temp/kindle-expert-world-20261027/qbert-4m-48-expert-world-check`.
Owner:`kindle-qbert-expert-world-check-20261027`. No retry, cap increase,
extension, reuse of check experience, full learner or automatic promotion.
