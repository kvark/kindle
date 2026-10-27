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
