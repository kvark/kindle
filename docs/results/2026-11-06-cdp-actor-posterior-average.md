# Frozen actor posterior averaging: implementation and CPU checks

The [development protocol](../experiments/2026-11-06-cdp-actor-posterior-average.md)
selects eight additional categorical draws for the actor's current decision.
The option is implemented and CPU checks pass. Native numerical qualification,
cost and gameplay are not yet measured; no GPU allocation is selected.

Each native decision repeats the current deterministic state, samples eight
categorical states from the same posterior, evaluates the shared actor in one
batch and averages softmax probabilities. It returns log mean probabilities
(floor 1e-30) to the existing action selector. Existing action masking and uniform
mixing apply there once. The original one-sample recurrent state and its RNG
remain separate; the native fixture must still verify this on the device.
This is a conditional Monte Carlo actor average, not averaging logits or latent
vectors, planning, or a full posterior over recurrent histories.

No parameters, optimizer slots, checkpoint fields, training graph or default
inference graph are added or changed. A frozen-only pre-observation API and
`--actor-posterior-samples` accept 2–16; only eight is selected. Native learning
entry points and Python scheduled learning reject the option. CLI/audit guards
require frozen restore and reject modal posterior or temperature combinations.
Evaluation logs record the count; auxiliary per-lane RNG state is transient.

CPU checks pass **204 Kindle +5 adapter tests**, **161 Python tests**, and strict
workspace/binding Clippy. The new graph preserves the actor's parameter names
and shapes at Tiny/12M and counts 2/8/16. Independent reference arithmetic checks
mixing, fixed-draw categorical samples, tie behavior and probability averaging
versus logit averaging. Python checks exercise invalid counts, unsupported old
binaries, complete mocked frozen lifecycles and malformed log headers. They use
an isolated copy of current Python sources with the retained latent-shape binary;
new APIs are mocked, with no GPU or real emulator construction. Binding Clippy
checks the new Rust API; these Python checks do not qualify the new native binary.

All four CPU execution receipts pass without source changes or failed test
stages. A prepared ignored native fixture uses two streams and eight draws over
four synthetic decisions, a deliberately nonuniform actor, an ordinary actor
reference, independent F64 sampling/averages, masks and sampled/greedy selection.
It checks parameter sharing, frozen parameters/moments/learning counters, and
identical live features and original RNG streams under the same forced actions.
The fixture has compiled but has **not executed**. GPU success is unproven.

Root `/x/Temp/kindle-actor-posterior-average-20261106` retains CPU sources,
commands, logs and hashes. Next package the exact source and separately declare
guarded native qualification; no unchanged closed experiment is restarted.
Retain Breakout 143.667 greedy/70.750 sampled and preset 32×16. The original
864/90%-two-wall and five-game gates remain unfinished.
