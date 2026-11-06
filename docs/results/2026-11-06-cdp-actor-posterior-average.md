# Frozen actor posterior averaging: implementation and CPU checks

The [development protocol](../experiments/2026-11-06-cdp-actor-posterior-average.md)
selects eight additional categorical draws for the actor's current decision.
The option passes CPU checks and the separately guarded Tiny native fixture.
Atari compatibility, inference cost and gameplay remain unmeasured. No Atari
allocation or learner is selected; the current recipe remains retained.

Each native decision repeats the current deterministic state, samples eight
categorical states from the same posterior, evaluates the shared actor in one
batch and averages softmax probabilities. It returns log mean probabilities
(floor 1e-30) to the existing action selector. Existing action masking and uniform
mixing apply there once. The original one-sample recurrent state and its RNG
remain separate; the native fixture below verifies this on bounded synthetic
inputs under identical forced actions.
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
At CPU closure the fixture had compiled but had not executed. The separate
native result below supplies the later device evidence.

Root `/x/Temp/kindle-actor-posterior-average-20261106` retains CPU sources,
commands, logs and hashes. Packaging and the separately declared native qualification now complete below;
no unchanged closed experiment was restarted.
Retain Breakout 143.667 greedy/70.750 sampled and preset 32×16. The original
864/90%-two-wall and five-game gates remain unfinished.

## Packaged native fixture selected

The release test executable and wheel build pass on source fddf48ba, with 249
pinned source files. Native SHA256 is
`960667d10fb031ec82963f5cd644802bd650c273c64db914ec6c46f6a6265680`;
test executable is `77ab9956ba5545f1e4f770127d93d9a5447da565d334571dc4e55e3ee0ea7322`.
The extracted package passes **1,174 Python tests /10 deselected** with zero native
construction attempts, including availability of the new API. Original CPU
receipts and source pins remain unchanged. The
[separate native protocol](../experiments/2026-11-06-cdp-actor-posterior-average-native.md)
selects one guarded Tiny fixture, 120s child/300s controller. No native result,
Atari comparison, learner or speedup is established at declaration.

## Native qualification complete

The single declared fixture passes on the AMD RX7900XT/RADV with Auto, source
**de97b762**, production package **fddf48ba/960667d1**. It finishes at
**12:06:09.504 UTC November 6**: **1.19s native test**, **2.866620s controller**,
**3.020570s wrapper**. All **254 pinned files** remain unchanged. The host guard
passes, original child PID1477457 exits successfully, and no child remains.
No host recovery, GPU telemetry campaign or Atari execution occurs.

Across four synthetic decisions per two streams and eight additional draws,
the new probability means satisfy the declared absolute 2e-4 bound against
independent F64 categorical reconstruction plus ordinary native actor evaluations.
Sample-dependent probabilities are nonuniform. Fixed sampled/greedy selections
and masks match exactly. All 11 actor parameters are shared, and both native
learning entry points reject the evaluation-only option. Expected caught guard
panics remain in stderr; the fixture exits successfully.

Under identical forced commands and RGB inputs, original recurrent features,
posterior RNG and policy RNG match the ordinary agent at every decision.
Parameters, optimizer moments and learning counters stay unchanged; environment
counters advance equally. This is a bounded inference routing/value result,
not a derivative, full-history marginalization or gameplay result. No Atari
steps or learner updates occur, and no declared check fails or repeats.

The separate read-only closure takes .039580s, confirms the original child is
terminal, no Kindle service/render holder remains, and rechecks all 254 pins plus
packaged artifacts. Wheel, native runtime and test executable remain retained.
A separate release receipt removes **5,136,346,520 logical bytes** of disposable
build cache and verifies all three retained artifact hashes unchanged. The
original build receipt is not rewritten.

| Artifact | SHA256 |
|---|---|
|`native-review.json`|`b49dbe63ca1d2c08477cf5b4b6abedaaaa3f4c53d692c98425afd50148eccf50`|
|`qualification-closure.json`|`30c9c9bef02119967af497863dfd6bcd592ab8497f8527c398ac4ce4f7d464f1`|
|`cache-release.json`|`3ada88786a34ae84754e665b15129a6d5d9c10ebaba02ef940e3da1e958130b6`|

Next, separately declare a bounded 12M frozen compatibility/cost check before
selecting a gameplay comparison. No new GPU allocation, learner, sample-count
sweep or automatic successor is selected. Retain the original five-game goal
and Breakout 143.667/70.750; neither speedup nor improved competence is established.


## Bounded 12M check selected

The [separate protocol](../experiments/2026-11-06-cdp-actor-posterior-average-bounded.md)
selects two frozen 4,096-action runs from the retained 200k checkpoint: ordinary
greedy and eight-sample averaging with greedy actions. CPU preflight passes in
3.466s (3.621s execution receipt), with two actual CLI paths, four invalid
commands, 14 invalid headers, five invalid ends, two invalid cohorts and six
decision fixtures. All 338 retained pins and 357 finite tensors pass; zero native
imports or native/emulator construction. The saved ordinary prefix contains
512 semantic events. Runtime/source/checkpoint identity, frozen tensors, full
replays and ≤2× observed per-action cost must pass before any separately declared
gameplay comparison. No learner, count sweep or automatic successor is selected.
