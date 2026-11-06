# Frozen actor posterior averaging: qualification and bounded audit failure

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


## Bounded control complete; original pipeline failed

The declared two-arm pipeline on sourced645e262 remains **failed**. Its ordinary
GPU child1478799 exits successfully after **4,096 actions /16,384 emulator frames**,
with **2.734722s acting /2.926305s construction**, zero updates and no completed
natural episode. The host guard passes with no unfinished child or recovery.
The subsequent CPU audit fails at `guard['child_pid']`: `gpu_host_guard.audit()`
verifies the sealed child identity but does not return that field. The eight-sample
candidate never starts. Original execution takes9.427889s and exits1; its failure,
source, declaration and log remain unchanged. No per-action cost ratio exists.

The [separate CPU repair](../experiments/2026-11-06-cdp-actor-posterior-average-bounded-control-repair.md)
reads the PID from the verified sealed guard result, then completes and saves the
ordinary audit and its first full recorded-action replay/video. All **357 finite
tensors**, including moments, remain identical. All **512 semantic events** match
the retained ordinary prefix exactly. Configuration differs only by the declared
explicit `replay_value_bootstrap=imagined` serialization default. Rewards,
actions, frames and boundaries replay exactly; frozen logs do not carry live image
hashes, so this does not claim unrecorded image equality or prefix competence.

That repair execution also **fails**, after saving the complete audit/replay:
its replay-only subprocess restriction intercepts the final `systemctl` read.
The5.973390s failed receipt is retained. A separate read-only closure uses the
saved review/replay and verifies their identities without repeating either one.
It completes at **12:24:32.045 UTC November6** in **2.415133s** (2.470204s wrapper).
All **616 original allocation pins**, **30 repair inputs** and **35 closure inputs**
pass. Both original failures remain failed. The original controller and GPU child
are gone; no Kindle service or render holder remains. Closure performs zero GPU
or emulator work. No production stage, replay or completed audit is repeated.

Ordinary compatibility is established for this prefix. Averaging compatibility,
its cost and gameplay benefit remain unmeasured; this repair cannot pass the
original two-arm protocol. No full evaluation, learner, count sweep or successor
is selected. Keep averaging off and retain143.667 greedy/70.750 sampled. Any new
candidate check needs a separate corrected declaration, reusing the saved control
without restarting the failed queue. The original864/90% and five-game gates
remain unfinished.

| Artifact | SHA256 |
|---|---|
|`bounded/failure.json`|`c46f87605c04bd6a1c9df902d62440b557f75c87d778761a3c09bbe7733e538f`|
|`bounded-execution-execution.json`|`83947243d6963316356afc42eeb2fef0cb768605a450dad3565d495d7d259a34`|
|`bounded/control-review.json`|`88b4b4413e2a45c02d52ec2a8105b9600488f66b9160074b41a51f17a979b288`|
|`bounded-control-repair-execution.json`|`58666587803db464d764e15edc3779f5b626b11207e7dcb7855c1fe5c28369af`|
|`bounded-control-closure/completion.json`|`37716513fa31ff7ee5d5c31772d54fdad32129905f9a4161c8ca51d185243e09`|


## Separate candidate compatibility check selected

The [new candidate-only protocol](../experiments/2026-11-06-cdp-actor-posterior-average-candidate-bounded.md)
selects the first4,096-action averaging execution, reusing the saved ordinary
control. The corrected controller reads the child PID from the audited sealed
result and leaves subprocess APIs unchanged. CPU preflight passes in3.498950s
(3.670944s wrapper), including the actual command, existing metadata/selection
rejections, seven invalid guard-result cases and the final service query.
Zero native import or native/emulator construction. The original failures remain
failed. No full evaluation or learner is selected; unchanged357 tensors, complete
replay and≤2× saved per-action cost must pass under the separate declaration.


## Separate candidate compatibility check complete

The new candidate-only declaration passes on sourceab1f5c21. It completes
**4,096 actions /16,384 emulator frames** with zero learner updates in
**2.785238s**, versus the saved ordinary2.734722s: **1.018472× observed per-action
cost**. Construction takes2.796826s versus ordinary2.926305s. These cold short
runs have different trajectories; this does not establish a general speed ratio.
No natural episode completes; partial returns are not competence scores.

All357 checkpoint tensors, including moments, remain finite and identical.
Configuration/inference/native metadata and full recorded-action replay/video
pass. All653 pins and the host guard pass; original child1479179 exits0 and no
render holder/Kindle service remains. Controller completes **12:30:43.268 UTC
November6** in14.435928s, wrapper14.583356s. No new declared stage fails or repeats;
the earlier pipeline and CPU-repair failures remain failed. This establishes
compatibility/cost for the new declaration only.

The [separately fixed full evaluation](../experiments/2026-11-06-cdp-actor-posterior-average-candidate-evaluation.md)
now selects one trained-checkpoint greedy24-natural-episode comparison, with
no learner or initial-model arm. Before execution, actual-command CPU preflight
passes in3.427866s with zero native import or native/emulator construction.
Favor averaging only at≥10% higher mean with positive paired eight-lane95%
interval, no fewer two-wall successes and≤2× saved ordinary per-action cost.
Original864/90%-two-wall and five-game gates remain unchanged. Gameplay benefit
is still unmeasured at this declaration; no automatic successor is selected.
