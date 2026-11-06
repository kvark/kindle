# Enumerated one-step actor: implementation and CPU qualification

The [development protocol](../experiments/2026-11-06-cdp-actor-expected-value.md)
implements `actor_expected_value`, default false. It replaces the sampled-action
score-function term with the policy-weighted expectation of detached one-step
values over every action at each replay posterior start. Each value includes the
predicted arrival reward, current action-effects bonus and predicted discounted
continuation times current critic value. Existing entropy, imitation, critic,
imagination, return normalization and collection behavior remain.

The action branches reuse the first ordinary imagination step's latent uniforms
for each start, with one sampled prior state per action. World parameters are
private GPU snapshot inputs copied before world learning. The critic and Q
targets are detached from this actor term. The actor has one optimizer owner;
no learned parameter, optimizer slot, replay field or image transform is added.
The new option changes the policy objective's horizon and state distribution
together with its estimator. It is not an isolated test of discrete-gradient
bias or an exact expectation over latent futures.

Legacy serialization omits the false flag. Configuration and CLI guards reject
other actor experiments, device targets, fresh overrides on restore/evaluation
and enabled checkpoints on old native packages. Reports expose the unscaled
expected-value term and snapshot submission time separately. The total policy
loss already contains this term; it must not be added again as an auxiliary loss.

CPU qualification passes **207 Kindle +5 adapter tests**, **192 focused Python
tests**, a separate **20-test guard run** (15 overlap the first Python run), and
strict workspace/all-target and binding/all-target Clippy. Graph checks preserve
actor/critic parameter names and shapes, compile the combined graph and isolated
objective, verify only actor gradients from that objective, and check branch
layout at3/18 actions. Python checks cover the actual CLI configuration path,
old-runtime rejection, restore/device guards, report identity and malformed
snapshot timing. These Python runs use current sources with the previous
qualified binary, mocking the new capability and constructing no native agent.

The first Rust compile fails on an untyped index closure in the prepared native
fixture. Its original source and failed receipt are preserved. Adding the explicit
`usize` annotation repairs that test-only issue; corrected CPU execution and
Clippy pass. No GPU fixture or learner has executed at CPU closure.

Two ignored native fixtures are prepared and compile. One checks the centered
mixed-policy objective and all six actor-logit derivatives against independent
F64 arithmetic, frozen values/weights and zero masks. The other checks every
Tiny branch against ordinary transition/head/value/bonus inference, matches the
original imagination's selected first-action branch, verifies private pre-update
snapshots and shared draws, then checks two synthetic updates and exact optimizer
restore. Nonuniform reward/value weights prevent an all-zero branch check.
Compilation does not establish native values, derivatives or learning benefit.

Root `/x/Temp/kindle-actor-expected-value-20261106` retains commands, source pins,
logs and the original failed fixture. Next package and separately qualify the
native path before selecting a bounded cost check. No GPU allocation, learner,
sweep or automatic successor is selected. Retain143.667 greedy/70.750 sampled,
ordinary inference and32×16; original864/90%-two-wall and five-game gates remain
unfinished.

## Packaging and native declaration

Release test binary and wheel package source `f2b80fe596ae553bbea3cc7a3c638ea55529dc22`,
with253 source pins unchanged during both builds. Native SHA256 is
`961f79583c1e2384f5dd2daba56fc2357c1a8dc280abd6bb8aef4d35fbf8b069`;
test executable SHA256 is
`e582bbce455f6e05b033ddf6a322c14ba513cfd756299e01b7708a96a93d1783`.
Packaged Python passes1194 tests with10 constructor checks deselected and zero
native construction attempts, in4.322s including the wrapper. This verifies the
actual package capability flag and legacy defaults without GPU allocation.

The [separate native protocol](../experiments/2026-11-06-cdp-actor-expected-value-native.md)
selects the two prepared fixtures once, with the existing AMD host guard. Two
synthetic optimizer steps and two Tiny learner updates, zero Atari actions.
A bounded cost check and full learner remain unselected.

## Native qualification complete

Both exact fixtures pass on AMD RX7900XT/RADV, source `bd22618f97fde749008314030161cf893dc39a0d`.
The objective fixture takes.71s and the branch/update/restore fixture.77s; guarded
controller3.467s. All259 pins remain unchanged. Independent F64 objective and
six derivatives meet2e-5 absolute tolerance; all24 branches per update, selected
ordinary-imagination branches and reconstructed objective meet declared1e-3
relative-plus-absolute tolerance. Private pre-update snapshots and shared inputs
are exact; two learner updates and exact parameter/moment restore pass.

Children1489300/1489350 exit0; host guards, terminal checks and idle render/service
checks pass at13:02:05.115 UTC November6. Two synthetic optimizer steps, two Tiny
learner updates, zero Atari actions. The original CPU compile failure remains
preserved; no native fixture fails or repeats. This is correctness evidence,
not a gameplay or efficiency result.

The [separate bounded protocol](../experiments/2026-11-06-cdp-actor-expected-value-bounded.md)
now selects8,192 fresh learning actions and1,024 frozen actions with this package.
The one-step expectation removes the sampled-action term but preserves entropy,
imitation, critic and ordinary imagination; its extra branch cost is unmeasured.
No full learner is selected. Keep the retained recipe pending measured outcomes.
