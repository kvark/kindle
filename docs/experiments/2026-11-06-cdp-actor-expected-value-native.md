# Guarded native qualification for the enumerated one-step actor

Select one execution of each prepared native fixture after the
[development checks](2026-11-06-cdp-actor-expected-value.md) and packaged Python
suite pass. Production source is `f2b80fe596ae553bbea3cc7a3c638ea55529dc22`.
Native SHA256 is `961f79583c1e2384f5dd2daba56fc2357c1a8dc280abd6bb8aef4d35fbf8b069`;
test executable SHA256 is `e582bbce455f6e05b033ddf6a322c14ba513cfd756299e01b7708a96a93d1783`.
The build receipt pins253 source files and the wheel, native library and test
binary. Packaged Python passes1194 tests with10 constructor checks deselected
and zero native construction attempts. Original compile failure stays preserved.

Run these tests individually with `--exact --ignored --nocapture --test-threads=1`:

- `dreamer::behavior::expected_value::tests::expected_value_native_objective_and_actor_derivatives`
- `dreamer::agent::expected_value_tests::expected_value_actor_matches_branches_and_restores_optimizer_owner`

The first compares the centered mixed-policy objective and all six logit
derivatives with independent F64 arithmetic, absolute tolerance2e-5. Values and
weights have no gradients and stay unchanged. A second optimizer step with zero
weights must yield zero objective and actor derivatives. These are two synthetic
optimizer steps, not environment interactions or learner updates.

The second uses Tiny, three actions, B2/T4/H3 and nonuniform reward/value heads.
Across two learner updates, compare every one of24 branches with ordinary native
transition, head, critic and bonus inference using the captured shared draw.
Reconstruct the objective from ordinary actor probabilities and detached values.
Require absolute error <1e-3×(1+abs(reference)) for values and objective; verify
the original imagination's selected first-action branch. Snapshot inputs must
exactly equal pre-update world parameters, original start states and draws must
match, and world/behavior optimizers advance once per update. Save to the fresh
`native-checkpoint` below the experiment root, restore the flag and all parameters,
moments and optimizer counts exactly; restored replay must be empty. CPU graph
checks already establish actor-only derivative ownership for this objective.

Use the existing host guard on AMD RX7900XT/RADV with Auto, checking current boot,
kernel and amdgpu identity and no live Kindle service/render holder. Pin the
protocol, controller, queue, helpers, source and binary before execution. Cap
each child at120s and controller at300s. Stop on real faults/nonfinite results;
no reset/reload/reboot or automatic retry. Retain original failures and terminal
child identities from sealed guard results, then verify no process/service or
render holder remains. Zero Atari actions; no gameplay claim follows.

Root `/x/Temp/kindle-actor-expected-value-20261106`. Success permits considering
a separately declared8,192-action learning-cost check; it does not select that
check or a full learner. Retain ordinary inference,32×16 and143.667 greedy/70.750
sampled. The original864/90%-two-wall and five-game gates remain unfinished.
