# Larger Breakout learner within a short training budget

Select one bounded execution and cost check for the12M preset atRGB64 with
B16/microbatch16. The retained4M64/B8/R16/expert100 learner completes200k actions
in17m45s, scoring9.458 sampled and12.208 greedy, both withzero two-wall successes.
Historical Breakout capacity results favor12M over1M; the intervening4M result
remains weak. This motivates a larger-capacity test without claiming that its
cost or task benefit is known on this configuration.

Capacity4M→12M and batch/microbatch8→16 change together. At fixed replay ratio16,
the larger batch halves optimizer cadence and spreads warmup over more actions.
This is a practical recipe comparison, not an isolated capacity effect or equal
update/wall-cost comparison. Keep native max-pooled frames and one GPU resize
to64 because the small Breakout ball loses detail at lower geometry.

Keep centered CDP,Auto/N8/T16/H15/R16/coefficient1/actor-unimix.01/full18,
base/actor4e-5,encoder6e-6,dynamics4e-4,warmup1000,replay100k,20% episode guidance,
cap1920,expert100/all18 weights1/world gradients/all-state corrective labels.
Use the unchanged edge-reset teacher, native447fe50 and matching Breakout-era
runtime. Seed1009/evaluation4000001009 are reused development inputs, not fresh
independent confirmation. No existing weights, experience, replay, belief, games
or RNG are reused. The completed Qbert guidance3840 life is terminal.

## Bounded execution and cost check

Run8192 fresh aggregate actions/489 updates, first update376 and residual credit.5,
under180s. Then restore the final checkpoint for1024 greedy frozen actions under
60s, with zero updates and guidance/labels off. Audit exact source/config/counters,
finite reports, changed CNN/reward/actor/value,357 finite initial/final tensors,
357 invariant frozen tensors, full training pixel/label/action replay and host
guards. Keep120s inner/150s outer replay,180s CPU audit and480s whole-service caps.
Stop on real faults or numerical failures; no driver recovery.

The12M64 schema is derived from the audited12M48 checkpoint and unchanged native
geometry formulas: encoded width576→1024, posterior input2624→3072,42 affected
tensor shapes,357 total tensors and16,334,353 trainable parameters. The native
check must verify that schema. The4M control has different shapes; do not claim
matching initial values across architectures. Retain this fresh initial checkpoint
for comparison with any separately selected full life.

CPU preflight must exercise both actual CLI paths before directory/environment/
native construction, the actual config assertion with negative cases, source/
schema provenance and update accounting. Pin the complete protocol and inputs.
No native build or broader backend qualification is required.

Use the fixed conservative projection:

`check_training_seconds * 200000 / 8192 * 1.10 <= 1800`.

All execution checks and this30-minute cost gate must pass before separately
selecting a full learner. The projection is early cost evidence, not a promised
runtime. No automatic promotion, retry, cap increase, alternate capacity,
extension or reuse of check experience is selected.

Root `/x/Temp/kindle-breakout-expert100-capacity-20261029/native-check`;
service `kindle-breakout-expert100-capacity-check-20261029`.

## Potential full comparison, not yet allocated

Only after the bounded check passes, consider one fresh200k-action/12,477-update
life under1800s with100k/6227 recovery. Compare fixed final and own actual initial
in sampled and greedy modes, sampled primary: first3 natural episodes per8
lanes=24 per arm,600k-action/300s limits,zero updates/truncations and357 invariant
tensors. Full-run initial tensors must match this check, not the4M control.
Keep the original90%-two-wall gate and report scores separately. Retain full
source/config/counter/replay/guard checks, curves and actual guidance exposure.
No intermediate evaluation, checkpoint selection, extension, broad seed campaign
or automatic successor. The five-game goal remains unfinished.

## CPU preparation passes

Preflight passes in.524s: both actual CLI paths stop before directory/environment/
native construction; ten actual config-assertion cases and replay/update accounting
pass. First update376,8192/489,100k/6227 and200k/12,477 each retain half credit.
All69 retained Breakout and227 completed Qbert input pins verify. The geometry
receipt verifies357 expected tensors and16,334,353 trainable parameters against
its retained schema and unchanged native source. No preparation failure, native
module load or construction attempt occurs. Preflight receipt
`a50fb136685db2bc7cf586e7f070c42be7825fd686f43158d20c03c6f1551a06`.
The bounded native check still has to verify execution, actual schema and cost.
