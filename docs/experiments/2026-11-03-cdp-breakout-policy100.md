# Bounded Breakout imagined-policy weight100 check

Select one fresh bounded check of `loss_scales.policy`1→100. The current
replay-value1 control scores70.750 sampled/143.667 greedy, all0/24 two-wall,
in21m27.910s. Its native high-score training exposure is sparse, and sampled
critic calibration is weak. Imagined-value.3 and actor/critic representation
gradients both lower frozen scores. Retain their original settings.

Native447fe50 already weights the imagined policy objective separately from
expert imitation. This experiment gives the imagined-policy term more weight
relative to expert100 while keeping the world imitation path at100. It scales
both the score-function term and its entropy regularizer together. It does not
separate their effects, turn off imitation, improve the critic by definition or
establish that current actor gradients are too small. Earlier raw-gradient
attribution measured the world graph of an older checkpoint, not current actor
updates. Earlier imitation decay changed both actor and world pressure and
regressed; this is a different, explicitly selected comparison.

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/lambda.95/
discount333/base-encoder4e-5/behavior8e-5/dynamics4e-4, imagined-value1,
replay-value1, expert100/unit action weights/world gradients/all-state labels,
actor_critic_gradient=false, sampled collection/posteriors, actor-unimix.01,
actor-entropy.0003, action-effects1, replay100k, warmup1000,20% episode guidance/
cap1920, seed1009/eval4000001009, full18/sticky.25/repeat4/noops0/native pooled
RGB/one GPU64 resize. Raw rewards, actual terminal boundaries and teacher stay
fixed. The effective entropy multiplier rises with the policy weight.

Use qualified native447fe50/SHA35ad0db and the corrective runtime unchanged.
Add only a fresh-CDP `--policy-loss-scale` option to retained runnercf4ea238.
Omission preserves the original config. Reject nonpositive/nonfinite/F32-invalid
values, evaluation overrides, restore overrides and non-CDP use before native
construction. Test both actual command boundaries and exact resulting configs
with a separate namespace proxy so lazy exports cannot replace constructor stubs.
CPU defaults may load the native extension; no native agent is constructed.

Require8,192 actions/489 updates, first update376 and residual credit.5 within
180s. Save actual initial/final weights; evaluate1,024 frozen greedy actions
within60s with no guidance, labels or updates. Require357 control-matching initial
and invariant frozen tensors,47 matching pre-update semantic events, exact config
except policy weight, finite reports and changed CNN/reward/actor/value tensors.
For all489 reports, require behavior total loss to reconstruct as
`100*policy_loss + value_loss + replay_value_loss + 100*expert_loss`, within
`2e-4*(1+abs(total))`. This verifies the reported weighting, not derivative or
optimizer correctness. Preserve16,334,353 parameters and existing host guards.
Replay all training pixels, actions, corrective labels, rewards and boundaries.

Qualify source identity, CLI, exact config, cadence, schema and loss arithmetic
on CPU before committing selection and launching on an idle host. Caps:
180s audit/480s main/600s original collector. No overlapping GPU work, retry,
extension, cap increase or repeated successful stage. Independently inspect
terminal identity and artifacts once after the original collector completes.

The conservative200k projection is training_seconds*200000/8192*1.10; require
at most1800s before separately selecting any full learner. A full comparison
would use fresh200k/12,477 updates,100k recovery and fixed final/actual-initial
sampled/greedy24-natural-episode assessments, with the unchanged90% two-wall gate.
No bounded experience transfers. No full learner, policy-weight sweep,
confirmation or automatic successor is selected by this bounded protocol.

Root `/x/Temp/kindle-breakout-policy100-20261103`; runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No native build or backend
optimization. Reused development seed/cohort and the coupled policy/entropy
scaling limit mechanism and reliability claims. The five-game goal stays open.

## CPU qualification

The private runner adds12 lines and deletes none from retainedcf4ea238; SHA
`5a86e224fa88f579af8bf2dcf1d0f71bd01bc9a26a6196a302b7b1f2567769de`.
Ten invalid-command cases, six constructor boundaries, two unchanged legacy
modes and two coefficient cases pass in.105s. The exact native447fe50 behavior
source confirms separate policy/expert scales; no native code changes.

Two arithmetic-fixture failures remain preserved. The first AST selector also
matched the nested initial-trace reader; its correction selects the branch that
contains behavior arithmetic. The second exposed `infinity <= infinity` in an
isolated relative-error check. The full audit already rejects nonfinite reports;
the arithmetic assertion now also explicitly requires finite error. The final
12 signed/wrong-scale/nonfinite cases pass in.018s. No GPU construction or
successful stage repeats, and the passing CLI suite is retained without repetition.

Bounded preflight passes **09:28:47.285 UTC November3**,1.034s body/1.066s wrapper:
two actual commands,20 exact-config cases, first376/489/6227/12477 cadence,357
tensor shapes and338 retained control pins. No native module, agent or emulator
is constructed during this preflight. The exact preparation failures, sources
and final provenance remain under the private root.

The [bounded result](../results/2026-11-03-cdp-breakout-policy100.md) passes once,
with no bounded stage repeated. Its private protocol snapshot preserves the
pre-result declaration. The separately specified fresh200k comparison is now
complete and lowers both frozen scores; retain policy1. See the result above.
