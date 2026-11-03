# Actor-only imitation decay passes numerical and short-run qualification

Actor imitation can now fade while world-model imitation remains strong. The
default-off option passes F64 loss/gradient checks and a fresh Breakout check:
**8,192 actions /32,725 frames /489 updates in47.329s**. Actor weight reaches
zero for234 reports while world weight stays100 throughout. This establishes
execution and early cost, not improved gameplay.

The [protocol](../experiments/2026-11-03-cdp-actor-only-imitation-decay.md) follows
the [current actor optimizer attribution](2026-11-03-cdp-breakout-actor-optimizer.md).
Conditional imitation contributions were7,493×/10,928× the imagined policy
signal on two batches. The [earlier shared fade](2026-11-01-cdp-breakout-expert-decay.md)
weakened both actor and world imitation on an older recipe and regressed. This
option permits a different test; neither diagnostic proves a gameplay cause.

## Native change and qualification

Native source `c2d627f3ddee08bc222b546a4614776720f421bd` adds
`expert_imitation.decay.actor_only`. False is omitted and preserves the existing
shared schedule. True requires world imitation, keeps its base weight and applies
the existing one-based hold/linear-fade schedule only to the actor. Raw CE, label
counts, actor-update gating, optimizer history and teacher guidance remain.
The new mode reports its world weight separately. Restore retains saved config
and counters; the fresh-run CLI checks explicit native capability.

All **149 Kindle +5 gym Rust CPU tests**, strict workspace/binding Clippy and
**971 packaged Python checks** pass. CPU graph checks preserve the world graph
and parameter layout; schedule, serialization, invalid scope, CLI capability
and report checks pass. Python tests record zero native construction attempts.

The first packaged Python run has970 passes and one failed test expectation:
the Boolean-scale fixture expects the specific imitation error, but the existing
numeric guard rejects it earlier. The exact old fixture/log remain retained.
Commit `5c6526fc` changes that expectation only; the full packaged suite then
passes without rebuilding or changing native/production Python code. The original
build process remains failed and packaging closes as `complete_with_cpu_test_repair`.
Commit `349b53ea`, also before native checks, clarifies one compatibility detail:
the later config explicitly serializes `replay_value_bootstrap=imagined`, which
was already the old runtime's behavior. Exactly that default addition is allowed.

The guarded numerical fixture checks actor weights100/50/0 against world
weights100/100/100, including masked weighted CE and an independent quadratic
gradient. Raw CE stays unchanged; world imitation derivatives remain after actor
fade. All original F64 loss/derivative bounds pass. Six synthetic optimizer
steps take.74s within a1.525s guarded command, with zero learner updates or game
actions. All228 pins and the host guard pass at **10:50:21.863 UTC November3**.

Use the isolated runtime `/x/Temp/kindle-actor-only-decay-20261103/runtime`:
native SHA256 `0015dff7b1de1fe480c430e9bdbf17b64cb9fade8a77d28a91f16fce21b5416a`.
The wheel, test executable, build cache, failed/passing CPU logs and packaging
receipt remain retained. Existing runtimes are unchanged.

## Fresh bounded check

The check retains12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/
dynamics4e-4, policy1/critic1/1/expert100, full BPTT/microbatch16, replay100k,
lambda.95 and seed1009. Teacher,20% episode guidance/cap1920 and all-state
corrective labels stay fixed. The compressed actor schedule holds100 through
update128, reaches50 at192 and zero at256; world stays100. No experience is reused.

| Stage | Actions | Frames | Updates | Time |
| --- | ---: | ---: | ---: | ---: |
| Retained-control legacy greedy restore | 1,024 | 4,096 | 0 | .880s |
| Fresh training | 8,192 | 32,725 | 489 | 47.329s |
| Candidate frozen greedy restore | 1,024 | 4,096 | 0 | .879s |

All489 reports are finite, actor weights match exact counters and world weight
remains100. Reported behavior totals reconstruct with the scheduled actor loss.
Raw world/actor expert CE differs by at most7e-7. All357 initial tensors and47
pre-update semantic events match the retained control. CNN, reward, actor and
value parameters change during learning. Candidate frozen restore retains the
schedule/counters and all357 tensors, with zero updates.

The separate legacy restore reproduces128 recorded semantic events from the
retained control's first1,024 greedy actions. Its357 tensors remain invariant,
and the only config normalization is the explicit imagined-bootstrap default.
All three host guards pass. Full training pixel/action/corrective-label replay
passes. Actual overrides are3,798/8,192 (**46.3623%**), with4,394 passive labels
and22 natural training episodes. Both short frozen stages finish no natural
episodes; their partial rewards are not competence assessments.

The conservative200k projection is **21m11.038s**, below30 minutes. This is an
early projection, not a promised full-run duration. The new native binary and
runner differ from retained447fe50; matching initialization and legacy actions
do not establish isolated causality or equivalence of all training behavior.
Zero current actor imitation does not erase saved optimizer history or remove
online teacher action guidance.

## Closure

Service `kindle-breakout-actor-only-decay-check-20261103` launches10:51:16 UTC,
PID1015127/invocation `cc77dbd06adf48fc9a01aec6a179b5db`, source `5c6526fc`.
The original service succeeds **10:52:27 UTC November3**; terminal collection
completes **10:52:49.664**, taking1.144s. All599 pins pass, including338 retained
control pins without historical substitutions. No native/GPU stage fails or
repeats. Numerical, legacy, training, frozen, replay and terminal stages are
closed; do not rerun them to regenerate reports.

Evidence root `/x/Temp/kindle-actor-only-decay-20261103`; bounded artifacts are
under `native-check/breakout-12m-64-b16-r16-actor-only-decay-check`.

- Packaging: `f36ad00b7a025fc1094f9897d054f60c909692460aeebbefa3fe145be983ad69`.
- Numerical review: `ea9e8d383b8eeaa4171ebb975cdc70d1d81727ffdbd083e91fd81c4386e1915f`.
- Bounded review: `f9b8ca51a83d44f7022b9193aaa2f270c410e79364c96dc024642b87d6de5827`.
- Completion: `5097159bc25ed5795c42c3b6d2b2f9922e5294633188228680a1559be9976ab3`.

Retain the existing70.750/143.667 sampled/greedy control, all0/24 two-wall. No
full learner, sweep, extension or automatic successor is selected by this
qualification. Any full comparison needs its own declaration and frozen
final/actual-initial sampled/greedy24-episode assessments. No Kindle GPU
allocation remains active at closure; check live state before further work.
The five-game goal remains unfinished.
