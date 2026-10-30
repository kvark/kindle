# Moderate encoder rate on the retained 12M Breakout recipe

Select one fresh bounded execution/cost check with encoder learning rate
6e-6→2e-5. The recent [native optimizer diagnostic](../results/2026-10-30-cdp-breakout-native-optimizer.md)
observes encoder update/weight norms about 62–70 times smaller than recurrent
updates on two fixed batches. Imitation has larger raw gradient norms than CDP
on those batches. This motivates testing optimization, without establishing
that the encoder is the causal gameplay bottleneck or that raising its rate helps.
The diagnostic's complete numerical comparison remains protocol-failed; its
group movement results do not overturn the six failed scalar comparisons.

The [earlier 4M/expert10 rate comparison](../results/2026-10-26-cdp-breakout-pixel-reference.md#higher-encoder-rate-lowers-frozen-breakout-quality)
raised 6e-6→4e-5 and lowered sampled score 4.292→1.458. This is a smaller increase
in the stronger 12M/expert100 regime, with that negative evidence retained.
It is one development candidate, not a rate sweep or a general rate recommendation.

Keep centered CDP/12M64/Auto/N8/B16/T16/H15/R16, microbatch16, full18 actions,
coefficient1, actor-unimix.01, base/actor4e-5, dynamics4e-4, warmup1000,
replay100k, 20% episode guidance/cap1920, expert100/all weights1, world gradients
and all-state corrective labels. Keep native447fe50 and its packaged runtime,
the edge-reset pixel teacher and one GPU resize from native max-pooled frames.
No centroid or open-loop auxiliary loss is enabled. The previously qualified
encoder-rate runner differs from the retained control only by its optional rate
flag, validation and configuration assignment. No production code/build changes.
Seeds1009/evaluation4000001009 and the evaluation cohort are reused development
inputs. No weights, replay, games, belief, RNG or experience are reused.

Run8192 aggregate actions/489 updates under180s, first update376/residual credit.5.
Then restore for1024 greedy frozen actions under60s, zero updates, guidance off.
Require357 matched actual-initial control tensors and matching pre-update semantic
events;357 finite initial/final and invariant frozen tensors; exact config,
source and counters; finite reports; changed CNN/reward/actor/value; complete
training label/input-pixel/action replay; and both existing AMD host guards.
The bounded frozen segment is an execution check, not a natural-episode score.
Keep120s inner/150s outer replay,180s CPU audit and480s whole-service caps.

CPU preflight checks both actual CLI paths before output/environment/native
construction, the actual audit config assertion with negative cases, update
accounting, runner provenance and all132 retained control pins. The known
12M64 schema must match the retained control's actual357 initial tensors.
Pin exact sources and protocol before launch. Stop on real faults or nonfinite
reports; no driver recovery, GPU retry, cap increase or experience reuse.

The cost gate remains
`check_training_seconds * 200000 / 8192 * 1.10 <= 1800`.
Only after execution and cost pass, separately consider a fresh200k/12,477-update
comparison under30 minutes, with100k/6227 recovery and final/actual-initial
sampled and greedy24-episode assessments. Retained control scores are13.208/17.875;
both have0/24 two-wall successes. Any full comparison keeps the zero-truncation,
frozen invariance, replay and unchanged90% two-wall requirements. No full learner,
extension, confirmation, intermediate checkpoint selection or successor is
allocated by this bounded protocol. The existing five-game goal stays unfinished.

Root `/x/Temp/kindle-breakout-encoder-moderate-20261030/native-check`;
service `kindle-breakout-encoder-moderate-check-20261030`.

## CPU preparation passes

Preflight takes.269s with two actual CLI paths,13 config-assertion cases, exact
update accounting, all132 retained control pins and357 actual initial schema
tensors verified. No environment/native construction or native module load
occurs. The qualified optional-rate runner diff is retained. No preparation
failure or GPU work has occurred; the bounded check is ready for declaration.
