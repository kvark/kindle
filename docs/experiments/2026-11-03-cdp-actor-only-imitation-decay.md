# Qualify actor-only imitation decay while preserving world supervision

The [current actor optimizer attribution](../results/2026-11-03-cdp-breakout-actor-optimizer.md)
finds conditional imitation contributions 7,493×/10,928× the imagined policy
score-function contribution. This does not prove a gameplay cause. The earlier
[shared decay](../results/2026-11-01-cdp-breakout-expert-decay.md) lowered frozen
scores, but weakened both actor and world imitation on an older recipe. Select
a default-off actor-only option to test that separation, with numerical and
bounded execution qualification before any full learner.

Add `expert_imitation.decay.actor_only`, omitted when false. True requires world
imitation. The actor retains the existing one-based hold/linear-fade schedule;
world imitation retains the base scale. Preserve raw CE metrics, actor-update
gating, optimizer history, labels and online teacher guidance. Report the world
scale for this new mode. Default omission and existing shared-decay semantics
must remain unchanged. Restore uses saved config/counters; the CLI permits only
fresh overrides and rejects runtimes without explicit actor-only capability.

Require CPU checks for boundaries, serialization, unchanged graph/parameter
layout, invalid scope, CLI propagation/capability and per-report world scale.
Run Rust/Python CPU suites, strict Clippy and package a separate runtime. Pin
sources and executable before native work; keep retained runtimes unchanged.
The native fixture must compare actor weights100/50/0 and world100/100/100 with
F64 weighted masked CE/derivatives, including an independent auxiliary gradient.
Require unchanged raw loss and nonzero retained world gradient after actor fade.
It is synthetic qualification, not learner training or measured gameplay.

After numerical qualification, select one fresh Breakout8,192-action check under
120 seconds, followed by1,024 frozen greedy actions under60 seconds. Use retained
12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4,
policy1/critic1/1/expert100, full BPTT/microbatch16, replay100k and seed1009.
Keep the original teacher,20% episode guidance/cap1920,all-state corrective
labels,lambda.95 and the remaining retained settings. Compress actor decay to
updates128→256 so489 updates cover hold/fade/zero; world remains100 throughout.
Explicitly select centered CDP and preserve native frames/one GPU64 resize.

Require exact config/cadence,finite reports,all357 matching initial tensors and
pre-update semantics,changed CNN/reward/actor/value parameters,raw actor/world CE
alignment,full training pixel/action/label replay,host guards and unchanged357
frozen tensors with zero updates. A separate legacy frozen restore must reproduce
the retained control's fixed1,024-action prefix with no new config fields or
updates. Checkpoint layout must come from the retained actual357-tensor schema.
Do not claim source equivalence to447fe50: the new binary contains intervening
default-off changes. Require conservative training_seconds*200000/8192*1.10
<=1800 before considering a full learner.

Evidence root `/x/Temp/kindle-actor-only-decay-20261103`. Preserve failure receipts;
repair CPU processing without repeating completed native stages. Check live state
before GPU work and avoid overlapping allocations. No check experience is reused.
No full learner, sweep, unchanged extension or successor is selected here. A later
comparison would require its own declaration and final/actual-initial sampled
and greedy24-episode assessments against retained70.750/143.667,all0/24 two-wall.
The five-game goal remains unfinished.
