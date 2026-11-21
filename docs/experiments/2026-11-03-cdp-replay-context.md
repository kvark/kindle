# Qualify an optional replay-state recomputation prefix

The [CPU age reconstruction](../results/2026-11-03-cdp-breakout-replay-age.md)
finds a median246-update age for unmasked contexts in the retained Breakout
learner's final50k actions. Age does not establish latent error or a gameplay
cause. Select an optional native prefix to test recurrent-state recomputation
without changing the16 learning steps,capacity or per-update target counts.
The earlier smaller-buffer and longer-sequence results remain separate.

Enable the existing `replay_context` field beyond its default1. One cached
posterior seeds the sequence; the next `replay_context-1` observations and
recorded actions recompute state with current parameters before the original
training sequence. Select context9: eight forward-only prefix steps, then16
learning steps. Respect resets throughout. Detach the resulting state at the
training boundary; prefix rows contribute no world,actor,value or auxiliary
targets. Only the16 training rows refresh replay caches. Use an independent
prefix RNG and no new trainable parameters. Context1 retains its exact path.
Causal video replay keeps its existing context1 restriction.

The prefix changes replay eligibility,fresh-sequence stride,sampling and RNG
work as well as state recomputation. With B16/T16/R16/N8,first update moves
376→440 actions; expect485 updates at8,192 and12,473 at200k, four fewer than
the control. Keep240 possible actor/critic target rows and256 training positions
per update. Do not claim an isolated staleness effect or identical optimizer
counts. Report prefix cost separately when enabled.

Require CPU checks for prefix/target boundaries,context1 compatibility,reset
masks,stream separation,eviction,refresh scope,configuration and CLI capability/
restore behavior. Preserve checkpoint tensor layout and default serialization.
Run Rust and packaged Python CPU suites and strict Clippy. Package an isolated
runtime; retain historical runtimes and failures. Before GPU work,pin sources,
protocol,executables and all retained evidence from a clean commit.

Guarded native qualification must compare the prefix's last state with the
corresponding longer posterior rollout under matched uniforms, test a reset
inside the prefix, and verify no parameter changes from prefix inference.
Check device replay prefix inputs against their expected stored rows. A legacy
frozen capture must reproduce128 retained semantic events and357 invariant
tensors with context1. This is execution qualification,not measured gameplay.

After native qualification, run one fresh Breakout8,192-action check under150s
and1,024 frozen greedy actions under60s. Use centeredCDP12M64/Auto/N8/B16/T16/
fullBPTT/micro16/H15/R16,encoder4e-5/behavior8e-5/dynamics4e-4,policy1,critic1/1,
constant actor/world expert100,all-state corrective labels,replay100k and
seed1009. Keep the original teacher,20% episode guidance/cap1920,lambda.95,
sampled collection/posteriors,full18/sticky.25/repeat4/native pooled RGB/one GPU64
resize. H3,centroid,crop and imitation decay remain off. No prior experience
transfers. Initial parameters must match all357 retained tensors; compare the
47 common pre-update transitions,not the later differing warmup interval.

Require exact config,cadence,finite reports,positive prefix timing,changed
CNN/reward/actor/value,aligned raw world/actor expert CE,full training pixel/
action/label replay,host guards and357 invariant frozen tensors with zero
updates. Normalize only the existing explicit imagined-bootstrap default when
comparing with447fe50. The native and runner differ; matching initial evidence
does not establish broad runtime equivalence or an isolated causal effect.
Conservative training_seconds*200000/8192*1.10 must be≤1800 before considering
a separately declared full learner. No cap increase or successful-stage repeat.

Root `/x/Temp/kindle-replay-context-20261103`. Check live state before every GPU
stage; no overlap,driver recovery or unchanged extension. Preserve failures and
repair CPU processing separately. No full learner,sweep,confirmation or
automatic successor is selected. The five-game goal remains unfinished.
