# Optional dynamics-gradient actor: implementation and qualification

The [declared development](../experiments/2026-11-06-cdp-actor-dynamics.md) adds
`actor_dynamics_scale`, default absent, to train the actor through imagined
RSSM transitions and decoded reward/continuation/critic predictions. The existing
score-function, entropy and imitation losses remain. Hard actions and latents
use mixed-softmax straight-through derivatives; this is a biased estimator.
Disagreement values, weights and normalization are detached. No gameplay benefit
or efficiency gain is established.

World snapshots are private graph inputs copied directly from GPU parameters
before posterior sampling and world training. They add no optimizer slots,
trainable weights or checkpoint tensors. Actor parameters and moments have one
owner; the auxiliary critic path freezes weights while differentiating inputs.
The runtime reuses original imagination starts and random draws, and reports
snapshot time and the unscaled auxiliary loss separately.

CPU checks pass **188 Kindle + 5 gym tests**, **16 CLI/report tests**, and strict
workspace and binding Clippy. The initial compile catches missing action-mask and counter
type errors in the new fixture. Subsequent graph construction catches scalar-only
scaling applied to row tensors. Gradient-ownership checks then find that the
compiler exposes a frozen scalar's zero-gradient placeholder as an optimizer
slot; using explicit snapshot inputs avoids ownership entirely. A compile check
also catches confusion between logical and physical parameter buffer views.
The CLI restore check initially runs only under an encoder override; its failing
test is preserved and the guard now applies to all restores before construction.
These failures and repaired executions remain at
`/x/Temp/kindle-actor-dynamics-20261106`; no GPU stage has run at this point.

Packaging and guarded native F64/snapshot/restore qualification
remain pending. No Atari check or full learner is allocated. Retain coordinates
off and Breakout 143.667 greedy / 70.750 sampled. The original 864/90%-two-wall
gate and five-game goal remain unfinished.
