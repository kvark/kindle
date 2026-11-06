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
`/x/Temp/kindle-actor-dynamics-20261106`; all occurred before GPU qualification.

Packaging and native qualification now pass on source744dd793, native SHA256
`876ed0c55a9dae95b414aea650d31bfa35544f866c71236da82f937dd50c1a23`.
The installed package passes1,108 Python tests (10 deselected), with zero native
construction attempts. The packaged wheel and test executable remain retained;
a separate verified receipt releases1,008,970,169 bytes of disposable build files
without changing the original packaging record.

The two guarded native fixtures finish in3.379s at05:29:54.539 UTC November6,
with241 unchanged pins, three synthetic forward/backward steps, two Tiny-agent
updates and zero Atari actions. The independent F64 recurrent surrogate checks
six actor derivatives, frozen model/critic inputs and zero-weight masks. A
255-bin fixture checks uniform zero, localized means and all510 derivatives.
The production normalized loss differs from the existing rollout-return formula
by at most4.05e-9 in the two Tiny updates; this is a bounded numerical result,
not a general derivative-accuracy claim. Pre-update world snapshots, shared
starts/draws, single optimizer ownership and parameter/moment restore pass.

Default-off legacy qualification finishes05:31:48.196 UTC:1,024 frozen greedy
actions from each retained final/actual-initial checkpoint,2,048 total and zero
updates. Both reproduce all128 saved transition rows exactly (excluding finite
wall-clock timestamps) and preserve all357 checkpoint tensors. All39 pins and
both guards pass. Neither native fixture nor frozen arm was repeated; no GPU
fault or nonfinite result occurred.

A separate [bounded protocol](../experiments/2026-11-06-cdp-actor-dynamics-bounded.md)
selects one fresh8,192-action/489-update cost and numerics check at coefficient.5,
followed by1,024 frozen actions. It preserves the retained recipe and tests
initial tensor/trajectory equality, full action/label replay, weighted loss
accounting, memory and a30-minute padded200k projection. No full learner is
selected. Keep coordinates off and Breakout143.667 greedy/70.750 sampled;
original864/90%-two-wall and five-game gates remain unfinished.
