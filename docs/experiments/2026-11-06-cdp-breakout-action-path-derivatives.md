# Frozen action-path derivative probe

The dynamics.5 learner regresses despite numerically correct actor derivatives.
Use the [completed matched contexts](../results/2026-11-06-cdp-breakout-action-derivative-contexts.md)
to distinguish local model derivatives from finite discrete plan differences and
known short rescue outcomes. This is a diagnostic within the five-game goal,
not a learner, planner, coefficient sweep or repeat of the old collectors.

Keep all 120 saved origins, both frozen 200k checkpoints and their recorded and
teacher nine-action plans. Each origin receives eight categorical draw streams,
using StdRng seeds 2026110600 + 8*origin_index + draw_index, in capture order.
Use identical uniforms for both plans, both objectives and both models. Run
eight origins per batch (64 rows), 15 batches per model, serially. Native Auto
precision, original model geometry and all weights remain unchanged. No actor
or world optimizer is constructed; only diagnostic action tensors are gradient
leaves, and their optimizer is disabled. There are no Atari actions or updates.

At each imagined arrival predict reward r[t], continuation c[t] (already including
gamma) and terminal critic V[9]. Define survival S=product(c[t]) and endpoint
return Q=sum(product(c[:t])*r[t])+S*V[9], for t=0..8. Both are maximized. The
critic includes the original intrinsic-reward training; this is not a calibrated
extrinsic return. Do not add disagreement bonuses, lambda weights, actor feedback,
entropy or imitation. Thus this is not the full auxiliary actor objective.

Actions are one-hot vectors, differentiated directly at each fixed plan. Latent
draws are hard Gumbel forward with mixed-softmax straight-through derivatives,
using the implementation's existing biased surrogate. For each objective save
all action derivatives and their dot product with teacher-minus-recorded action
vectors, at both plan endpoints. Save finite plan differences separately. A
positive derivative predicts improvement locally; a positive finite difference
ranks the entire teacher plan higher. Neither is an exact discrete-action
derivative or a guarantee about an actor optimizer step.

Qualify before production: CPU graph checks allow gradients only for diagnostic
action tensors, and an independent F64 recurrent fixture checks both objectives
and all nine action derivatives against the declared anchored categorical
surrogate. Check frozen weights and disabled optimizer counters. Verify Tiny
and the first production batch's forward values against separate existing
transition/head/value inference graphs with the same draws. Check all finite
values, probabilities, frozen inputs and unchanged checkpoint files. Numerical
qualification must pass before selecting production.

Primary descriptive slices: the 18 differing game-over outcomes for survival,
and the 92 short teacher rescues for Q. Report each model's sign counts for the
eight-draw mean gradient and finite difference, their agreement, draw dispersion
and all-origin results. Use absolute tolerance 1e-6 for ties and report the raw
values; no tuned threshold, favorable-origin filtering or performance gate.
The 19 identical-plan pairs must give exact zero differences and directions.
All real nine-step rewards are zero. Intermediate life loss is not environment
termination; short rescues supply no complete alternative return. These selected
origins cannot establish population calibration or the cause of the learner's
regression. Model parameters differ, and all states follow retained-control
history rather than the candidate's own policy.

Root `/x/Temp/kindle-breakout-action-path-20261106`. Build and CPU checks allocate
no GPU. Qualification is capped at 180s; after it passes, separately declare two
serial production jobs capped at 180s each, controller 480s. Use the existing
host guard, an idle healthy device and at least 2GiB native budget headroom. Pin
binary, sources, inputs, checkpoints and protocol. Preserve every original
failure; no automatic stage repeat, cap increase, GPU reset or new learner.
