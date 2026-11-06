# Residual actor development and qualification

The [development protocol](../experiments/2026-11-06-cdp-residual-actor.md) selects
an optional independent reward-learning correction to the imitation actor.
`actor_residual=true` adds a zero-initialized affine recurrent-feature-to-logit
head. Imagined policy/entropy gradients train the correction; expert imitation
trains the base MLP. Acting and imagination use their summed logits. The world
expert loss still uses only the frozen base actor. Existing rates, per-parameter
clipping and LaProp remain, with separate moments for the new parameters.

At12M/full18 the correction adds46,098 trainable parameters, total16,380,451.
There are no CNN/RSSM/critic changes. The option defaults off and rejects unsupported
loss/decay combinations. Restore preserves it and old native packages reject
an enabled checkpoint before construction. This addresses an optimizer-routing
hypothesis; it does not establish that imagination supplies useful action values.

CPU qualification passes196 Kindle and5 adapter tests, strict workspace Clippy,
and17 focused CLI checks including the existing width controls. Graph traversal
checks that policy/entropy gradients reach only the correction, expert gradients
reach only the base, and world-training graphs remain identical. The actual
native derivatives, zero-logit initialization, synthetic learning, sharing and
restore fixtures are compiled but have not executed. Packaging passes on source7e0bb827: native SHA256 prefix3003151f and test
executablea0b707a9, with243 unchanged source pins. The installed package passes
1,125 Python tests/10 deselected with zero native construction attempts. Strict
binding Clippy also passes. The [separate native declaration](../experiments/2026-11-06-cdp-residual-actor-native.md)
now selects two serial guarded fixtures,120s each/controller300s. No Atari cost
check or full learner is selected.

Original preparation failures are retained at
`/x/Temp/kindle-residual-actor-20261106`: the Rust test build catches an old dynamics
fixture still passing the newly wrapped actor and a missing mutable checkpoint
borrow. CLI fixtures initially misuse the positional environment, then omit the
guidance cap and dummy action names. An explicit-protocol fixture adjustment
still fails on the same missing names. The final fixture repair passes; no native
or emulator construction occurs in those tests. No production learner stage ran.

The completed terminal-weight/replay/prior-continuation and actor-width/dynamics
experiments remain closed. This is a separate parameter-routing proposal, not a
repeat or coefficient sweep. Retain143.667 greedy/70.750 sampled, preset256 base,
coordinates and dynamics gradients off. Original864/90%-two-wall and five-game
gates remain unfinished. Numerical qualification and cost must precede a separate
learning declaration; no efficiency or gameplay improvement is claimed.
