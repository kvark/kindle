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
reach only the base, and world-training graphs remain identical. The separate native results and fixture failure are recorded below. Packaging passes on source7e0bb827: native SHA256 prefix3003151f and test
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

## Original native qualification and separate fixture repair

At09:30:20.869 UTC November6, the original controller fails after two native
jobs,3.821s wrapper. The derivative fixture passes in.73s: initial combined logits
match the base exactly, all independent F64 affine/loss/derivative checks pass,
permitted gradients are nonzero, forbidden gradients are zero, and parameters/
optimizer counters remain unchanged. Its host guard passes. Do not repeat it.

The learning fixture fails in.29s after native construction, before any synthetic
transition or update. Its first observation uses `Observation::from_vec`, which
requires3136 fixed features, with3072 RGB32 values. The guard fails because the
child exits101; this is a fixture assertion, not evidence of a GPU fault. There
are zero Atari actions and no learner update in the original qualification.

The [separate repair](../experiments/2026-11-06-cdp-residual-actor-fixture-repair.md)
changes only the test observation factory to the existing RGB constructor, with
identical synthetic values. CPU coverage constructs all34 observations. A separate
test executable and new output path will qualify only missing learning/restore;
original package/runtime/production math stay unchanged. The original native
qualification remains failed even if the separate repair passes.

The repair passes three affected CPU tests, strict Clippy and release compilation.
Only `agent/vector/residual_tests.rs` differs among243 source pins from the
original package. Its separate test executable completes the missing native
learning/restore fixture at09:34:22.254 UTC in2.970s wrapper, with248 unchanged
pins and a passing host guard. Two synthetic updates change base/residual/critic;
scalar/vector/imagination synchronization and exact weights/moments restore pass.
No completed derivative fixture, emulator or learner stage is repeated.

`qualification-review.json` records `complete_with_native_fixture_repair`, while
the original native qualification remains failed. The wheel/native and both
original/repaired test executables remain retained. A separate cache-release
receipt removes disposable build files after verifying those four artifacts.

## Selected bounded check

The [fixed8,192-action protocol](../experiments/2026-11-06-cdp-residual-actor-bounded.md)
selects one fresh Breakout learner plus1,024 frozen actions, retaining all baseline
settings except the residual actor. CPU preflight passes in.217s wrapper with
zero native/emulator constructions, both exact queued commands, missing-flag
rejection and changed/extra/missing initial-tensor cases. It requires357 matching
legacy initial tensors, six zero residual additions,47 matching initial events,
363 frozen invariant tensors, complete replays and finite weighted losses.
Native3003151f/package7e0bb827 remain unchanged. No full learner is selected.

## Completed bounded check and selected full comparison

At09:40:58.358 UTC November6, source542bea7f completes the bounded controller
in66.755s (66.792s external wrapper). Training covers8,192 aggregate actions,
32,734 emulator frames and489 updates in46.534734s; retained cost is47.689684s,
a2.422% observed reduction. Construction takes2.594s. This is a short historical
comparison with different experience, not an isolated speed improvement.

All357 legacy initial tensors and47 pre-update semantic events match retained;
exactly six residual parameter/moment tensors start at zero. Initial/final/frozen
schemas contain363 tensors (world280,behavior72,slow critic11). Encoder, reward,
continuation, base/residual actor and critic parameters change. All489 reports
are finite, both weighted loss identities pass, and full training pixel/action/
label/override replay passes. The1,024-action frozen restore performs zero updates
and preserves all363 tensors; it completes zero natural episodes and supplies no
competence evidence. Both host guards and all49 allocation/243 source pins pass.
No bounded stage fails or repeats.

The padded200k projection is1,249.712s (20m49.712s), below1,800s. Minimum observed
headroom across3,097 snapshots is17,988,620,288 bytes. Subtracting the entire
5,939,200,000-byte full replay footprint leaves12,049,420,288 bytes, above2GiB;
this conservatively counts already allocated pages twice. Reported update sums
are16.587s imagination,13.628s world training,6.207s posterior,2.196s behavior
and1.232s replay; aggregate learner time is40.146s. There are3,727 overrides
(45.496%) and117,053 repeated expert-label rows, not unique experience.

The [fresh200k protocol](../experiments/2026-11-06-cdp-breakout-residual-actor-200k.md)
selects one candidate with the same package and recipe, initialized afresh.
CPU helpers qualify the complete comparison, configuration rejection, weighted
losses, exposure, plotting and terminal-message handling in3.533s wrapper with
zero native/emulator construction. Exact queued commands are checked separately
before launch. This tests whether independent reward-update parameters improve
frozen gameplay per minute; it does not establish an imitation mechanism or
repair inaccurate model/critic forecasts. Retain143.667/70.750 meanwhile.

Artifacts: `/x/Temp/kindle-residual-actor-20261106/bounded/check/completion.json`,
`review.json`, `guidance-replay.json` and guarded train/frozen evidence. Review
SHA256 `532199ca7286596878ffa7f1fdad5cdec3af07c5375bd6c5c133b7947cff6eaf`;
replay SHA256 `9f35eb4ffc25c133333a633896666a830cfa12208d346dd15839d2c86133a195`.
New full helpers/declaration live under the separate `full` directory. Original
native failure remains failed; the separate repair and closed bounded stages
are not repeated. No extension or automatic successor is selected.

The full candidate launches at09:51:14.567 UTC November6 from sourcee22b94df,
main PID1448396/invocation`de33e7330d9f474199d2512d308cd558`, with separate original
recovery and terminal observers. Exact five-command preflight passes in.381s
wrapper with351 unchanged pins, zero native/emulator construction, three
training-option rejections and both replay argv paths checked. Startup passes
at09:51:20.255: all363 bounded tensors,357 retained tensors plus six zeros,
and both47-event pre-update traces match. Training is active; no final gameplay
result exists yet. The fixed protocol and declaration stay unchanged.
