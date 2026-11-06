# Residual actor development and qualification

The [full200k comparison](#full-200k-comparison) is now complete:117.625 greedy/
49.333 sampled in21m18.238s, below retained143.667/70.750 in21m27.910s.
It fails retention; keep the optional residual actor off. Qualification and
original fixture failures remain below, followed by the completed learning result.

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

## Full 200k comparison

The [fixed fresh protocol](../experiments/2026-11-06-cdp-breakout-residual-actor-200k.md)
completes and fails its retention criteria. The correction gives imagined
policy/entropy gradients independent parameters and moments, while imitation
trains the base. That recipe does not improve frozen gameplay per minute at
this budget. Keep residual actor off, base width256, coordinates and dynamics
gradients off. Retain the original control; no extension, confirmation, sweep
or automatic successor is selected.

| Measurement | Retained control | Residual actor |
| --- | ---: | ---: |
| Aggregate training actions |200,000|200,000|
| Training time |21m27.910s|21m18.238s|
| Greedy return,24 natural episodes |143.667|117.625|
| Sampled return,24 natural episodes |70.750|49.333|
| Actual-initial greedy return |2.208333|2.208333|
| Actual-initial sampled return |1.416667|1.416667|
| Greedy own-initial gain/min |6.590132|5.417615|
| Sampled own-initial gain/min |3.230038|2.249190|
| Two-wall successes, each final arm |0/24|0/24|

Training covers799,615 emulator frames and12,477 updates, ending10:12:37.118 UTC
November6. Construction costs2.547s. The0.751% observed timing reduction does
not offset lower returns: greedy gain/min falls17.792%, sampled30.366%.
The paired eight-lane greedy raw difference is−26.041667,95% interval
[−74.083333,27.500000]; gain/min difference−1.172517 has interval
[−3.413039,1.330529]. Three of eight greedy lane means improve. Sampled raw
difference−21.416667 has interval[−32.834375,−8.040625], and gain/min difference
−.980848 has interval[−1.514898,−.355244]; one of eight lanes improves. These
10,000-resample,seed0 intervals reuse development lanes and do not assess
training-seed reliability. The primary point estimates and positive-lower-bound
retention requirements fail; original mean864/90%-two-wall remains unfinished.

All363 initial tensors match the bounded candidate. All357 retained tensors
match exactly, with six zero residual parameter/moment additions. Both47-event
pre-update traces match. Actual-initial sampled799/greedy794 semantic events
also match retained, excluding declared wall-clock fields. Each frozen arm
selects the first3 natural episodes per8 lanes, with zero selected truncations
or learning updates and363 invariant tensors. Full action/outcome/frame/boundary
replays and videos pass; frozen logs still lack live image hashes.

The independent100k/6,227-update recovery passes in.358s with351 unchanged pins,
finite363 tensors, matching saved hashes and read-only file modes, while the
original learner identity stays active. Weights/moments are retained, not live
replay/games/beliefs/RNG. Full training pixel/action/label/override replay,
12,477 finite reports, both weighted losses, actor sharing and parameter-change
checks pass. Every residual weight/bias changes, as do the checked base actor,
CNN, reward, continuation and critic parameters. All five host guards pass.

Late base-only imitation CE is.557574 versus retained.560514. These use different
own-replay distributions and do not measure imitation of the combined residual
policy. Guidance overrides43.300% of actions under the unchanged20%-episode rule.
There are192,188 actions below216 cumulative points (96.094%),7,812 at216–431,
and none at432 or above;1,388 actions in216–431 are native. These score bands
are coverage observations, not wall events or frozen competence. Learning takes
1,134.265s of1,278.238s training; acting62.995s and environment62.266s. No CPU
observer time is subtracted.

Main pipeline and original terminal collector finish **10:16:34.650** and
**10:16:47.324 UTC November6** on declared sourcee22b94df. All351 current and338
retained pins pass, with exact original service/invocation lineage. Main,
recovery and terminal services all exit successfully. The separate read-only
publication helper passes in.026s with18 unchanged helper sources. No full-run
preparation/production stage fails, no GPU stage or learner update repeats,
and no device reset occurs. Original CPU/native fixture failures and the
separate repaired qualification remain preserved; full success does not erase
them. The packaged runtime and checkpoints stay retained after the earlier
cache release.

Artifacts remain at `/x/Temp/kindle-residual-actor-20261106/full`: declaration,
fixed protocol, queues, recovery receipt, `candidate/learning-review.json`,
replays/videos, curves/exposure/coverage, `comparison.json`, `completion.json`,
original-service journal, `terminal-summary.json` and `publication-execution.json`.
Completion SHA256 `d0244af9c40911510a3ead9105052d9753fc9af23ebb4c6e0296b7628bc12f19`.
Native/runner bytes differ from the historical control; scoped checks and equal
initial traces do not isolate every runtime effect. This single recipe does not
prove imitation interference absent, nor that all independent reward heads fail.
The five-game goal remains active and unfinished.
