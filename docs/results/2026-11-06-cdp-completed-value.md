# Completed replay-value targets: implementation and qualification

The [optional implementation](../experiments/2026-11-06-cdp-completed-value.md)
is packaged; the [native qualification](../experiments/2026-11-06-cdp-completed-value-native.md)
is selected next. No Atari learner or gameplay benefit is established.

`replay_value_completed`, default false, substitutes completed observed game
returns for available replay-value labels. Incomplete rows retain the original
lambda target. The existing critic and world representation loss consume those
labels with unchanged scales, masks, two-hot encoding and slow-value targets.
No head, trainable parameter or graph operation is added. Imagined learning and
the optional completed-return actor remain separate. Active model-derived bonus
and auxiliary life boundaries are rejected to preserve the reward/horizon
meaning. Reports count valid replaced rows, including zero.

Rust CPU checks pass 220 Kindle and five gym tests, including independent forward
return sums, stored reward channels, stream/reset/timeout boundaries, ring
eviction, context sampling, valid zero/negative targets and default-off behavior.
Strict workspace and binding Clippy pass. The isolated wheel is built from
`d0e7bb79dd0d82692eee60529c241573baac31d9`, native SHA256
`3146e0dca5a008f9a3335c823572a862833b93983f6f3bf7fce5664bb9dbfc37`.

The first packaged Python run records **1,158 passed / 12 failed / 10 deselected**
and zero native construction attempts. The failures are in new test fixtures:
the config comparison reuses an output path, and synthetic reports omit the
`loss_scales` dictionary. Original test bytes and the failed execution stay
retained. A separate fixture-only repair passes the **12 affected cases** in
.125320s (14 unaffected cases deselected), with zero native construction.
Production/native bytes are unchanged; no compilation or passed test stage is
repeated. The build receipt explicitly says `complete_with_fixture_repair`.

Root `/x/Temp/kindle-completed-value-20261106` retains 261 pre-edit source files,
original build declarations/logs, historical source resolution, repaired tests,
wheel and validation executable. A separate receipt verifies release of
1,010,800,124 bytes of disposable build cache after packaging. CPU success does
not establish native label delivery, learning cost or frozen competence.

Any later cost or learning comparison must disclose bonus/ensemble removal
relative to the primary retained recipe. Completed returns also reflect the
historical actor/teacher mixture, with variance and policy-mismatch limits.
Keep primary Breakout 143.667 greedy / 70.750 sampled; original 864/90% two-wall
and the five-game goal remain unfinished. No full learner or automatic successor
is selected.
