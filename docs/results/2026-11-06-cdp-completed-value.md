# Completed replay-value targets: implementation and qualification

The [optional implementation](../experiments/2026-11-06-cdp-completed-value.md)
is packaged and natively qualified. The [bounded activity/cost check](../experiments/2026-11-06-cdp-completed-value-bounded.md) completes 8,192 actions / 489 updates in **36.403s**, with active completed-return targets. No full learner or gameplay benefit is established.

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

## Native qualification and default-off compatibility

The guarded Tiny fixture passes in **2.766637s** at **19:32:33.930 UTC November6**,
with all277 pins unchanged. Initial parameter names/values and unchanged imagined
labels/masks/slow targets match the control. Available completed labels reach the
native behavior loss; an unavailable label retains its original target. The two
direct behavior steps produce different critic weights, and the candidate's two
full synthetic updates remain finite. Config and world/behavior parameters
restore with empty replay. No Atari environment is constructed.

Default-off final and actual-initial restores then complete1,024 actions each
in.898828s/.903966s, with zero updates. All128 transition rows per arm reproduce
the retained semantic prefixes exactly, and all357 checkpoint tensors remain
unchanged. The two guards and40 pins pass. Queue11.715211s; review.911104s,
complete19:33:09.491 UTC. All original children are terminal; no GPU failure or
stage repeat occurs. Production native remainsd0e7bb79/SHA3146e0dc.

The separate bounded protocol selects 8,192 fresh actions with the bonus off
and completed replay values on, followed by 1,024 frozen actions. Its CPU entry
preflight passes in.360673s with26 pins, both real command paths mocked, zero
native/emulator construction and unchanged tracking/predictive/feedback policy
ASTs. Native and runner versions differ from the saved bonus-off control; exact
initial weights and pre-update traces must match before a paired claim. No full
learner or gameplay improvement follows from these qualification checks.

## Completed bounded activity and cost check

The fresh candidate completes **8,192 aggregate actions / 32,729 emulator
frames / 489 updates in 36.403001s**. Construction takes 2.383180s separately.
The saved bonus-off control took 36.676835s; the new point is 0.747% lower.
This short historical comparison does not establish an isolated speed change:
experience and native/runner versions differ. No control is rerun.

Completed observed labels replace **56,061 replay-value rows across 434 of
489 updates**, first at action **1,224**, when the first true terminal is
observed. Counts describe repeated replay exposures, not unique states or
independent samples. The actor auxiliary stays off, and reported model-derived
intrinsic rewards remain exactly zero. All 489 weighted world/behavior loss
checks pass, with maximum absolute errors 9.715e-5 / 3.287e-5.

All **261 actual-initial tensors** and **47 pre-update transitions** match the
retained bonus-off control. CNN, reward, continuation, actor and critic weights
change. The frozen restore takes **.816224s** for **1,024 actions / 4,096 frames**,
with zero updates, zero completed natural episodes and all 261 tensors invariant.
Complete training/frozen replay verifies every action, reward, frame and boundary,
plus corrective labels and training guidance pixel hashes. Actual overrides are
3,698/8,192 (45.142%); guided training returns are not frozen competence.

| Summed learner component | Seconds |
|---|---:|
| Replay | 1.228 |
| Posterior | 5.496 |
| Imagination | 7.574 |
| World training | 13.156 |
| Behavior training | 2.509 |
| Total learner time, including refresh/sync | 30.184 |

The padded 200k projection is **977.620s (16m17.620s)**, below the 30-minute cap.
Across 3,097 native memory snapshots, minimum observed headroom is
18,738,585,600 bytes. Subtracting the entire 5,939,200,000-byte full replay leaves
**11.920GiB**. These are conservative short-run projections, not measured full
training cost or a gameplay result.

Execution source `20dc4d43b0ee6867865d8ba29214b52d2ff57f73`, unchanged native
`d0e7bb79` / SHA `3146e0dc`. Controller finishes **19:42:33.689 UTC November 6**
in 54.804038s. Both GPU guards, 57 allocation pins and all audits pass; no GPU
stage fails or repeats. Independent CPU recount finishes **19:42:54.153 UTC**
in .580474s, verifying 57 pins, 46 completion artifacts, nine checkpoint tensor
files, target activity and projection arithmetic. Original train/frozen children
1542951/1543117 are terminal; no Kindle service or render holder remains.

Root `/x/Temp/kindle-completed-value-20261106/bounded` is closed. Its complete
training/frozen logs, initial/final/frozen weights, replays and receipts remain
retained. Do not repeat this check or reuse its experience in a full learner.
A separately declared fresh fixed-budget comparison can now assess useful frozen
learning per minute against both the bonus-off reference and primary retained
recipe. No full learner, sweep, extension or automatic successor is selected.
Original 864/90% two-wall and the five-game goal remain unfinished.

## Full200k comparison selected

The [separate fixed protocol](../experiments/2026-11-06-cdp-completed-value-200k.md)
selects one fresh200k learner after bounded execution/activity/cost gates pass.
No bounded experience transfers. Compare the matching historical bonus-off
control and the stronger retained bonus-on primary, with four frozen24-natural-
episode arms, exact initial controls,100k recovery, replay and independent
collection. Relative to bonus-off, only the new boolean differs in effective
learning config; native/runner/guidance-wrapper identities differ as disclosed.
No gameplay result exists yet; preparation remains CPU-only. No extension,
second learner or automatic successor is selected.

The initial full CPU helper preflight fails before GPU launch: it assumes an
episode record precedes a learner report at the same action. The original
`helpers-execution.json` and `before-target-order-repair-util.txt` are retained.
A separate action-counter audit repair finishes only the unreached checks in
1.759393s (wrapper1.850254s), including four corrupted-activity cases and the
complete comparison entry. Passed helper stages are not repeated. This changes
no learner, target or native code. The original failure is not relabeled passed.
