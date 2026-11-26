# Completed replay-value targets: implementation and qualification

The [optional implementation](../experiments/2026-11-06-cdp-completed-value.md)
is packaged and natively qualified. Its [fresh200k comparison](#full200k-comparison)
scores **89.958 greedy /50.500 sampled in17m03.021s**. Point scores improve over
bonus-off65.208/46.458, but paired gain/min intervals span zero. The stronger
retained143.667/70.750 remains preferable under the declared criterion. Keep the
option off; original864/90% two-wall and the five-game goal remain unfinished.

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
This bounded declaration selected no full learner, sweep or extension. The
separately declared fresh comparison below is now complete; no bounded experience
was transferred and no automatic successor follows.
Original 864/90% two-wall and the five-game goal remain unfinished.

## Full200k comparison

The [fixed protocol](../experiments/2026-11-06-cdp-completed-value-200k.md) completes
one fresh seed1009 learner, with no bounded weights or experience transfer.
Compared with the historical bonus-off control, only `replay_value_completed`
changes in effective learning configuration. Native/runner/guidance-wrapper
versions differ; core teacher class ASTs match, and exact initial tensors and
trajectories are checked. Primary comparison additionally removes disagreement.
This is a development comparison, not an isolated timing or broad reliability claim.

| Recipe | Greedy | Sampled | Training time | Own-initial greedy gain/min |
|---|---:|---:|---:|---:|
| Retained primary, bonus on |143.667|70.750|21m27.910s|6.590132|
| Historical bonus off |65.208|46.458|17m05.486s|3.686056|
| Completed replay values, bonus off |89.958|50.500|17m03.021s|5.146521|
| Earlier completed-return actor, bonus off |95.375|56.542|17m12.959s|5.411638|

All final modes score0/24 two-wall. Candidate and bonus-off actual-initial
scores are2.208333 greedy/1.416667 sampled. These use the same reused cohort
4000001009, first three natural episodes per eight lanes, full18/sticky.25/
repeat4/noops0, guidance/labels/learning off. No comparison control is rerun.
The completed-return actor is a distinct historical objective, included as context.

Against bonus-off, greedy return improves24.750 and own-initial gain/min39.621%,
with6/8 improved lanes. However, paired eight-lane95% intervals are
[-1.250,53.584] for raw return and[-.064167,3.151568] for gain/min. Sampled return
improves4.042, gain/min9.236%, with5/8 improved lanes; intervals[-2.750,11.000]
and[-.154062,.651004] also span zero. These use10,000 lane resamples, seed0;
they do not measure training-seed reliability.

Against primary, greedy return falls53.708 and gain/min21.906%, despite20.567%
less training time. Raw interval[-92.125,-18.247]; gain/min interval
[-3.336835,.386041], with1/8 improved lanes. Sampled return falls20.250 and
gain/min10.876%; raw interval[-29.833,-10.292], gain/min[-.808061,.112065],
with3/8 improved gain/min lanes. The declared adoption criterion fails. Keep
primary and the completed-value option off; no coefficient sweep, extension,
second learner or automatic successor is selected.

Training completes **200,000 actions /799,584 frames /12,477 updates in
1023.021210s**, excluding2.227587s construction. The first update is at376;
read-only100k/6,227 recovery passes independent verification in.674414s while
the original learner remains active. All261 initial arrays match bounded and
bonus-off, and all47 pre-update events agree. Both weighted losses are checked;
maximum world-total reconstruction error9.715e-5. CNN/reward/continuation/actor/
critic parameters change, actor copies agree, and every frozen export preserves
all261 tensors. Initial sampled/greedy semantic traces match bonus-off across
799/794 events; only finite episode wall times and timestamps differ.

Completed targets replace **2,531,238 replay rows on12,422 updates**, first at
**action1,224**, matching the first true terminal. Counts by successive50k-action
windows are528,121 /662,002 /671,441 /669,674. These are repeated row exposures,
not unique states. Model-derived bonuses remain zero; the completed-return
actor is absent. Returns reflect the collecting actor/teacher mixture, which
limits their interpretation as current-policy values. The experiment measures
no new value-calibration or action-ranking diagnostic, so it cannot establish
why the point scores change.

Training stage times: act62.710s, environment61.921s, observe12.398s,
learn879.909s, reset1.128s and checkpoint.449s. Learning consumes86.011% of
measured training wall time, Atari stepping6.053%. Summed learner timings are
replay68.155s, posterior163.132s, imagination216.117s, world training347.038s,
replay refresh4.033s, world sync.923s, behavior training66.838s and behavior
sync.754s. These component timings describe this run; they are not an isolated
algorithmic speed comparison.

Full training replay passes in the110.227s training audit, including all pixels,
actions, corrective labels, overrides, rewards, frames and boundaries. Actual
action overrides are44.3545% under20% episode selection. Guided online returns
remain separate from frozen competence. Curves, complete exposure/score-range
accounting, videos and all checkpoint milestones remain outside Git.

| Frozen arm | Actions | Emulator frames | Wall time |
|---|---:|---:|---:|
| Final sampled |30,088|120,310|18.788s|
| Actual-initial sampled |5,944|23,719|4.073s|
| Final greedy |37,696|150,743|39.407s|
| Actual-initial greedy |5,936|23,706|3.897s|

Each arm has zero learner updates,24 selected natural episodes and zero selected
truncations, with complete replay/video. Frozen logs omit live pixel hashes;
replay therefore does not establish unrecorded live-image equality.

The initial full CPU helper preflight fails before GPU launch because it assumes
an episode record precedes a learner report at the same action. Original
`helpers-execution.json` and `before-target-order-repair-util.txt` are preserved.
A separate action-counter repair completes only the unreached checks in1.759393s
(wrapper1.850254s), including four corrupted-activity cases and the complete
comparison entry. Passed helper stages are not repeated; no learner, target or
native code changes. The failure is not relabeled passed. All five actual mocked
command paths subsequently pass in.994570s, with zero native/emulator construction.
Earlier packaged fixture failures and their separate repair also remain retained.

Execution source **393246191af4d6eea28a69a5fdca76b1f3347012**, unchanged native
**d0e7bb79/SHA3146e0dc**, root `/x/Temp/kindle-completed-value-20261106/full`.
Original controller1543615/invocationf0806cb2c90e47aaa5aa60c4541fb751 launches
19:57:02.775 UTC November6. Main completes20:18:20.849 and terminal collector
20:18:50.515. Independent CPU closure finishes **20:19:20.913 UTC** in2.123761s
(wrapper2.172284s), verifying425 active/338 retained pins,55 completion artifacts,
28 checkpoint files,384 candidate/reference episodes, all three paired
comparisons and target counts. Main/recovery/collector and all five GPU children
are terminal success; no render holder remains. No GPU stage fails or repeats.

This closes the allocation. The original864/90% two-wall and five-game goal
remain unfinished. The result adds evidence about completed-value supervision;
it does not reproduce SuperMimic or establish that model/critic calibration is
now sufficient for efficient policy learning.
