# Breakout action rankings on shared retained histories

**Complete:** both full captures finish74,032 actions/296,076 source frames in
115.848s,zero updates. Only8/128 origins distinguish the candidates; neither
model demonstrates gain over uniform. All original audits andservices pass,
withno repeated stage. Main succeeds00:18:33 UTC November2,collector completion
00:18:46. See the [result](../results/2026-11-02-cdp-breakout-current-ranking.md).
Preparation andactive-launch statements below arehistorical; do not rerun.

Select CPU preparation within the five-game goal. The completed critic capture
finds weak sampled value calibration, but does not measure alternative-action
ranking. No learner or loss change follows directly from that observation.

Use the retained encoder4e-5/dynamics4e-4 greedy trajectory: 37,016 actions,
148,038 frames, learner1009/evaluation4000001009, score104.667, zero two-wall
successes. Both the4e-4 and1e-4 trained checkpoints receive exactly these raw
histories. They share the same architecture, learner seed and12,477 updates;
only the declared dynamics learning rate may differ in their configs. Their
learned weights differ. This is a shared-input diagnostic, not a causal effect
on policy performance or a symmetric sample of both policies' states.

Keep the existing native447fe50/SHA35ad0db runtime at
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No build or training.
Extend only the Python probe's explicit config comparison and origin selection.
Preserve its existing default Qbert phase-balanced behavior and strict config
checks. Breakout's two-wall milestone does not supply two populated phases.

Fix eight evenly spaced valid origins in each of the second and third natural
episodes per lane: 128 origins across eight lanes. Eligible indices are on the
existing16-action grid and their15-step continuations cannot cross a reset.
Select without rewards, labels, policy disagreement or branch outcomes. No
phase-stratified metrics. Do not replace uninformative origins after the oracle.

At every origin, compare18 candidates: hold each legal action for four policy
steps, then use the recorded continuation through15 steps. The exact recorded
15-action plan is a separate parity control. The CPU oracle forks each state
to preserve sticky-action memory; validate the full source trajectory, every
recorded-plan branch, unchanged parent state and one reversed candidate order
per lane. Record pooled-pixel hashes for all eight source histories. Generate
the successful oracle only once, in a process with no GPU context.

For each model, first run a128-action bounded capture (120s cap), then one full
37,016-action capture (600s cap). Require357 invariant tensors, finite forecasts,
matching source pixels/transitions/counters/config/native identity, repeated
recorded-plan forecasts, unchanged live beliefs/probabilities/counters, zero
updates and host guards. Scalar re-observation differs from the original vector
actor's posterior draws; no exact original-belief claim. Diagnostic draws are
paired among plans; repeated forecasts are not independent posterior samples.

Rank predicted external rewards using preceding predicted continuations, which
already include discount. Compare with actual gamma-discounted emulator rewards,
gamma=F32(1−1/333). Use existing tie-aware pairwise agreement, actual return of
the selected plan, uniform/best/recorded baselines, regret and forecast MAE.
Count informative origins and unequal pairs explicitly. Report model differences
and5,000-resample intervals over eight lanes with fixed seed20261029. These
intervals condition on reused lanes and one paired emulator future per plan.

The primary score omits intrinsic reward and terminal critic bootstrap. It
measures short external-reward ranking, not the complete actor objective or
long-term survival. A recorded continuation may be unsuitable after a changed
action. The scalar actor's preferred first action interpreted as a four-step
hold is descriptive, not a replacement policy's gameplay result. Do not claim
that accurate reward forecasts establish useful decisions or that this probe
isolates the cause of the two-wall failure.

CPU qualification checks both original learner closures and227/272 pins, the
completed critic's73 pins, both357-tensor schemas, four actual CLI commands,
config/origin/forecast-ledger tests and private controller bindings. Preserve
any failures without repeating successful stages. Before separate GPU selection,
require a clean committed worktree, matching idle host and closed original
critic services; pin protocols, source copies, queues, runtimes, checkpoints,
logs, oracle and dependencies. Main/collector caps2400/2700s.

Root `/x/Temp/kindle-breakout-current-ranking-20261102`; proposed main
`kindle-breakout-current-ranking-20261102` and collector
`kindle-breakout-current-ranking-collection-20261102`. This boundary selects
CPU preparation only. No learner, retry, cap increase, origin replacement,
extension or automatic successor. The five-game goal remains unfinished.


## CPU qualification and separate frozen selection

CPU preparation passes00:15:02.990 UTC November2/source3d31ccd in25.071s
(25.120s wrapper). It checks four actual commands, both357-tensor schemas,
227/272 learner pins and73 preceding critic pins. The single CPU oracle takes
21.987s for37,016 source actions and153,132 branch frames, with128 recorded-plan
matches andeight reverse-order checks. No native module isloaded. The66-input
preflight SHA is`39702dc199625fff5706263b69ebef7aca88e6f161ba998ca66028cf84dd8c77`;
oracle SHA is`27ed4a178b74478cf9c94f67a108e0e72a7504fd55425fcc13145e7ceaac1f00`.

All66 origin/config/ranking/forecast-ledger tests pass in.09s. The first run
has65 passes andone failure because the new valid-rate fixture omitted required
`observation_kind`; the fixture iscorrected, with production validator unchanged.
The original source andfailure receipt remain in`test-fixture-failure`. A separate
controller fixture incorrectly forbids importingALE, which the adapter imports
without constructing a game. Its failure isretained; corrected review blocks
construction andpasses ten direct-global screens andsynthetic ranking arithmetic,
withzero construction attempts. No emulator oracle orGPU stage isrepeated.

Select the fixed bounded pair andfull pair after original critic closure and
another idle-host check. The worktree iscommitted before selection; all inputs
are pinned before launch. Caps,origins,candidates andanalysis remain asdeclared.
No learner,loss change,retry orautomatic successor. Frozen capture launches00:16:26 UTC November2/sourcefc5032a with71 pins.
MainPID871905/invocation9ed47a38d0d342ac836ed4b91ee019a1 and original
collectorPID871908/invocation8299622f9c054bf3afc5bdb3f4a5bff9 areactive.
No full ranking result isavailable. Do not restart oroverlap another GPU job.
Declaration SHA256
`a94e57ecea32816c2dd0b822178220533bf76d75cb026c28eeae288412dc50c2`.
