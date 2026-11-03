# Conservative completed-return actor passes native and short training checks

## Fresh200k assessment is running

The [selected assessment](../experiments/2026-11-03-cdp-conservative-actor-200k.md)
launches **18:11:37 UTC November3**, sourcecf8c53fc, with308 input pins and a
30-minute training cap. MainPID1057479,invocation
`d9abb71b942440559ae740a11c7ee1bc`, service
`kindle-conservative-actor-200k-20261103`, runs one fresh200k/12,477-update
candidate followed by final/actual-initial sampled/greedy24-episode assessments.
The100k/6,227 recovery is saved and independently verified; the terminal
collector remains active. No full frozen result is available and the five-game
goal remains unfinished.

Startup verifies357 exact initial tensors,47 pre-update events and first
update376. An early read at18:12:51.734 finds11,768 actions/712 reports,
653 positive-target updates,first positive1224,and all308 pins unchanged.
The.333s startup check and.548s host review remain in actual wall time.
Root `/x/Temp/kindle-conservative-actor-20261103/full` is active; do not restart
it, repeat earlier stages or schedule another GPU job while it runs.

The recovery observer completes **18:22:27.826 UTC** in.330s. Independent CPU
verification completes **18:24:21.872** in.614s: counters100k/6,227,357 finite
tensors/51,502,659 elements, exact config/schema and metadata/file hashes,
read-only0444 files/0555 directory,all308 pins unchanged. MainPID/invocation
remain identical before and after. This retains weights and optimizer moments,
not replay, live games, beliefs or all RNG state. CPU co-host time remains in
actual wall time. At18:24:59 the learner reaches122,880 actions/7,657 updates.
Verification receipt SHA
`13e911dd3532670203774894235914404a47388e7f657f34aa3a0e07744fa7c8`;
recovery metadata SHA
`a49c406693ca12925adc5f4243c71832896dededae1e07de29fbd2ce5d176c0e`.

Actual CPU postprocessing passes on a separate retained-data fixture before
launch:200k exposure accounting, exact teacher metrics and curve generation.
Six teacher-data corruptions,five trace corruptions and three decision cases
pass; the exact queued CLI configuration,357-tensor schema and bounded target
accounting also pass. Helpers take1.868s and queue preflight1.097s. A private
copy-script count assertion fails before execution and remains recorded; the
copy is corrected before these passing checks. No learner/GPU stage repeats.

## Completed bounded qualification

The optional completed-return actor now works with the retained model
disagreement bonus. A fresh8192-action candidate takes **50.754s versus47.402s
(+7.07%)**, receives33,588 positive replay targets, and preserves the bonus.
Its padded200k projection is **22m43.035s**, below the30-minute cap.
Neither1024-action frozen arm completes a natural episode; gameplay benefit
remains unmeasured.

The [qualification protocol](../experiments/2026-11-03-cdp-conservative-actor.md)
follows the [CPU target audit](2026-11-03-cdp-completed-return-bonus-audit.md).
`replay_actor_observed_lower_bound=true` explicitly allows the existing
`clip((G_observed - V_combined)/return_scale,0,1)` target with active disagreement.
It is false and omitted by default. Without it, the prior rejection remains.
The CLI, native validation and log auditor reject incomplete or meaningless
combinations; capability checks cover fresh construction and restore.

For fixed value and positive scale, omitting nonnegative completed bonuses
gives a lower bound on their hypothetical actor weight. This is a conservative
threshold, not an extrinsic-advantage estimate, a true-value bound or an unbiased
policy gradient. The loss arithmetic, actor features, normalizer, optimizer and
parameter layout are unchanged. Existing terminal/mask rules remain, and this
experimental path still requires CPU target construction on AMD Auto.

| Fresh bounded arm | Actions / frames | Updates | Training time | Frozen1024 time |
| --- | ---: | ---: | ---: | ---: |
| Retained bonus-on control |8192 /32727|489|47.402s|.870s|
| Actor100, conservative mode |8192 /32722|489|50.754s|.865s|

Both use12M64/Auto/N8/B16/T16/H15/R16,encoder4e-5/dynamics4e-4/behavior8e-5,
constant expert100,critic1/1,disagreement1,seed1009,full18/sticky.25/repeat4,
native RGB and one GPU64 resize. Teacher and guidance are unchanged. No
experience is transferred between arms. Both first updates occur at376 actions.

The candidate has55,469 completed and33,588 positive replay rows, normalized
weight30,327.206370 and raw loss394.076030. Positives first appear at1224 actions
and occur in431 updates. These are repeated replay exposures, not unique
experiences. Both imagined and posterior disagreement means remain positive;
their sums across reports are10.029/10.066 control and10.041/10.059 candidate.
All489 loss coefficient checks pass per arm.

## Native qualification and compatibility

Native3f6c6648 passes164 core Rust and5 gym CPU tests, strict workspace and
binding Clippy, and1,018 packaged Python checks with10 excluded native
construction checks. The package test wrapper attempts zero GPU constructions.
Twelve guarded native F64 cases pass: expert absent/present, actor unimix0/.01,
and partial/zero/unit weights. They check actor-bias derivatives, zero remaining
derivatives and unchanged parameters under this explicitly enabled mode. This
is numerical qualification, not evidence of useful trained gradients.

The retained checkpoint reproduces128 semantic events over1024 frozen actions,
with all357 tensors unchanged. Both new initial checkpoints match all357
retained initial tensors; the two training arms share47 pre-update events.
CNN,reward,actor and value parameters change during training. Final/frozen
arrays remain equal, all357 tensors per arm are finite, and both full training
pixel/action/corrective-label replays pass. Actual overrides are3902/3749;
guided episode scores do not measure unaided competence.

A separate CPU comparison of the completed control finds **all357 final tensors
and1,065 transition/episode/reset events exactly equal** to the earlier8192-action
retained check. Only wall-clock fields are excluded from semantic comparison.
It completes17:58:41.844 UTC in1.124s. This strengthens bounded default-mode
compatibility with native447fe50, but does not prove full-life replay/RNG
equivalence or isolate a future historical-comparison effect.

## Execution and retained failures

All five bounded GPU stages pass once. Service
`kindle-conservative-actor-check-20261103`,PID1055780,invocation
`355a1feb3ac34fbf800cfd2757285193`, runs17:53:53–**17:56:08 UTC November3**.
Independent terminal collection completes **17:58:00.438** in1.187s, verifying
258 current/338 retained pins and215 preserved previous source files. The
transient unit is unloaded; its original invocation has all five completion
messages and the main completion marker, and no original PID or active Kindle
service remains. No successful GPU stage or prior closed experiment repeats.

CPU setup failures are retained: an initial CLI fixture omitted `--cdp`; an
unscoped tensor-count substitution altered dates in private harness paths and
failed two imports; a CLI preflight mock omitted the passive-label signature.
All are corrected before learner execution. The actual queued control and
candidate CLI commands then reach exactly their expected native configurations
without constructing an agent. Existing checkpoint/schema, report accounting
and semantic helpers pass against retained fixtures. Native production code
needs no correction after qualification. Brief host observations remain in
wall time; no timing subtraction or kernel-exclusive speed claim is made.

Runtime `/x/Temp/kindle-conservative-actor-20261103/runtime` contains native SHA
`ffb532c04bc5d3fb4b38a1b775d56127e12989d160e6fdf7ef9c071b7708e01c`.
Wheels,validation binary,build cache and previous source snapshots remain.
Bounded root `check` and its terminal collector are closed; do not rerun them.

The full assessment above is separately selected after this bounded closure.
Breakout and the five-game goal remain unfinished.

| Artifact | SHA256 |
| --- | --- |
| Build receipt |`aebedf9c60cb1d820fc69468078cb8368af63f1a739fcb96e0da80f9ea108462`|
| Native qualification |`4c3bee16aa0fa6a91920a6633ec54a8461539df3a4992b76c2ee5cd35dd4745c`|
| Bounded declaration |`a47ce8cca788750edabfb11995da10e98d1047fed63aa2928eb859b0b1537a1e`|
| Completion |`ddde4e0780aa0118e7da673e6663e91c1e96f9b5c6f78adb242b9ab1715f8129`|
| Terminal summary |`4d3ce2ba97f241d4206b9a3bee2c20e472c02a35d53ad8bd9bbfae8b00f4734c`|
| Retained-control comparison |`a05e96f04e49939f4b7bfc3c0263c0c8e40f273995d56efa7761d47aa69f1816`|
