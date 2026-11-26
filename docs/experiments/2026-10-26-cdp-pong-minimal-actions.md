# Six-action Pong at the compact B8 budget

**Completed; retain full18.** Six actions finish500k/124,939 updates in44m21s,
11.17% less training time, but frozen10.375 is below full18's14.958; both24/24
wins and below mean15. All audits, recoveries, replays, guards and comparison
pass; service exits17:28:27 UTC. No extension or conditional4013 confirmation.
See [completed result](../results/2026-10-25-cdp-cheap-pong.md#six-actions-save-time-but-lower-frozen-quality).

Started16:42:36 UTC, source`5043b7a`; startup checks pass. Selected
after the completed Qbert H30 review. H30 costs35.34% more training
time and lowers frozen score/coverage; retain H15 without extension. Pong's
best compact B8/H15/coefficient.01 learners score14.958 and14.0, both24/24 wins,
but miss the mean15 gate. B16 and H5 save time but lower frozen quality; doubled
B16 learning rates also hurt. The interrupted R40 life is not resumed.

Test Pong's native six-action vocabulary: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE,
LEFTFIRE. This may reduce redundant choices and action-dependent model work.
Breakout's smaller vocabulary reduced wall time and improved its weak score;
Qbert's reduced wall time but hurt coverage. Neither predicts Pong's outcome.

Keep fresh1009/eval4000001009, centered CDP4M/RGB48/Auto/N8/B8/T16/H15/R32,
full BPTT/microbatch8, replay100k, coefficient.01, actor mixture0 and original
rates4e-5/6e-6/4e-4. Only native action_count18→6 and the corresponding
published→published-minimal protocol change. Sticky.25/repeat4/noops0 and
100000-frame natural episodes remain. Native max-pooled frames receive one GPU
resize. No action aid, teacher, RAM policy input or reward shaping.

Actual parameters3,568,997 versus3,578,225. Dimensions, initialization and
policy probabilities change, so there is no matched-initial-weight claim.
Keep own actual-initial controls. The existing six-action4M48 native shape has
completed Qbert learning and frozen evaluation; no native rebuild or additional
GPU qualification is needed. That evidence establishes execution, not Pong skill.

Fixed500k aggregate actions,124,939 updates, first248 and15,992,192 sampled
replay positions. Keep the full18 control's55-minute training cap, including
construction. Actual command`--checkpoint-every 100000` must match declared
100k/200k/300k/400k recoveries and500k final, with24939/49939/74939/99939/124939
updates and zero debt. Both preparation and controller startup check this
explicitly, following Qbert H30's missed recovery. No cap increase, resumed
learning, extension, earlier-checkpoint selection or automatic next allocation.

Evaluate fixed final and own initial on the first three natural sampled matches
per lane,24 per arm, under600k-action/300s caps. Retain mean15/90%-wins,
at least20 natural matches, zero truncations/updates, excess matches and partial
tails. Check all numeric reports, full config/source/counters, finite checkpoints,
346 unchanged frozen tensors per arm, complete CPU replays/videos and three
host guards. Plot action/time learning curves and compare with completed full18.
Audit1200s/service5160s; monitoring has no authority to extend or retry training.

Same RX7900XT/RADV/Mesa26.0.8/native binary as the control; kernel7.0.0-34
versus7.0.0-38 and a changed AMD module remain confounds. This is one reused
development seed; a useful result would require focused confirmation before a
reliability claim. Small parameter-count savings do not predict end-to-end speed.

CPU replay now explicitly accepts Pong's six actions and verifies their exact
order/count in both log and manifest. Tests exercise real full/minimal Pong
trajectories, native/64 pixels, sticky0/.25, natural boundaries/tails, match
scoring and rejection of mismatched declarations/protocol versions. The initial
test run has173 passes and8 failures caused by a wrong test assertion field
after replay passed. Correcting that field and bounding Pong fixtures at one
natural match per lane gives30/30 Pong checks passing;151 other checks already
pass and their paths are unchanged. Native runtime and training CLI are unchanged.
Validation:`/x/Temp/kindle-cheap-20261025/pong-minimal-replay-support/validation.json`,
SHA256`b031d8a30597b74789104e23d826546396e92088c222e0c73e2fde8b23c6c623`.
The original replay script and failed test output remain in that directory.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-minimal-intrinsic001-500k`.
Controller`run_pong_minimal_intrinsic001_500k.py`, SHA256
`77c623edb40a8f5f8c1235a7d955565444fbd44e394e8a4c6e09ea8caf69aba9`;
queue`610a05d69eef1fcf2a2f531e72b4a85407780aff6093fe83e08067e1265fe1b1`;
prepared declaration`a1d5d264d4facd0812d4f756b01e889479b01351a4664841707c0c2b8b4a7801`;
manifest`f72a3754402d933279c76b5c77907ac7b474c483b8c8c5ac37e7154d87ef0363`.
Runtime remains native`bfb8c31`/CLI`22bd61b`; CPU replay SHA256
`0109aeef4c8f6a347353bdac0e57656d017f869f6b4869a196d2d415880e74e4`.
Preparation verifies normalized queue differences, actual checkpoint schedule,
manifest/runtime pins, completed-control comparison reader and unselected refusal.
Selection verifies completed Qbert audits and its retained protocol deviation,
CPU replay validation, exact runtime/script/manifest pins, actual checkpoint
schedule, current host and no concurrent learner, with clean pushed source.
Selected declaration SHA256
`1e66261fdf0a53dcbf3d0f6f2a71541869b71cc62e643811655f26e7ba8f5290`.

Started16:42:36 UTC/source`5043b7aac7d29ebc76290fcf84bb223db7524de4`, owner
`kindle-cheap-pong-minimal-intrinsic001-500k-20261026`, invocation
`c0b42deafb764ce5b8f52ace4e08e33b`. Startup passes full native config equality
except action_count18→6,3,568,997 parameters, the declared action names,
346 finite own initial tensors and first updates248:1/256:2/256:3. No initial-
weight equality to full18 is claimed. Initial metadata SHA256
`480c713b07b57e832d718773969ab9cf93f1ccfaeb8c9bf8ab43e1fd0b8b8b27`;
startup review`6fe25b478c767da9568bcf51af2effda4087d19af5cf36d70e260584ce97f555`.
Recovery monitoring, automatic fixed-final/initial evaluation, CPU replay/video,
curves and completed comparison are attached. Frozen quality remains pending.

## Early measured cost

The first16,384 actions/4,035 updates take83.943s versus94.054s for the full18
control, **10.75% less wall time**. Emulator frames65,525 versus65,524;
construction1.949 versus2.140s is separate. Both have zero debt. All prefix
reports/counters pass and the candidate's exact byte prefix is stable on a
second read. The complete config differs only in action_count, while the
documented kernel/module confound remains. This is early cost evidence, not
full-budget speed or frozen learning; no budget or evaluation changes.
Review:`prefix-cost-review.json`, SHA256
`8d465502b9f3a76a9ac24878f90abf4e509080d8950c017d4a96c6f6b0b9d3ba`.

## 100k recovery

The100k checkpoint passes at24,939 updates, with all346 tensors finite and full
config/counter checks. The stable complete log prefix contains105,792 actions
and26,385 finite reports; latest102,400-action progress is541.144s/25,539 updates,
zero debt. Metadata SHA256
`d086d92a02f0da72c1f5e731b94d2663982253451bd790e1238ba6d48ab9a4a2`.
This verifies recoverability and recorded learning work, not frozen competence.

## 200k and300k recoveries

Both checkpoints pass full configuration/counter checks and all346 finite
tensors:200k/49,939 updates and300k/74,939 updates. The300k review verifies a
stable303,120-action prefix with75,719 finite learner reports. Latest progress
there is303,104 actions/75,715 updates in1610.434s, zero debt. The run remains
active; fixed-final frozen quality is pending and its budget is unchanged.

Metadata SHA256:200k
`51aee6fa9cd0087afd6c5a583ff28236184e4459a26499f8efff178b63946629`,
300k`1d8bec54ac34ea1ed013d8b433b4f523b8c3f86bb4c7b80bb8b5542e4b58df05`.
Review SHA256:200k
`c5e353540f1e3ab1eeb7e32fc7bc729bcbbb4aff026728fbda19a584dd2354ab`,
300k`c2a9c6b82b090b0788e19e66a5d3d81de1cd79ddda9c13a42bbb85ebe897e5c8`.

The400k recovery also passes at99,939 updates with346 finite tensors. Its stable
prefix contains404,584 actions/101,083 finite reports;401,408-action progress
is2134.646s/100,291 updates, zero debt. Metadata SHA256
`0d336351bed9367efea9600f1ada94b6a90823875a42dfce326c21ccf30c3c33`.
All four recovery reviews are complete and their watcher exits successfully;
training and final-result monitoring remain active.

## Conditional second-seed preparation

**Not selected after the completed primary result:**10.375 fails the original
mean15 gate and is below full18's14.958. The original preparation is retained;
`selection-decision.json` records this decision. No process or GPU work starts.

While the primary remains active, a fresh4013/eval4000004013 configuration is
prepared outside Git, **unselected**. It changes only learner/evaluation seeds
and output paths relative to six-action1009: same500k/124,939 updates,4M/RGB48,
Auto/N8/B8/T16/H15/R32, coefficient.01/actor0, original rates, no aid, four100k
recoveries and55-minute training cap. Fixed final/own initial evaluation keeps
24 natural matches, the mean15/90%-wins gate and all existing audits/replays.

Selection requires the primary's completed audits, recoveries, replays, guards,
curves/comparison and terminal service, **plus a passing primary frozen gate**.
Its outcome must be reviewed before a separate allocation decision. The
controller refuses unselected execution and requires the primary gate pass.
There is no automatic launch or extension. A primary failure does not activate
this confirmation. Seed4013 was used in earlier full18 development; it is not
held out or broad reliability evidence.

CPU preparation checks normalized queue equality apart from paths/seeds, all
actual checkpoint intervals and3300/300/300-second job caps. A separate
explicit-seed comparison reader validates the completed4013 full18 control
(14.000,24/24 wins), rejects a wrong seed, and leaves active pinned helpers
unchanged. The prepared comparison includes both same-seed full18/minimal6
results and the two fresh six-action learners with their own initial controls;
it does not assume equal initial weights across vocabularies or seeds. The
full18 control's older kernel remains a comparison confound.

Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-r32-minimal-intrinsic001-500k-seed4013`.
Prepared source`3f359ad`; controller`run_pong_minimal_intrinsic001_500k_4013.py`,
SHA256`7f114c3825b49672b6b3640389e01d7620a124876140d49f3ad713fa52714c4f`;
queue`94cbe0f8c1b72705d85a7c53d5b88b412795c1a53ba767e786549e74e8dff613`;
unselected declaration`69d634bd9039c4e95eb275c14be3708ccb8351743f78240cc03bda59e7d996aa`;
CPU review`83aeb8c24093b6d970ce48b883adec0928b9e660697e5b270528cc2ea0884a22`.
No production or active-run file changes and no new GPU work.
