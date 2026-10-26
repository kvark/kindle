# Six-action Pong at the compact B8 budget

**Active from16:42:36 UTC**, source`5043b7a`; startup checks pass. Selected
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
