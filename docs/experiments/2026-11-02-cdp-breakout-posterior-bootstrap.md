# Breakout: observed-state bootstrap for replay value targets

Select one optional native target variant within the five-game goal. The
retained replay-value1 critic has weak sampled-return calibration; a fixed
scalar correction recovers only a small gain over a constant. This does not
identify the cause. Test whether replay targets benefit from bootstrapping
directly with values of observed posterior states, removing their extra
imagined-return contribution. This is a development hypothesis, not a bug fix.

The default remains `replay_value_bootstrap: imagined`, including old checkpoint
metadata with no field. `posterior` changes only the bootstrap array passed to
the real-replay lambda-return calculation. Observed external rewards, the
existing intrinsic bonus, real termination/boundary flags, discount1−1/333,
lambda.95, replay weighting, slow-value regularization and all imagined
actor/value targets stay unchanged. The world and behavior losses consume the
same revised replay targets. Parameter names/shapes and checkpoint format stay
unchanged. A fresh-only CLI option records the choice and rejects old runtimes
without support; restore retains the checkpoint configuration.

Implement both host and resident target paths; current AMD runs continue using
host targets. `KINDLE_DEVICE_TARGETS` remains unavailable. Offline shader
validation does not establish execution correctness on the unavailable path.
No BF16/F16, backend optimization or dependency change is selected.

Before any learner, require exact numerical continuing/terminal/timeout cases
for both target modes, old-config default and explicit-mode round trips,
invalid-mode rejection, fresh CLI forwarding, restore/evaluation rejection,
older-runtime capability rejection, Rust CPU checks, strict Clippy and relevant
packaged Python tests. The loss derivative is unchanged; do not repeat unrelated
gradient attribution or parity projects. Package a separate runtime and retain
its source, wheel, native and validation receipts.

Then require one guarded legacy-default1024-action frozen check against the
retained replay-value1 sampled trajectory: zero updates, original semantic
events,357 invariant tensors and compatible config with only the new default
field added. This checks the new runtime before a fresh learner. Never relabel
a mismatch as passed or repeat a successful stage.

The separately launched bounded candidate uses8192 fresh actions, seed1009,
12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4,
replay-value1/expert100, sampled training collection and the retained teacher.
Only the replay bootstrap changes. Preserve full18/sticky.25/repeat4/noops0,
native max-pooled RGB and one GPU64 resize. Require489 updates, finite reports,
357 matching initial tensors against the retained control, changed CNN/reward/
actor/value tensors, complete guided pixel/action/label replay and host guards.
Follow with1024 frozen actions and zero updates. Use the same established
conservative cost projection and a30-minute200k training cap; do not increase
the cap to pass. A bounded check measures execution and early cost, not task
competence.

No full learner is selected until the package, default compatibility and
bounded checks close. A later fresh comparison must retain200k actions,100k
recovery, final and actual-initial sampled/greedy24-episode assessments against
70.750/143.667 and the unchanged two-wall gate. Reused1009/cohort is development
evidence, not independent reliability. No extension, broad sweep or automatic
successor. Check live state before GPU work; do not overlap jobs or repeat
completed captures, learners or collectors.

Root `/x/Temp/kindle-replay-bootstrap-20261102`. Keep private build/controllers,
large data and checkpoints outside Git. The five-game goal remains unfinished.
