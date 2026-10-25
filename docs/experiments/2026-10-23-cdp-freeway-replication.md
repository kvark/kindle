# Freeway: replicate the fixed exploration rescue

Training, not backend parity, is the priority. The fixed persistent-uniform
intervention rescues seed3019 from0 to32.96, but a selected failed-seed rescue
is not a reliable recipe. Train the remaining original seeds1009/2017 from
scratch with that identical intervention. No tuning between seeds.

Each gets500k actions/124,939 updates,12M/B8/T16/BF16/ratio32, centered CDP,
action-effects scale1, eight environments and batched inference. On5% of fixed
16-action blocks, hold one uniformly sampled action from all18. No game-specific
hint or reward shaping. Keep the native binary and all other configuration
unchanged. Compare actual initial tensors against each same-seed unassisted run.

After each run, freeze and evaluate final and actual-initial policies on the
existing fixed24-natural-round cohort, without exploration overrides. Verify
all346 tensors unchanged, action/replay accounting, finite checkpoints and whole
video replays. Fixed200k-action/1800s cap per frozen arm; incomplete cohorts stay
incomplete. Training has a7200s cap per seed; expected about38 minutes each.
The controller stops on faults or audit failure and never extends a budget.
Workers12GiB/no swap, CPU review2GiB/one CPU; host-only monitoring records
allocation warnings. No GPU reset or backend rebuild.

Report both new seeds plus the retained3019 intervention separately from the
unassisted cohort. The intervention was selected using3019 and these are reused
development seeds, not an untouched final test. Intermediate success requires
mean>=25 and at least90% of rounds>=25 for every seed. The published33.40
reference and full five-game supremacy are not waived by this gate.

Run sequentially after the current Breakout state capture, not concurrently.
Artifacts: `runs/cdp-freeway-persistent-replication-20261023.mr2oPmJq`.
Service: `kindle-cdp-freeway-persistent-replication-20261023`.
Two learners total1M actions/249,878 updates; no other learner queued.
