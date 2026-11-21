# Smaller Breakout exploration bonus at the fixed200k budget

**Complete: smaller bonus lowers frozen score 9.542→5.042**, both 0/24 two-wall
successes, at 200k actions/49,939 updates in 21m07s. All audits and comparison
pass after a CPU-only repair of the omitted replay manifest. The original
service failure is retained; no GPU work is repeated. Retain coefficient 1;
no extension or confirmation. See the
[result](../results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score).
The selection and startup records below describe the completed run.

Selected after completed PongH5 evaluation and comparison. H5 saves23.65%
training time but lowers frozen score14.958→8.667; the primary mean15 gate fails,
so its conditional confirmation remains unselected. Proceed with this fixed
Breakout test, keeping its prepared recipe and budget. No automatic next queue.

The completed four-action Breakout control scores9.542 versus initial1.167,
with0/24 two-wall successes, in20m59s. Its last12,500 updates have mean scaled
imagined bonus0.009418 and inferred game reward0.009768, a0.964 ratio. The full18
mixture control's ratio is0.225. These are observed minibatch means on different
trajectories, not a causal diagnosis. Native combined reward includes the scaled
bonus, so inferred game reward subtracts it. Actual observed rewards are
nonnegative; learned predictions need not be. Neither ratio measures gradient
contribution or establishes the best coefficient.

Reducing Pong's coefficient1→.01 improves its two completed frozen scores;
that motivates a bounded Breakout test, without predicting transfer. Keep the
existing four-action recipe and change only native intrinsic_reward_scale1→.01.
The ensemble remains active, with unchanged architecture and arithmetic; this
tests learning incentives rather than assuming an immediate runtime reduction.

Fresh1009/eval4000001009, centered CDP4M/RGB64/Auto/N8/B8/T16/H15/R32,
4,120,963 parameters, full BPTT/microbatch8, replay100k and original rates.
Keep native actor_unimix.01, minimal4 NOOP/FIRE/RIGHT/LEFT, sticky.25/repeat4,
noops0 and100000-frame episode limit. Native max-pooled frames receive one GPU
resize. No external action aid, RAM policy input, teacher or reward shaping.

Keep200k aggregate actions/49,939 updates, first248,6,392,192 sampled replay
positions,100k recovery at24,939 updates and30-minute training cap including
construction. Actual initial must match all346 coefficient1 control tensors.
Evaluate fixed final/own initial for the first three natural episodes per lane,
24 per arm, under600k-action/300s caps. Retain the original864-point two-wall/
90%-success gate, at least20 completed episodes and zero updates/truncations;
retain excess episodes and partial tails. Native policy mixture remains active.
Reused development seeds and the minimal protocol are explicitly labelled.

Audit all numeric reports, full config/source/counters, finite checkpoints,
346 unchanged frozen tensors per arm, complete CPU replays/videos and three
guards. Compare curves against the completed same-protocol control and check
semantic initial-trace equality. Audit1200s/service3660s. Retain any cutoff;
no cap increase, resume, extension or earlier-checkpoint substitution.

Runtime remains native`bfb8c31`, CLI`22bd61b`, replay`e853818` with no production
edits. Artifacts:`/x/Temp/kindle-cheap-20261025/breakout-4m-64-r32-minimal-unimix001-intrinsic001-200k`;
controller:`run_breakout_minimal_intrinsic001_200k.py`, SHA256
`18950938f7f8e6699ea87d0e900f6418534acd564f345e73c541dd0bd20d008a`;
queue`2f71a628173ff71ca0731d45ef6a5ece5ccb64a1e4f984e88b13a94d64d30071`;
prepared declaration`9187477073196e02422d05a37cbd0c843be26c7531b875fe0b79f9c396061990`.
Compilation, exact queue normalization and unselected refusal pass. The CPU
comparison reader validates the completed control and reproduces its prior
late-bonus analysis. Comparison SHA256
`1174a7a7857c2b236b36a37343c640e854bbb9c331ab05dc519eea3823342901`.
Postprocessing and comparison watchers are prepared; none is started.

## Started on AMD

Started October26 at13:22:20 UTC, source`66d1aed26c54e903b4ae039faedae4a46fcfd048`,
owner`kindle-cheap-breakout-minimal-intrinsic001-200k-20261026`, invocation
`35e56c1dfa2049b7b6f56cda07769a12`. Selection verifies completed Pong reviews,
comparison and terminal service, clean pushed source, host identity and exact
runtime/controller/queue/analysis hashes. Selected declaration SHA256
`8d90301a704e9c8fd34318e0752759a1e2dcf317ee52cae32ef2e9926e8ee3d2`.

Startup passes full native config equality except coefficient1→.01,4,120,963
parameters, fresh1009/no external aid, minimal4, native mixture.01/H15,
all346 finite initial tensors matching the control and numeric reports at
248:1/256:2/256:3. Initial metadata SHA256
`d058f39eeb342eb1d5713f8fbfc253ad8c4ef475caf85d6614fd1494bce66528`;
startup review`85e14c09c88c3675ee46131f8ea220e450851434b1cb79c96d44c86e6e05927e`.
Recovery, fixed-final/own-initial evaluation, full replay/guards, curves and
completed comparison are running or waiting automatically. Startup verifies
execution and initial equality; learning remains pending.

The100k recovery passes at24,939 updates,346 finite tensors and25,531 strict
reports through102,376 actions. Captured98,304-action progress is620.077s with
zero debt. Checkpoint metadata SHA256
`5c8374f85b10ebde1e6cae8bcd30daeb3486b512463e1fabdd69211922d9e37b`.
The recovery watcher exits successfully; training and fixed frozen result remain
pending. No budget or recipe change follows from this recovery check.
