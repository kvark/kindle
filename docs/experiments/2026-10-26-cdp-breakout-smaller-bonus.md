# Smaller Breakout exploration bonus at the fixed200k budget

**Prepared, not selected or running.** Review the active PongH5 fixed-final
result, comparison and terminal service before selecting any next allocation.
No automatic queue follows from this preparation.

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
