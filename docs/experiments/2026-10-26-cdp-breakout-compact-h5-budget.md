# More compact Breakout experience within a fixed wall budget

The compact4M/RGB64/H15/R32 policy with1% native mixture scores7.083 after200k
actions in22m29s, with0/24 two-wall successes. The mixture greatly reduces FIRE
stalls but does not establish competent play. Test a fresh H5 life with500k
actions under a45-minute training cap: spend less work on each imagined rollout
and allow more real experience and updates within a bounded allocation.

This is a combined horizon/budget recipe test. The earlier retained12M H5 warm
start regressed67.583→27.208 after50k new actions, so shorter imagination is not
an established learning improvement. This test uses fresh compact weights,
replay, games, belief and RNG; it is neither that lineage's continuation nor an
unchanged extension of a completed run. Do not attribute any change to H5 alone.

Keep seed1009/evaluation4000001009, centered CDP/action-effects1, Size4M/RGB64/
Auto,4,131,729 parameters, N8/B8/T16/R32, full BPTT/microbatch8, replay100k and
actor_unimix.01. Relative to the completed compact mixture control, the native
configuration changes only imagination_length15→5. Discount horizon remains333.
Preserve full18/sticky.25/repeat4, native max-pooled frames and one GPU resize.
The1% native mixture remains in frozen sampling. No external action aid, pixel
controller, RAM input, reward shaping, native rebuild or optimizer change.

Save the actual zero-experience checkpoint and compare its346 tensors with the
compact mixture control at startup. Keep this life's own frozen initial control
regardless of the old control's score. Starting learner/environment counters are
zero. CLI`1ed07af` already exposes fresh imagination length; training source and
native`bfb8c31` remain unchanged.

Fixed500,000 aggregate actions, roughly2M emulator frames,124,939 expected
updates and first update248. The2700-second training cap includes construction.
Save100k/200k/300k/400k recovery milestones by replacing the latest checkpoint,
then evaluate fixed500k final. Record a cutoff if the cap prevents completion;
retain the latest checkpoint without increasing the cap, resuming automatically
or selecting an earlier favorable policy. The200k control is a cost reference,
not an extrapolated500k score or runtime guarantee.

Frozen final/actual-initial controls use the first three natural sampled
episodes per lane,24 per arm, zero updates/truncations,600k-action/300-second
caps. Retain excess episodes and partial tails. Preserve the existing two-wall
864-point task definition, at least20 episodes and90% success fraction; score
improvement alone does not pass this gate. Seed1009 is reused development data.

Audit every numeric report, complete source/config/counters, finite checkpoints,
all346 unchanged frozen tensors, full CPU replay/videos and host guards. Audit
cap1200 seconds; service cap4560 seconds. Stop on real faults/numerical failures.
No automatic extension or broad seed queue follows.

Selected after the six-action Qbert trial finishes605.208 with0/24 pyramids,
lower score/coverage than its full18 control despite18m21s runtime. All audits
pass and its service exits. The separate CPU Breakout pixel follower scores
11.333 with0/24 successes and exact replay; it is too weak to justify training
guidance. Both negative results are retained.

Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-4m-64-h5-r32-unimix001-500k`.
Controller: `run_breakout_compact_h5_500k.py`; intended service:
`kindle-cheap-breakout-h5-unimix-500k-20261026`. Exact commands, caps and control
hashes are declared. CPU preparation verifies the command changes are only H5,
training budget/cap and artifact paths. Startup and learning results follow.
