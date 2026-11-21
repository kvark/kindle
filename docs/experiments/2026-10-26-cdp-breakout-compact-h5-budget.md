# More compact Breakout experience within a fixed wall budget

**Result:** the45-minute cap stops training at487,736 actions/121,873 updates.
The declared separate frozen assessment scores8.458 versus own initial1.375,
0/24 two-wall successes. The500k target remains incomplete; numerical,
checkpoint/accounting and frozen replay checks pass. See the
[result](../results/2026-10-26-cdp-breakout-compact-h5-budget.md). No extension.

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
Controller: `run_breakout_compact_h5_500k.py`; service:
`kindle-cheap-breakout-h5-unimix-500k-20261026`. Exact commands, caps and control
hashes are declared. CPU preparation verifies the command changes are only H5,
training budget/cap and artifact paths.

Started05:41:58 UTC, source`8910efe`. Startup confirms complete native config
changes only H15→5,4,131,729 parameters, zero starting counters, no external aid,
finite346-tensor initial checkpoint matching the control exactly, and first
updates248:1,256:2,256:3. Initial metadata SHA256
`7f95379c05f7caff754f9f34040308fae47084b51e4f75eb457f2fbf19538c61`.
Selected declaration SHA256
`057296c3ef2466cae098bad796bb17443aded4304a39c85cb8ee463762a3e9f6`.
Recovery-checkpoint audits and completed-run plotting are monitored automatically.
The later cutoff and separate frozen assessment are recorded in the result;
finite startup alone did not establish a learning gain.

At the shared 196,608-action/49,091-update prefix, H5 takes **1,086.206s
(18m06s)** versus H15's **1,325.662s (22m06s)**: 18.06% less wall time, or
1.220× the action throughput. Same native binary, GPU, model, resolution,
initial tensors and update count; the imagined horizon and learned trajectories
differ. This is a single-seed recipe-cost observation, not a hardware speedup
or evidence of improved play. The planned total budgets also differ, so no
full-run runtime or frozen-score extrapolation follows.

Both 100k/200k recovery checkpoints pass; the 200k checkpoint contains 49,939
updates and finite346 tensors. The prefix comparison independently checks all
49,091 reports and action/update counters in each prefix, pins their exact byte
lengths/hashes, and rechecks all346 matching initial tensors. Artifact:
`prefix-cost-comparison-196608.json`, SHA256
`30a240423ffeb71688cdb4829717ba7ef848c39196eba890d4458fc033dd2324`.
The fixed45-minute cap remains unchanged; the later cutoff is retained.

## Conditional assessment if the wall cap interrupts training

Prepared while training is still active, before any cutoff or frozen result:
if the 2700s cap interrupts the run, retain that failed budget outcome and audit
the exact graceful-stop checkpoint. Require terminal service state, a recorded
time-budget stop with no fault/validation event, a clean worker exit, complete
action/update accounting and finite weights. The original host guard remains
failed for exceeding time; this audit does not turn a cutoff into completion.

After that review, assess the exact cutoff weights and this life's actual
initial weights in a separate frozen phase. Keep the originally declared
24-natural-episode selection, seeds, protocol, task gate and per-arm
600k-action/300s caps. Preserve all346 tensors and source file hashes, plus full
CPU replay/videos. No new training actions, updates, changed configuration,
earlier-policy selection, cap increase or resumed learning. Label every result
with the actual training budget; it is not the planned500k endpoint.

This conditional preparation changes only evaluation output paths. The
unselected controller refuses before GPU/log artifacts. Selection requires the
completed cutoff review, terminal preceding service and exact source hashes.
Artifacts: `breakout-4m-64-h5-r32-unimix001-500k-cutoff-frozen` under the same
runtime root; controller `run_breakout_h5_cutoff_frozen.py`, intended service
`kindle-cheap-breakout-h5-cutoff-frozen-20261026`. Audit cap1200s and separate
frozen-service cap1860s. Selected after the cutoff audit; started06:28:51 UTC,
source`bd652bc`. The assessment completes8.458/1.375 with0/24 successes and all
frozen checks passing. It performs zero further learning and exits successfully.
